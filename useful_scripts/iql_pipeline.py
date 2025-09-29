#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Robotic Offline RL with IQL (robust H5 dataset support, best model saving)
"""

import os, random, argparse
import h5py, numpy as np
from typing import Dict, List
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from r3m import load_r3m
import copy

# ===================== Utils =====================

def set_seed(seed:int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


# ===================== Dataset =====================
class H5TrajectoryDataset(Dataset):
    def __init__(self,h5_path:str,demo_names:List[str],use_modalities:Dict[str,bool],image_size:int=128):
        super().__init__()
        self.h5_path=h5_path
        self.demo_names=demo_names
        self.use_modalities=use_modalities

        self.img_tf=transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((image_size,image_size),antialias=True)
        ])

        self.index=[]
        with h5py.File(self.h5_path,'r') as f:
            for demo in demo_names:
                T=f['data_frames'][demo]['actions'].shape[0]
                for t in range(T-1): 
                    self.index.append((demo,t))

    def __len__(self): 
        return len(self.index)

    def _get_obs(self,demo,t):
        obs={}
        obs_group=demo['observations']
        if self.use_modalities.get('agentview_rgb',True):
            obs['agentview_rgb']=self.img_tf(obs_group['agentview_rgb'][t])
        if self.use_modalities.get('hand_camera_rgb',True):
            obs['hand_camera_rgb']=self.img_tf(obs_group['hand_camera_rgb'][t])

        vec_keys=['ee_poses','hole_poses','peg_poses','robot_joint_states',
                  'robot_measured_joint_forces','gripper_joint_states','peg_hole_forces']
        for k in vec_keys:
            if self.use_modalities.get(k,True):
                if k in demo:
                    obs[k]=torch.tensor(np.array(demo[k][t])).float()
                elif k in obs_group:
                    obs[k]=torch.tensor(np.array(obs_group[k][t])).float()
        return obs

    def _concat_obs(self,obs):
        vec_keys=['ee_poses','hole_poses','peg_poses','robot_joint_states',
                  'robot_measured_joint_forces','gripper_joint_states','peg_hole_forces']
        vecs=[]
        for k in vec_keys:
            if k in obs:
                v=obs[k]
                v=v.unsqueeze(0) if v.ndim==0 else v
                vecs.append(v)
        vec=torch.cat(vecs,-1) if vecs else torch.zeros(0)
        imgs={k:obs[k] for k in ['agentview_rgb','hand_camera_rgb'] if k in obs}
        return vec,imgs

    def __getitem__(self,idx):
        demo_name,t=self.index[idx]
        with h5py.File(self.h5_path,'r') as f:
            demo=f['data_frames'][demo_name]
            obs=self._get_obs(demo,t)
            obs2=self._get_obs(demo,t+1)
            vec,imgs=self._concat_obs(obs)
            vec2,imgs2=self._concat_obs(obs2)
            done=torch.tensor([1.0 if t+1==demo['actions'].shape[0]-1 else 0.0])
             # action at t
            action = torch.from_numpy(np.array(demo['actions'][t])).float()
            # reward
            if 'rewards' in demo:
                r_t = float(np.array(demo['rewards'][t]))
            elif self.reward_hook is not None:
                r_t = float(self.reward_hook.compute(demo, t))
            else:
                r_t = 0.0
            reward = torch.tensor([r_t], dtype=torch.float32)

        return {
            'vec':vec,
            'imgs':imgs,
            'next_vec':vec2,
            'next_imgs':imgs2,
            'done':done,
            'action':action,
            'reward':reward,
        }


# ===================== Models =====================

class SimpleCNN(nn.Module):
    def __init__(self,in_ch=3,out_dim=512):
        super().__init__()
        self.net=nn.Sequential(
            nn.Conv2d(in_ch,32,7,2,3),nn.ReLU(),
            nn.Conv2d(32,64,5,2,2),nn.ReLU(),
            nn.Conv2d(64,128,3,2,1),nn.ReLU(),
            nn.Conv2d(128,256,3,2,1),nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.proj=nn.Linear(256,out_dim)
    def forward(self,x): return self.proj(torch.flatten(self.net(x),1))

class VisualEncoder(nn.Module):
    def __init__(self,r3m:bool,out_dim=512,freeze=True):
        super().__init__()
        if r3m:
            try:
                backbone=load_r3m("resnet50"); backbone.eval()
                self.encoder=backbone; self.proj=nn.Linear(2048,out_dim)
                if freeze: [setattr(p,"requires_grad",False) for p in backbone.parameters()]
            except Exception as e:
                print("[WARN] R3M failed:",e); self.encoder=SimpleCNN(3,out_dim); self.proj=None
        else: self.encoder=SimpleCNN(3,out_dim); self.proj=None
    def forward(self,img): h=self.encoder(img); return self.proj(h) if self.proj else h

class MLP(nn.Module):
    def __init__(self,in_dim,out_dim,hidden=(512,512)):
        super().__init__(); layers=[]; last=in_dim
        for h in hidden: layers+=[nn.Linear(last,h),nn.ReLU()]; last=h
        layers.append(nn.Linear(last,out_dim)); self.net=nn.Sequential(*layers)
    def forward(self,x): return self.net(x)

class Actor(nn.Module):
    def __init__(self,in_dim,act_dim): super().__init__(); self.mu=MLP(in_dim,act_dim)
    def forward(self,x): return self.mu(x)

class Critic(nn.Module):
    def __init__(self,in_dim,act_dim): super().__init__(); self.q1=MLP(in_dim+act_dim,1); self.q2=MLP(in_dim+act_dim,1)
    def forward(self,x,a): xu=torch.cat([x,a],-1); return self.q1(xu),self.q2(xu)

class ValueNet(nn.Module):
    def __init__(self,in_dim): super().__init__(); self.v=MLP(in_dim,1)
    def forward(self,x): return self.v(x)


# ===================== IQL Agent =====================
# 增加 target_value 与 soft_update 更新 
class IQLAgent(nn.Module):
    def __init__(self,obs_vec_dim,act_dim,img_keys,r3m=True,encoder_out_dim=512,repr_dim=512,
                 beta=3.0,expectile=0.7,gamma=0.99,tau=0.005, freeze_encoder=True):
        super().__init__()
        self.img_keys=img_keys
        self.gamma=gamma; self.tau=tau; self.beta=beta; self.expectile=expectile
        self.encoders=nn.ModuleDict({k:VisualEncoder(r3m,encoder_out_dim) for k in img_keys})
        in_repr=obs_vec_dim + encoder_out_dim * max(1,len(img_keys))
        self.repr_mlp=MLP(in_repr,repr_dim)
        self.actor=Actor(repr_dim,act_dim)
        self.critic=Critic(repr_dim,act_dim)
        self.value=ValueNet(repr_dim)
        self.target_critic=Critic(repr_dim,act_dim)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.target_value=ValueNet(repr_dim)
        self.target_value.load_state_dict(self.value.state_dict())
        [setattr(p,"requires_grad",False) for p in self.target_critic.parameters()]
        [setattr(p,"requires_grad",False) for p in self.target_value.parameters()]

        if freeze_encoder:
            for enc in self.encoders.values():
                for p in enc.parameters():
                    p.requires_grad = False

    def encode_imgs(self, imgs):
        if not imgs:
            return None
        parts=[]
        any_present = next((v for v in imgs.values()), None)
        B = any_present.shape[0] if any_present is not None else None
        device = any_present.device if any_present is not None else None
        for k in self.img_keys:
            if k in imgs:
                x = imgs[k]
                if x.dim()==3: x = x.unsqueeze(0)
                parts.append(self.encoders[k](x))
            else:
                if B is None:
                    continue
                # create zeros with encoder output dim (try to infer)
                enc = self.encoders[k]
                # attempt to get out-dim from proj or last layer; default to encoder_out_dim guess
                out_dim = None
                if hasattr(enc, 'proj') and isinstance(enc.proj, nn.Linear):
                    out_dim = enc.proj.out_features
                elif hasattr(enc, 'encoder') and hasattr(enc.encoder, 'fc'):
                    out_dim = enc.encoder.fc.out_features
                else:
                    out_dim = 512
                parts.append(torch.zeros(B, out_dim, device=device))
        if not parts: return None
        return torch.cat(parts, dim=-1)

    def make_repr(self, vec, imgs):
        if vec.ndim==1: vec=vec.unsqueeze(0)
        z = self.encode_imgs(imgs)
        if z is None:
            x = vec
        else:
            if vec.numel()==0:
                x = z
            else:
                x = torch.cat([vec, z], -1)
        return self.repr_mlp(x)

    def soft_update(self):
        for tp,p in zip(self.target_critic.parameters(), self.critic.parameters()):
            tp.data.copy_(tp.data*(1-self.tau) + p.data*self.tau)
        for tp,p in zip(self.target_value.parameters(), self.value.parameters()):
            tp.data.copy_(tp.data*(1-self.tau) + p.data*self.tau)

# ===================== Training =====================

def expectile_loss(diff,expectile):
    weight=torch.where(diff>0,expectile,(1-expectile))
    return (weight*diff.pow(2)).mean()

def eval_agent(agent,val_loader,device,img_keys,gamma):
    agent.eval(); total_loss=0; n=0
    with torch.no_grad():
        for batch in val_loader:
            batch = [
                        {
                            k: (
                                v.to(device)
                                if not isinstance(v, dict)
                                else {ik: iv.to(device) for ik, iv in v.items()}   # 去掉 unsqueeze(0)
                            )
                            for k, v in b.items()
                        }
                        for b in batch
                    ]
            vec=torch.stack([b['vec'] for b in batch]).to(device)
            imgs={k:torch.stack([b['imgs'][k] for b in batch]) for k in img_keys}
            act=torch.stack([b['action'] for b in batch]).to(device)
            rew=torch.stack([b['reward'] for b in batch]).to(device)
            nvec=torch.stack([b['next_vec'] for b in batch]).to(device)
            nimgs={k:torch.stack([b['next_imgs'][k] for b in batch]) for k in img_keys}
            done=torch.stack([b['done'] for b in batch]).to(device)

            z=agent.make_repr(vec,imgs); zn=agent.make_repr(nvec,nimgs)
            with torch.no_grad(): target_v=agent.value(zn)
            target_q=rew+gamma*((1-done)*target_v)
            q1,q2=agent.critic(z,act); loss_q=F.mse_loss(q1,target_q)+F.mse_loss(q2,target_q)
            total_loss+=loss_q.item(); n+=1
    return total_loss/max(1,n)

def train_iql(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    set_seed(args.seed)

    # =================== 数据集 ===================
    with h5py.File(args.data_dir, 'r') as f:
        demos = list(f['data_frames'].keys())
    random.shuffle(demos)
    n_val = max(1, int(len(demos) * args.val_split))
    val_demos, train_demos = demos[:n_val], demos[n_val:]

    use_mod = {k: True for k in
               ['agentview_rgb','hand_camera_rgb','ee_poses','hole_poses','peg_poses',
                'robot_joint_states','robot_measured_joint_forces','gripper_joint_states','peg_hole_forces']}

    train_ds = H5TrajectoryDataset(args.data_dir, train_demos, use_mod, args.image_size)
    val_ds = H5TrajectoryDataset(args.data_dir, val_demos, use_mod, args.image_size)

    # weighted sampler to handle sparse reward
    weights = []
    for i in range(len(train_ds)):
        r = float(train_ds[i]["reward"])
        weights.append(5.0 if r > 0.5 else 1.0)   # upweight success
    sampler = torch.utils.data.WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, collate_fn=lambda x:x)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=lambda x:x)

    # =================== Agent ===================
    sample = train_ds[0]
    obs_vec_dim = sample['vec'].numel()
    act_dim = sample['action'].numel() if 'action' in sample else 0
    img_keys = list(sample['imgs'].keys())
    agent = IQLAgent(
        obs_vec_dim=obs_vec_dim,
        act_dim=act_dim,
        img_keys=img_keys,
        r3m=args.r3m,
        encoder_out_dim=args.encoder_out_dim,
        repr_dim=args.repr_dim
    ).to(device)

    opt_actor = torch.optim.Adam(agent.actor.parameters(), lr=3e-4)
    opt_critic = torch.optim.Adam(agent.critic.parameters(), lr=3e-4)
    opt_value = torch.optim.Adam(agent.value.parameters(), lr=3e-4)

    # target value net for stability: deepcopy the online value net
    target_value = copy.deepcopy(agent.value).to(device)
    for p in target_value.parameters():
        p.requires_grad = False


    best_val = float("inf")
    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, "best_model.pth")

    train_logs, eval_logs = [], []

    # =================== 训练循环 ===================
    for epoch in range(1, args.epochs + 1):
        agent.train()
        total_q, total_v, total_pi = 0,0,0
        total_rewards = []

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
            # move batch to device
            for b in batch:
                for k,v in b.items():
                    if isinstance(v,dict):
                        for ik,iv in v.items(): b[k][ik]=iv.to(device)
                    else:
                        b[k]=v.to(device)

            vec=torch.stack([b['vec'] for b in batch])
            nvec=torch.stack([b['next_vec'] for b in batch])
            done=torch.stack([b['done'] for b in batch]).float()
            act=torch.stack([b['action'] for b in batch])
            rew=torch.stack([b['reward'] for b in batch]).float()
            imgs={k:torch.stack([b['imgs'][k] for b in batch]) for k in img_keys}
            nimgs={k:torch.stack([b['next_imgs'][k] for b in batch]) for k in img_keys}

            z=agent.make_repr(vec,imgs); zn=agent.make_repr(nvec,nimgs)

            # critic update
            with torch.no_grad():
                target_v = target_value(zn)
                target_q = rew + args.gamma * ((1-done) * target_v)
            q1,q2=agent.critic(z.detach(),act)
            loss_q = F.mse_loss(q1,target_q) + F.mse_loss(q2,target_q)
            opt_critic.zero_grad(); loss_q.backward(); opt_critic.step()

            # value update
            with torch.no_grad():
                q=torch.min(*agent.critic(z.detach(),act))
            v=agent.value(z.detach())
            loss_v=expectile_loss(q-v,agent.expectile)
            opt_value.zero_grad(); loss_v.backward(); opt_value.step()

            # actor update
            adv=(q-v).detach()
            adv=adv.clamp(-5,5)  # prevent exploding
            exp_a=torch.exp(agent.beta*adv).clamp(max=100)
            pred_act=agent.actor(z.detach())
            loss_pi=(exp_a*((pred_act-act)**2).sum(-1,keepdim=True)).mean()
            opt_actor.zero_grad(); loss_pi.backward(); opt_actor.step()

            # soft update
            agent.soft_update()
            for tp, p in zip(target_value.parameters(), agent.value.parameters()):
                tp.data.copy_(tp.data*(1 - agent.tau) + p.data * agent.tau)

            total_q+=loss_q.item(); total_v+=loss_v.item(); total_pi+=loss_pi.item()
            total_rewards+=rew.cpu().numpy().tolist()

        # epoch logging
        mean_r=np.mean(total_rewards)
        print(f"[Epoch {epoch}] q={total_q/len(train_loader):.3f} v={total_v/len(train_loader):.3f} pi={total_pi/len(train_loader):.3f} avg_rew={mean_r:.3f}")
        train_logs.append((total_q/len(train_loader), total_v/len(train_loader), total_pi/len(train_loader), mean_r))

        # eval
        val_loss = eval_agent(agent, val_loader, device, img_keys, args.gamma)
        success_rate = np.mean([b["reward"].item()>0.5 for batch in val_loader for b in batch])
        print(f"           val_loss={val_loss:.3f}, success_rate={success_rate*100:.1f}%")
        eval_logs.append((val_loss, success_rate))

        if val_loss < best_val:
            best_val=val_loss
            torch.save({"epoch":epoch,"agent":agent.state_dict()}, save_path)
            print(f"[INFO] saved best model @epoch {epoch}, val={val_loss:.3f}")

    # plot
    qs,vs,pis,rs=zip(*train_logs)
    vals,srates=zip(*eval_logs)
    plt.figure(figsize=(10,6))
    plt.subplot(2,1,1)
    plt.plot(qs,label="critic loss");plt.plot(vs,label="value loss");plt.plot(pis,label="actor loss")
    plt.legend();plt.grid();plt.title("Training losses")
    plt.subplot(2,1,2)
    plt.plot(vals,label="val loss");plt.plot(rs,label="avg train reward");plt.plot(srates,label="val success rate")
    plt.legend();plt.grid();plt.title("Validation metrics")
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir,"Train_Val_metrics.png"))

# ===================== CLI =====================

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--data_dir",type=str,required=True)
    p.add_argument("--r3m",action="store_true")
    p.add_argument('--gamma', type=float, default=0.99)
    p.add_argument("--image_size",type=int,default=128)
    p.add_argument("--encoder_out_dim",type=int,default=512)
    p.add_argument("--repr_dim",type=int,default=512)
    p.add_argument("--batch_size",type=int,default=32)
    p.add_argument("--epochs",type=int,default=60)
    p.add_argument("--val_split",type=float,default=0.3)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--cpu",action="store_true")
    p.add_argument("--save_dir",type=str,default="./checkpoints")
    return p.parse_args()

if __name__=="__main__":
    args=parse_args()
    train_iql(args)
