#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IQL Training on Robotic H5 Dataset (Single File, Multiple Demos)

This version is rewritten to exactly fit the user's H5 structure:

H5 Structure (per demo_N under "data_frames"):
  group: demo_0
    attr: demo_0.demo_name = "1754211626.9062428_episode_194.json"
    dataset: actions                     (T, 7)   float64
    dataset: episode_step                (T,)     int64
    dataset: gripper_joint_states        (T, 2)   float32
    dataset: hole_poses                  (T, 7)   float32
    dataset: is_success                  (T,)     bool
    group: observations
      dataset: agentview_rgb             (T, H, W, 3) uint8
      dataset: ee_poses                  (T, 7)       float64
      dataset: hand_camera_rgb           (T, H, W, 3) uint8
    dataset: peg_hole_forces             (T, 3)   float32
    dataset: peg_poses                   (T, 7)   float32
    dataset: rewards                     (T,)     float64
    dataset: robot_joint_states          (T, 14)  float32
    dataset: robot_measured_joint_forces (T, 54)  float32

Notes:
- Validation split is by demos (not files).
- Images are read from observations group; vector features from both observations and demo root.
- Adds hole_poses & peg_poses into proprioceptive vector.
"""

import os
import json
import random
import argparse
from glob import glob
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
from r3m import load_r3m


# ------------------------------------------------------------
# Utils
# ------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def exists(p: Optional[str]) -> bool:
    return p is not None and len(p) > 0 and os.path.exists(p)


# ------------------------------------------------------------
# Reward Hook (Optional)
# ------------------------------------------------------------

class RewardHook:
    def compute(self, demo: h5py.Group, t: int) -> float:
        """Override to define custom reward from h5 demo group."""
        return 0.0


# ------------------------------------------------------------
# Dataset
# ------------------------------------------------------------

class H5TrajectoryDataset(Dataset):
    """
    Dataset that reads transitions (s_t, a_t, r_t, s_{t+1}, done) from a single .h5 file
    containing multiple demos under group 'data_frames'. Adapted for the given structure.
    """
    def __init__(
        self,
        h5_path: str,
        demo_names: List[str],
        use_modalities: Dict[str, bool],
        image_size: int = 128,
        reward_hook: Optional[RewardHook] = None,
        dtype=torch.float32,
    ):
        super().__init__()
        self.h5_path = h5_path
        self.demo_names = demo_names
        self.use_modalities = use_modalities
        self.reward_hook = reward_hook
        self.dtype = dtype

        # Image transform: HWC uint8 -> CHW float [0,1], then resize
        self.img_tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((image_size, image_size), antialias=True),
        ])

        # Build index of (demo_name, t)
        self.index: List[Tuple[str, int]] = []
        self.traj_meta: List[Dict] = []

        with h5py.File(self.h5_path, 'r') as f:
            assert 'data_frames' in f, "Missing group 'data_frames' in H5."
            df_group = f['data_frames']
            for demo_name in demo_names:
                if demo_name not in df_group:
                    print(f"[WARN] Demo {demo_name} not found in {self.h5_path}")
                    continue
                demo = df_group[demo_name]
                assert 'actions' in demo, f"Missing 'actions' in {demo_name}"
                T = demo['actions'].shape[0]

                self.traj_meta.append({
                    'demo_name': demo_name,
                    'T': T,
                    'has_reward': 'rewards' in demo,
                    'has_success': 'is_success' in demo
                })

                # transitions from t in [0, T-2], making (s_t -> s_{t+1})
                for t in range(T - 1):
                    self.index.append((demo_name, t))

        assert len(self.index) > 0, "No transitions found. Check H5 structure and demo names."

    def __len__(self):
        return len(self.index)

    # ---- helpers ----

    def _get_obs(self, demo: h5py.Group, t: int) -> Dict[str, torch.Tensor]:
        """
        Assemble an observation dict for time index t by reading:
          - images from demo['observations']
          - vectors from both demo['observations'] and demo root
        """
        obs: Dict[str, torch.Tensor] = {}

        obs_group = demo['observations'] if 'observations' in demo else None

        # RGB images
        if obs_group is not None:
            if self.use_modalities.get('agentview_rgb', True) and 'agentview_rgb' in obs_group:
                img = obs_group['agentview_rgb'][t]  # (H, W, 3), uint8
                obs['agentview_rgb'] = self.img_tf(img)
            if self.use_modalities.get('hand_camera_rgb', True) and 'hand_camera_rgb' in obs_group:
                img = obs_group['hand_camera_rgb'][t]
                obs['hand_camera_rgb'] = self.img_tf(img)

        # From observations group (vector)
        if obs_group is not None and self.use_modalities.get('ee_poses', True) and 'ee_poses' in obs_group:
            obs['ee_poses'] = torch.from_numpy(np.array(obs_group['ee_poses'][t])).float()

        # From root of demo (vectors)
        root_vec_keys = [
            'hole_poses',
            'peg_poses',
            'robot_joint_states',
            'robot_measured_joint_forces',
            'gripper_joint_states',
            'peg_hole_forces',
        ]
        for k in root_vec_keys:
            if self.use_modalities.get(k, True) and k in demo:
                obs[k] = torch.from_numpy(np.array(demo[k][t])).float()

        return obs

    def _concat_obs(self, obs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Concatenate selected vector features into a single 1D tensor, and collect images.
        Vector key order is fixed for reproducibility.
        """
        vec_keys = [
            'ee_poses',
            'hole_poses',
            'peg_poses',
            'robot_joint_states',
            'robot_measured_joint_forces',
            'gripper_joint_states',
            'peg_hole_forces',
        ]
        vecs = []
        for k in vec_keys:
            if k in obs:
                v = obs[k]
                if v.ndim == 0:
                    v = v.unsqueeze(0)
                vecs.append(v)
        vec = torch.cat(vecs, dim=-1) if len(vecs) > 0 else torch.zeros(0, dtype=self.dtype)
        imgs = {k: obs[k] for k in ['agentview_rgb', 'hand_camera_rgb'] if k in obs}
        return vec, imgs

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        demo_name, t = self.index[idx]
        with h5py.File(self.h5_path, 'r') as f:
            demo = f['data_frames'][demo_name]

            # current and next observation
            obs_t = self._get_obs(demo, t)
            obs_tp1 = self._get_obs(demo, t + 1)
            vec_t, imgs_t = self._concat_obs(obs_t)
            vec_tp1, imgs_tp1 = self._concat_obs(obs_tp1)

            # action at t
            action = torch.from_numpy(np.array(demo['actions'][t])).float()

            # reward
            if 'rewards' in demo:
                r_t = float(np.array(demo['rewards'][t]))
            elif self.reward_hook is not None:
                r_t = float(self.reward_hook.compute(demo, t))
            else:
                r_t = 0.0
            reward = torch.tensor([r_t], dtype=self.dtype)

            # done if next step is terminal (i.e., t+1 == T-1)
            T = demo['actions'].shape[0]
            done = torch.tensor([1.0 if t + 1 == T - 1 else 0.0], dtype=self.dtype)

            # optional success label
            success = None
            if 'is_success' in demo:
                # bool -> float
                success = torch.tensor([float(np.array(demo['is_success'][t]))], dtype=self.dtype)

        sample: Dict[str, torch.Tensor] = {
            'vec': vec_t,
            'imgs': imgs_t,
            'action': action,
            'reward': reward,
            'next_vec': vec_tp1,
            'next_imgs': imgs_tp1,
            'done': done,
            'traj_idx': torch.tensor([hash(demo_name) % (10**7)], dtype=torch.long),
        }
        if success is not None:
            sample['success'] = success
        return sample


# ------------------------------------------------------------
# Models
# ------------------------------------------------------------

class SimpleCNN(nn.Module):
    def __init__(self, in_ch=3, out_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 7, stride=2, padding=3), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 5, stride=2, padding=2), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.proj = nn.Linear(256, out_dim)

    def forward(self, x):
        h = self.net(x)
        h = torch.flatten(h, 1)
        return self.proj(h)


class VisualEncoder(nn.Module):
    """
    Minimal visual encoder; if r3m_path is provided and loadable as a nn.Module or state_dict,
    we try to load it; otherwise fallback to SimpleCNN.
    """
    def __init__(self, r3m: bool, out_dim: int = 512, freeze: bool = True):
        super().__init__()
        self.use_r3m = r3m
        if self.use_r3m:
            try:
                ckpt = load_r3m("resnet50")
                ckpt.eval()
                print(f"[INFO] Loaded R3M")
                if isinstance(ckpt, nn.Module):
                    self.encoder = ckpt
                else:
                    self.encoder = SimpleCNN(3, out_dim)
                    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
                        self.encoder.load_state_dict(ckpt['state_dict'], strict=False)
                if not isinstance(self.encoder, nn.Module):
                    self.encoder = SimpleCNN(3, out_dim)
            except Exception as e:
                print(f"[WARN] Failed to load R3M, fallback to SimpleCNN. Error: {e}")
                self.encoder = SimpleCNN(3, out_dim)
        else:
            self.encoder = SimpleCNN(3, out_dim)

        if freeze and self.use_r3m:
            for p in self.encoder.parameters():
                p.requires_grad = False
        self.out_dim = out_dim

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        return self.encoder(img)


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=(512, 512), act=nn.ReLU):
        super().__init__()
        layers: List[nn.Module] = []
        last = in_dim
        for h in hidden:
            layers += [nn.Linear(last, h), act()]
            last = h
        layers += [nn.Linear(last, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Actor(nn.Module):
    def __init__(self, in_dim, act_dim, hidden=(512, 512)):
        super().__init__()
        self.mu = MLP(in_dim, act_dim, hidden)

    def forward(self, x):
        return self.mu(x)


class Critic(nn.Module):
    def __init__(self, in_dim, act_dim, hidden=(512, 512)):
        super().__init__()
        self.q1 = MLP(in_dim + act_dim, 1, hidden)
        self.q2 = MLP(in_dim + act_dim, 1, hidden)

    def forward(self, x, a):
        xu = torch.cat([x, a], dim=-1)
        return self.q1(xu), self.q2(xu)


class ValueNet(nn.Module):
    def __init__(self, in_dim, hidden=(512, 512)):
        super().__init__()
        self.v = MLP(in_dim, 1, hidden)

    def forward(self, x):
        return self.v(x)


# ------------------------------------------------------------
# IQL Agent
# ------------------------------------------------------------

class IQLAgent(nn.Module):
    def __init__(
        self, obs_vec_dim: int, act_dim: int,
        r3m: bool, img_keys: List[str],
        encoder_out_dim: int = 512, enc_freeze: bool = True,
        repr_mlp_dim: int = 512,
        actor_hidden=(512, 512), critic_hidden=(512, 512), value_hidden=(512, 512),
        beta: float = 3.0, expectile: float = 0.7, discount: float = 0.99,
        tau: float = 0.005, awr_clip: float = 100.0
    ):
        super().__init__()
        self.img_keys = img_keys
        self.discount = discount
        self.tau = tau
        self.beta = beta
        self.expectile = expectile
        self.awr_clip = awr_clip

        # Encoders per image key
        self.encoders = nn.ModuleDict({k: VisualEncoder(r3m, out_dim=encoder_out_dim, freeze=enc_freeze)
                                       for k in img_keys})
        total_img_dim = encoder_out_dim * len(img_keys)

        # Representation (concat(vec, img_feats) -> repr)
        in_repr = obs_vec_dim + total_img_dim
        self.repr_mlp = MLP(in_repr, repr_mlp_dim, hidden=(512, 512))

        # Heads
        self.actor = Actor(repr_mlp_dim, act_dim, actor_hidden)
        self.critic = Critic(repr_mlp_dim, act_dim, critic_hidden)
        self.value = ValueNet(repr_mlp_dim, value_hidden)

        # Target critic
        self.target_critic = Critic(repr_mlp_dim, act_dim, critic_hidden)
        self.target_critic.load_state_dict(self.critic.state_dict())
        for p in self.target_critic.parameters():
            p.requires_grad = False

    def encode_imgs(self, imgs: Dict[str, torch.Tensor]) -> torch.Tensor:
        feats = []
        for k in self.img_keys:
            if k in imgs:
                feats.append(self.encoders[k](imgs[k]))
        if len(feats) == 0:
            # batch size fallback: 0-dim feats; infer batch from any existing tensor if possible
            raise RuntimeError("No image keys found in batch. Check use_modalities and dataset.")
        return torch.cat(feats, dim=-1)

    def make_repr(self, vec: torch.Tensor, imgs: Dict[str, torch.Tensor]) -> torch.Tensor:

        if vec.ndim == 1:
            vec = vec.unsqueeze(0)
        z_img = self.encode_imgs(imgs) if len(self.img_keys) > 0 else torch.zeros(vec.shape[0], 0, device=vec.device)
        if vec.numel() > 0:
            x = torch.cat([vec, z_img], dim=-1)
        else:
            x = z_img
        print("vec.shape:", vec.shape)
        print("z_img.shape:", z_img.shape)
        print("x.shape:", x.shape)
        return self.repr_mlp(x)

    @torch.no_grad()
    def act(self, vec: torch.Tensor, imgs: Dict[str, torch.Tensor]) -> torch.Tensor:
        x = self.make_repr(vec, imgs)
        return self.actor(x)

    def soft_update_targets(self):
        with torch.no_grad():
            for tp, p in zip(self.target_critic.parameters(), self.critic.parameters()):
                tp.data.copy_(tp.data * (1.0 - self.tau) + p.data * self.tau)


# ------------------------------------------------------------
# Collate & Device
# ------------------------------------------------------------

def collate_fn(samples: List[Dict]):
    # image keys union
    keys_img = set()
    for s in samples:
        keys_img.update(s['imgs'].keys())
    batch: Dict = {
        'vec': torch.stack([s['vec'] for s in samples]),
        'action': torch.stack([s['action'] for s in samples]),
        'reward': torch.stack([s['reward'] for s in samples]),
        'next_vec': torch.stack([s['next_vec'] for s in samples]),
        'done': torch.stack([s['done'] for s in samples]),
        'imgs': {},
        'next_imgs': {},
        'traj_idx': torch.stack([s['traj_idx'] for s in samples]).squeeze(-1),
    }
    # image tensors: if a sample is missing a key, create zeros of correct shape
    def zeros_like_img(img: torch.Tensor):
        return torch.zeros_like(img)

    for k in keys_img:
        imgs_t = []
        imgs_tp1 = []
        for s in samples:
            if k in s['imgs']:
                imgs_t.append(s['imgs'][k])
            else:
                # fallback to any existing image to get shape
                ref = next(iter(s['imgs'].values())) if len(s['imgs']) > 0 else None
                if ref is None:
                    raise RuntimeError("Cannot infer image shape for missing key.")
                imgs_t.append(zeros_like_img(ref))

            if k in s['next_imgs']:
                imgs_tp1.append(s['next_imgs'][k])
            else:
                refn = next(iter(s['next_imgs'].values())) if len(s['next_imgs']) > 0 else None
                if refn is None:
                    raise RuntimeError("Cannot infer next image shape for missing key.")
                imgs_tp1.append(zeros_like_img(refn))
        batch['imgs'][k] = torch.stack(imgs_t)
        batch['next_imgs'][k] = torch.stack(imgs_tp1)

    if 'success' in samples[0]:
        batch['success'] = torch.stack([s['success'] for s in samples])

    return batch


def to_device_batch(batch: Dict, device: torch.device, img_keys: List[str]) -> Dict:
    out = {k: batch[k].to(device) for k in ['vec', 'action', 'reward', 'next_vec', 'done', 'traj_idx']}
    out['imgs'] = {k: batch['imgs'][k].to(device) for k in img_keys if k in batch['imgs']}
    out['next_imgs'] = {k: batch['next_imgs'][k].to(device) for k in img_keys if k in batch['next_imgs']}
    if 'success' in batch:
        out['success'] = batch['success'].to(device)
    return out


# ------------------------------------------------------------
# Losses (IQL)
# ------------------------------------------------------------

def expectile_loss(diff: torch.Tensor, expectile: float):
    """
    Expectile regression loss: E_w[(diff)^2] with asymmetric weights by sign(diff).
    diff = q - v
    """
    w = torch.where(diff > 0, expectile, 1 - expectile)
    return (w * diff.pow(2)).mean()


def awr_weights(q: torch.Tensor, v: torch.Tensor, beta: float, awr_clip: float):
    """
    Advantage-weighted regression weights: exp((q - v)/beta), clipped for stability.
    q, v have shape [B, 1].
    """
    adv = (q - v).detach()
    return torch.exp(adv / beta).clamp(max=awr_clip)


# ------------------------------------------------------------
# Evaluation (offline stats proxy)
# ------------------------------------------------------------

@torch.no_grad()
def evaluate_offline(agent: IQLAgent, loader: DataLoader, device: torch.device, img_keys: List[str]) -> Dict[str, float]:
    returns = []
    success = []
    qs = []

    cur_traj_id = None
    cum_ret = 0.0

    for batch in loader:
        batch = to_device_batch(batch, device, img_keys)

        # representation
        x = agent.make_repr(batch['vec'], batch['imgs'])

        # actor produces actions; critic evaluates them (not necessary for IQL eval, but gives a proxy)
        a_pred = agent.actor(x)
        q1, q2 = agent.critic(x, a_pred)
        q_min = torch.min(q1, q2)
        qs.extend(q_min.squeeze(-1).detach().cpu().numpy())

        # accumulate returns per trajectory id
        rewards = batch['reward'].detach().cpu().numpy().flatten()
        traj_ids = batch['traj_idx'].detach().cpu().numpy().flatten()
        for r, tid in zip(rewards, traj_ids):
            if cur_traj_id is None:
                cur_traj_id = tid
            if tid != cur_traj_id:
                returns.append(cum_ret)
                cum_ret = 0.0
                cur_traj_id = tid
            cum_ret += r

        if 'success' in batch:
            success.extend(batch['success'].detach().cpu().numpy().flatten())

    if cur_traj_id is not None:
        returns.append(cum_ret)

    stats = {
        'mean_return': float(np.mean(returns)) if len(returns) > 0 else 0.0,
        'std_return': float(np.std(returns)) if len(returns) > 0 else 0.0,
        'mean_q': float(np.mean(qs)) if len(qs) > 0 else 0.0,
    }
    if len(success) > 0:
        stats['success_rate_proxy'] = float(np.mean(success))
    return stats


# ------------------------------------------------------------
# Training Loop (IQL)
# ------------------------------------------------------------

def train_iql(args):
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    set_seed(args.seed)

    # Resolve H5 path
    if os.path.isfile(args.data_dir) and args.data_dir.endswith('.h5'):
        h5_path = args.data_dir
    elif os.path.isdir(args.data_dir):
        candidates = sorted(glob(os.path.join(args.data_dir, "*.h5")))
        assert len(candidates) > 0, f"No .h5 file found in directory: {args.data_dir}"
        h5_path = candidates[0]
        print(f"[INFO] Using .h5 file: {h5_path}")
    else:
        raise ValueError("--data_dir must be a .h5 file or a directory containing one")

    # List all demo names
    with h5py.File(h5_path, 'r') as f:
        assert 'data_frames' in f, "Missing group 'data_frames' in H5."
        all_demo_names = list(f['data_frames'].keys())
    assert len(all_demo_names) > 0, "No demos found under 'data_frames'."

    # Split demos into train/val
    random.shuffle(all_demo_names)
    n_val = max(1, int(len(all_demo_names) * args.val_split))
    val_demo_names = all_demo_names[:n_val]
    train_demo_names = all_demo_names[n_val:]

    print(f"[INFO] Total demos: {len(all_demo_names)} | Train: {len(train_demo_names)} | Val: {len(val_demo_names)}")

    # Save split
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, 'dataset_split.json'), 'w') as f:
        json.dump({'train': train_demo_names, 'val': val_demo_names}, f, indent=2)

    # Modalities (use all by default)
    use_mod = {
        'agentview_rgb': True,
        'hand_camera_rgb': True,
        'ee_poses': True,                  # observations
        'hole_poses': True,                # root
        'peg_poses': True,                 # root
        'robot_joint_states': True,        # root
        'robot_measured_joint_forces': True,
        'gripper_joint_states': True,
        'peg_hole_forces': True,
    }

    # Datasets/Dataloaders
    train_ds = H5TrajectoryDataset(h5_path, train_demo_names, use_mod, image_size=args.image_size)
    val_ds   = H5TrajectoryDataset(h5_path, val_demo_names,   use_mod, image_size=args.image_size)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn
    )

    # Infer dims
    sample = train_ds[0]
    obs_vec_dim = sample['vec'].numel()
    act_dim = sample['action'].numel()
    img_keys = [k for k in ['agentview_rgb', 'hand_camera_rgb'] if use_mod.get(k, False)]
    print("obs_vec_dim:", obs_vec_dim)
    print("act_dim:", act_dim)
    print("img_keys:", img_keys)
    print("sample['vec'].shape:", sample['vec'].shape)
    print("sample['imgs'].keys():", sample['imgs'].keys())
    for k in img_keys:
        if k in sample['imgs']:
            print(f"{k} shape:", sample['imgs'][k].shape)
            
    # Agent
    agent = IQLAgent(
        obs_vec_dim=obs_vec_dim,
        act_dim=act_dim,
        r3m=args.r3m,
        img_keys=img_keys,
        encoder_out_dim=args.encoder_out_dim,
        enc_freeze=not args.finetune_encoder,
        repr_mlp_dim=args.repr_dim,
        actor_hidden=(args.hidden, args.hidden),
        critic_hidden=(args.hidden, args.hidden),
        value_hidden=(args.hidden, args.hidden),
        beta=args.beta,
        expectile=args.expectile,
        discount=args.gamma,
        tau=args.tau,
        awr_clip=args.awr_clip,
    ).to(device)

    # Optimizers
    opt_actor  = torch.optim.AdamW(agent.actor.parameters(), lr=args.lr)
    opt_critic = torch.optim.AdamW(agent.critic.parameters(), lr=args.lr)
    opt_value  = torch.optim.AdamW(agent.value.parameters(), lr=args.lr)

    enc_params = list(agent.repr_mlp.parameters())
    if args.finetune_encoder:
        enc_params += list(agent.encoders.parameters())
    opt_repr = torch.optim.AdamW(enc_params, lr=args.lr) if len(enc_params) > 0 else None

    # ---- Training ----
    global_step = 0
    best_val = -1e9

    for epoch in range(1, args.epochs + 1):
        agent.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for batch in pbar:
            global_step += 1
            batch = to_device_batch(batch, device, img_keys)

            # Zero all grads ONCE
            opt_actor.zero_grad(set_to_none=True)
            opt_critic.zero_grad(set_to_none=True)
            opt_value.zero_grad(set_to_none=True)
            if opt_repr is not None:
                opt_repr.zero_grad(set_to_none=True)

            # Forward (s, a, r, s')
            x = agent.make_repr(batch['vec'], batch['imgs'])
            with torch.no_grad():
                x_next = agent.make_repr(batch['next_vec'], batch['next_imgs'])

            # ------- Targets -------
            # Stop gradient through target value as in IQL
            with torch.no_grad():
                v_next = agent.value(x_next).detach()  # [B,1]
                target = batch['reward'] + (1.0 - batch['done']) * args.gamma * v_next  # [B,1]

            # ------- Critic loss -------
            q1, q2 = agent.critic(x, batch['action'])                 # grads flow
            q_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)  # target no grad

            # ------- Value loss (expectile) -------
            # q should NOT backprop into critic here
            q1_sg, q2_sg = q1.detach(), q2.detach()
            q_min_sg = torch.min(q1_sg, q2_sg)          # stop-grad
            v = agent.value(x)                           # grads flow to value (& repr/enc if enabled)
            v_loss = expectile_loss(q_min_sg - v, agent.expectile)

            # ------- Actor loss (AWR regression) -------
            # weights computed from stop-grad advantage
            w = awr_weights(q_min_sg, v.detach(), agent.beta, agent.awr_clip)  # [B,1]
            pred_a = agent.actor(x)  # grads flow to actor (& repr/enc if enabled)
            a_loss = (w * (pred_a - batch['action']).pow(2).sum(dim=-1, keepdim=True)).mean()

            # ------- Single backward pass -------
            total_loss = q_loss + v_loss + a_loss
            total_loss.backward()

            # Clip and step each optimizer
            torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), args.grad_clip)
            torch.nn.utils.clip_grad_norm_(agent.value.parameters(),  args.grad_clip)
            torch.nn.utils.clip_grad_norm_(agent.actor.parameters(),  args.grad_clip)
            if opt_repr is not None:
                torch.nn.utils.clip_grad_norm_(enc_params, args.grad_clip)

            opt_critic.step()
            opt_value.step()
            opt_actor.step()
            if opt_repr is not None:
                opt_repr.step()

            agent.soft_update_targets()

            if global_step % args.log_interval == 0:
                pbar.set_postfix(
                    v_loss=f"{v_loss.item():.4f}",
                    q_loss=f"{q_loss.item():.4f}",
                    a_loss=f"{a_loss.item():.4f}"
                )

        # ---- Validation ----
        agent.eval()
        val_stats = evaluate_offline(agent, val_loader, device, img_keys)
        # ---- Use mean_q as evaluation metric ----
        val_score = val_stats['mean_q']  # 改用 mean_q 作为 best.pt 判定依据

        # ---- Save ----
        ckpt = {
            'agent': agent.state_dict(),
            'args': vars(args),
            'epoch': epoch,
            'val_stats': val_stats,
        }
        torch.save(ckpt, os.path.join(args.out_dir, f'checkpoint_epoch{epoch}.pt'))
        if val_score > best_val:
            best_val = val_score
            torch.save(ckpt, os.path.join(args.out_dir, 'best.pt'))

        print(f"[INFO] Epoch {epoch} | Val stats: {json.dumps(val_stats, indent=2)}")

    print(f"[INFO] Training complete. Best val return: {best_val:.3f}")



# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="IQL training on robotic H5 dataset (adapted structure).")
    p.add_argument('--data_dir', type=str, required=True, help='Path to .h5 file or a directory containing it.')
    p.add_argument('--out_dir', type=str, default='./runs_iql_h5', help='Output directory.')
    p.add_argument('--r3m', type=str, default=True, help='whether to use R3M or not')
    p.add_argument('--finetune_encoder', action='store_true', help='Finetune visual encoders.')
    p.add_argument('--image_size', type=int, default=128)
    p.add_argument('--encoder_out_dim', type=int, default=512)
    p.add_argument('--repr_dim', type=int, default=512)
    p.add_argument('--hidden', type=int, default=512)
    p.add_argument('--gamma', type=float, default=0.99)
    p.add_argument('--beta', type=float, default=3.0)
    p.add_argument('--expectile', type=float, default=0.7)
    p.add_argument('--tau', type=float, default=0.005)
    p.add_argument('--awr_clip', type=float, default=100.0)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--grad_clip', type=float, default=1.0)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--val_split', type=float, default=0.1)
    p.add_argument('--log_interval', type=int, default=50)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--cpu', action='store_true')
    return p.parse_args()


# ------------------------------------------------------------
# Entry
# ------------------------------------------------------------

if __name__ == '__main__':
    args = parse_args()
    """
    Example usage:
      python train_iql_h5.py --data_dir /path/to/dataset.h5 --out_dir ./runs_iql_h5 --epochs 50 --batch_size 128
      # (Optional) finetune visual encoders:
      python train_iql_h5.py --data_dir /path/to/dataset.h5 --finetune_encoder
    """
    train_iql(args)
