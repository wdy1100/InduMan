#!/usr/bin/env python3
"""
Behavior Cloning (BC) training & evaluation pipeline
===================================================

Implements a multimodal BC policy matching the experiment description:
- Inputs: two 128x128 RGB images (front & wrist), end-effector pose (6D),
         joint forces, contact forces.
- Image encoder: ResNet‑50 (ImageNet init), shared weights for both cameras.
- Non-visual encoder: MLP for proprioceptive & force signals.
- Fusion: concat(vision, proprio) → MLP → 7D action (3D pos, 3D ori, 1D gripper).
- Loss: MSE(action_pred, action_expert).
- Optimizer: Adam(lr=3e-4), batch=64, epochs=70. No DA / regularization by default.

This file is self-contained and provides:
- Dataset & transforms, including standardization for non-visual signals.
- Train/val split for your *own* dataset; separate loader for an *external* test dataset.
- Offline evaluation (MSE on val/test) and checkpointing of best model.
- Optional AMP for speed.

Expected data layout (flexible, see `BehaviorCloningDataset` for details):
Your training dataset directory should contain per-sample `.npz` files with keys:  
  - front_rgb:      (H, W, 3) uint8 or float32 in [0,255] or [0,1]
  - wrist_rgb:      (H, W, 3)
  - ee_pose:        (6,)  → [x, y, z, roll, pitch, yaw] or similar
  - joint_forces:   (J,)  → any size J >= 1
  - contact_forces: (C,)  → any size C >= 1
  - action:         (7,)  → [dx, dy, dz, d_roll, d_pitch, d_yaw, gripper]
Optionally, you may store trajectories and pre-slice into per-timestep samples.

For a second dataset (cross-dataset evaluation), keep the same `.npz` schema.

Usage examples
--------------
1) Train on your dataset and evaluate on held-out split:
python ./wdy_file/useful_scripts/bc_pipeline.py 
--train_dir=/home/wdy02/wdy_program/simulation_plus/IsaacLab/wdy_data/bc_data/npz/ONE_PEG_IN_HOLE 
--out_dir=/home/wdy02/wdy_program/simulation_plus/IsaacLab/wdy_data/bc_data/bc_ckpts/ONE_PEG_IN_HOLE 
--epochs=1 
--batch_size=1 
--expand_trajectories

2) Evaluate on *another* dataset using the best checkpoint:
python wdy_file/useful_scripts/bc_pipeline.py --eval_only 
--test_dir=/home/wdy02/software/isaacsim/wdy_data/bc_data/npz/ONE_PEG_IN_HOLE_60 
--ckpt=/home/wdy02/wdy_program/simulation_plus/IsaacLab/wdy_data/bc_data/bc_ckpts/ONE_PEG_IN_HOLE_60/best.pt 
--scaler=/home/wdy02/wdy_program/simulation_plus/IsaacLab/wdy_data/bc_data/bc_ckpts/ONE_PEG_IN_HOLE_60/scaler.pkl 
--expand_trajectories
Notes
-----
- Images are resized to 128x128 and normalized by ImageNet stats.
- Non-visual features are Z-scored via `StandardScaler` fitted on training set.
  The scaler is saved & reused for validation/test.
- No data augmentation or explicit regularization is applied by default to
  mirror the provided experiment; toggles exist but default to off.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split

# torchvision is used for ResNet-50 and standard transforms
import torchvision
from torchvision import transforms


# ------------------------------
# Utilities
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def plot_losses(train_losses: List[float], val_losses: List[float], action_dim: int, out_dir:str):
    """
    绘制训练和验证损失曲线，支持整体 MSE 和每个动作维度的 MSE。
    
    train_losses / val_losses: List[Dict]，每个 Dict 包含：
        - 'mse': 整体 MSE
        - 'mse_dim_0', 'mse_dim_1', ..., 'mse_dim_{action_dim-1}': per-dim MSE
    action_dim: 动作维度
    out_dir: 保存路径
    """
    os.makedirs(out_dir, exist_ok=True)
    epochs = len(train_losses)
    epoch_range = range(1, epochs + 1)

    # 绘制整体 MSE
    plt.figure(figsize=(7, 4))
    train_mse_all = [x['mse'] for x in train_losses]
    val_mse_all = [x['mse'] for x in val_losses]
    plt.plot(epoch_range, train_mse_all, label='Train MSE', marker='o')
    plt.plot(epoch_range, val_mse_all, label='Validation MSE', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.title('Overall Train and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'Train_Val_MSE_overall.png'))
    # plt.show()

    # 绘制每个动作维度的 MSE
    plt.figure(figsize=(10, 6))
    for i in range(action_dim):
        train_dim = [x[f'mse_dim_{i}'] for x in train_losses]
        val_dim = [x[f'mse_dim_{i}'] for x in val_losses]
        plt.plot(epoch_range, train_dim, label=f'Train dim {i}')
        plt.plot(epoch_range, val_dim, '--', label=f'Val dim {i}')
    plt.xlabel('Epoch')
    plt.ylabel('MSE per dimension')
    plt.title('Train and Validation MSE per Action Dimension')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'Train_Val_MSE_per_dim.png'))
    # plt.show()

# ------------------------------
# Dataset
class BehaviorCloningDataset(Dataset):
    """Loads per-sample .npz files with the expected keys.

    Each `.npz` may optionally contain arrays of shape (T, ...) for trajectories.
    If so, set `expand_trajectories=True` to expand to per-timestep samples.
    """

    def __init__(
        self,
        root: str | Path | List[str] | List[Path],
        resize_hw: Tuple[int, int] = (128, 128),
        expand_trajectories: bool = False,
        transform: Optional[Callable] = None,
    ) -> None:
        # 允许 root 是目录 或 文件列表
        if isinstance(root, (str, Path)):
            root = Path(root)
            assert root.is_dir(), f"Directory not found: {root}"
            self.files = sorted([p for p in root.glob("*.npz")])
        else:
            # 文件列表
            self.files = [Path(p) for p in root]
            assert all(f.suffix == ".npz" for f in self.files), "All files must be .npz"

        assert len(self.files) > 0, f"No .npz files found in {root}"

        self.expand_trajectories = expand_trajectories
        self.resize_hw = resize_hw

        # Image transforms (ImageNet normalization)
        if transform is not None:
            self.tf = transform
        else:
            self.tf = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(self.resize_hw, antialias=True),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),  # 光照/颜色变
                transforms.RandomHorizontalFlip(p=0.5),  # 水平翻转（如果任务对称）
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

        # Build index mapping if expanding trajectories
        self.index: List[Tuple[int, Optional[int]]] = []  # (file_idx, t)
        if self.expand_trajectories:
            for i, p in enumerate(self.files):
                with np.load(p) as data:
                    if data['front_rgb'].ndim == 4:
                        T = data['front_rgb'].shape[0]
                    elif data['front_rgb'].ndim == 3:
                        T = 1
                    else:
                        raise ValueError("front_rgb must be 3D or 4D")
                for t in range(T):
                    self.index.append((i, t if T > 1 else None))
        else:
            self.index = [(i, None) for i in range(len(self.files))]

    def __len__(self) -> int:
        return len(self.index)

    def _get_from_npz(self, p: Path, t: Optional[int]) -> Dict[str, np.ndarray]:
        with np.load(p) as data:
            def pick(key: str) -> np.ndarray:
                arr = data[key]
                if t is not None and arr.ndim >= 1 and arr.shape[0] > 1:
                    arr = arr[t]
                return arr

            sample = {
                'front_rgb': pick('front_rgb'),
                'wrist_rgb': pick('wrist_rgb'),
                'ee_pose': pick('ee_pose'),
                'joint_forces': pick('joint_forces'),
                'contact_forces': pick('contact_forces'),
                'joint_states': pick('joint_states'),
                'action': pick('action'),
            }
        return sample

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        fidx, t = self.index[idx]
        p = self.files[fidx]
        s = self._get_from_npz(p, t)

        # Images
        front = self.tf(s['front_rgb'])
        wrist = self.tf(s['wrist_rgb'])

        # Non-visual features
        ee = torch.from_numpy(s['ee_pose']).float().view(-1)
        jf = torch.from_numpy(s['joint_forces']).float().view(-1)
        cf = torch.from_numpy(s['contact_forces']).float().view(-1)
        joint_states = torch.from_numpy(s['joint_states']).float().view(-1)
        proprio = torch.cat([ee, joint_states, jf, cf], dim=0)

        action = torch.from_numpy(s['action']).float().view(-1)

        return {
            'front_rgb': front,
            'wrist_rgb': wrist,
            'proprio': proprio,
            'action': action,
        }

# ------------------------------
# Non-visual standardization

class Standardizer:

    """Z-score standardizer for 1D feature vectors stored inside samples.
    Keeps mean/std and applies to tensors on CPU/GPU.
    """

    def __init__(self):
        self.mean: Optional[torch.Tensor] = None
        self.std: Optional[torch.Tensor] = None

    def fit(self, loader: DataLoader, key: str = 'proprio', max_batches: Optional[int] = None):
        sums = None
        sums2 = None
        n = 0
        for i, batch in enumerate(loader):
            x = batch[key].float()
            if sums is None:
                d = x.shape[1]
                sums = torch.zeros(d)
                sums2 = torch.zeros(d)
            sums += x.sum(dim=0)
            sums2 += (x ** 2).sum(dim=0)
            n += x.shape[0]
            if max_batches is not None and (i + 1) >= max_batches:
                break
        assert n > 0
        mean = sums / n
        var = sums2 / n - mean ** 2
        std = torch.sqrt(torch.clamp(var, min=1e-8))
        self.mean, self.std = mean, std

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        assert self.mean is not None and self.std is not None, "Standardizer not fitted"
        return (x - self.mean.to(x.device)) / (self.std.to(x.device) + 1e-8)

    def save(self, path: str | Path):
        with open(path, 'wb') as f:
            pickle.dump({'mean': self.mean, 'std': self.std}, f)

    @staticmethod
    def load(path: str | Path) -> 'Standardizer':
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        st = Standardizer()
        st.mean = obj['mean']
        st.std = obj['std']
        return st

# action standardizer
class ActionStandardizer(Standardizer):
    """Z-score standardizer specifically for actions"""
    def fit(self, loader: DataLoader, key: str = 'action', max_batches: Optional[int] = None):
        super().fit(loader, key=key, max_batches=max_batches)

    def inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        assert self.mean is not None and self.std is not None, "ActionStandardizer not fitted"
        return x * (self.std.to(x.device) + 1e-8) + self.mean.to(x.device)
    
    @staticmethod
    def load(path: str | Path) -> 'ActionStandardizer':
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        st = ActionStandardizer()
        st.mean = obj['mean']
        st.std = obj['std']
        return st

# ------------------------------
# Model
class ImageEncoder(nn.Module):
    def __init__(self, out_dim: int = 2048, pretrained: bool = True, train_bn: bool = True):
        super().__init__()
        base = torchvision.models.resnet50(weights=(
            torchvision.models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        ))
        # Remove fc, keep global pooling output (2048)
        self.backbone = nn.Sequential(
            base.conv1, base.bn1, base.relu, base.maxpool,
            base.layer1, base.layer2, base.layer3, base.layer4,
            base.avgpool,  # → [B, 2048, 1, 1]
        )
        self.out_dim = out_dim
        self.train_bn = train_bn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.backbone(x)
        y = torch.flatten(y, 1)  # [B, 2048]
        return y

    def train(self, mode: bool = True):
        super().train(mode)
        if not self.train_bn:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
        return self

class ProprioEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden: Sequence[int] = (256, 256)):
        super().__init__()
        layers: List[nn.Module] = []
        last = in_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.ReLU(inplace=True)]
            last = h
        self.net = nn.Sequential(*layers)
        self.out_dim = last

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class BCPolicy(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_fusion: Sequence[int] = (512, 512, 256),
        output_dim: int = 7,
        pretrained_backbone: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        # Shared image encoder
        self.img_enc = ImageEncoder(pretrained=pretrained_backbone)
        vis_dim = self.img_enc.out_dim
        self.pro_enc = ProprioEncoder(input_dim)

        fusion_in = vis_dim * 2 + self.pro_enc.out_dim
        layers: List[nn.Module] = []
        last = fusion_in
        for h in hidden_fusion:
            layers += [nn.Linear(last, h), nn.ReLU(inplace=True)]
            if dropout > 0:
                layers += [nn.Dropout(p=dropout)]
            last = h
        layers += [nn.Linear(last, output_dim)]
        self.fusion = nn.Sequential(*layers)

    def forward(self, front_rgb: torch.Tensor, wrist_rgb: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        f1 = self.img_enc(front_rgb)
        f2 = self.img_enc(wrist_rgb)
        p = self.pro_enc(proprio)
        x = torch.cat([f1, f2, p], dim=1)
        act = self.fusion(x)
        return act

# ------------------------------
# Training & Evaluation

def build_loaders(cfg) -> Tuple[Optional[DataLoader], Optional[DataLoader], Optional[DataLoader], int, Standardizer, Standardizer]:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    scaler = Standardizer()
    action_scaler = ActionStandardizer()
    proprio_dim = None

    train_loader = val_loader = test_loader = None

    # ----------- 修改这里：基于文件级别划分 -----------
    if cfg.train_dir and not cfg.eval_only:
        all_files = sorted(list(Path(cfg.train_dir).glob("*.npz")))
        assert len(all_files) > 0, f"No .npz files found in {cfg.train_dir}"

        random.seed(cfg.seed)
        random.shuffle(all_files)

        n_total = len(all_files)
        n_val = int(n_total * cfg.val_split)
        val_files = all_files[:n_val]
        train_files = all_files[n_val:]

        ds_train = BehaviorCloningDataset(root=train_files, expand_trajectories=cfg.expand_trajectories)
        ds_val   = BehaviorCloningDataset(root=val_files,   expand_trajectories=cfg.expand_trajectories)

        # 推断 proprio 特征维度
        sample = ds_train[0]
        proprio_dim = sample['proprio'].numel()

        train_loader = DataLoader(ds_train, batch_size=cfg.batch_size, shuffle=True,
                                  num_workers=cfg.num_workers, pin_memory=(device == 'cuda'))
        val_loader = DataLoader(ds_val, batch_size=cfg.batch_size, shuffle=False,
                                num_workers=cfg.num_workers, pin_memory=(device == 'cuda'))

        # 拟合标准化器
        scaler.fit(train_loader, key='proprio')
        action_scaler.fit(train_loader, key='action')

    # Test from test_dir
    if cfg.test_dir:
        ds_test = BehaviorCloningDataset(cfg.test_dir, expand_trajectories=cfg.expand_trajectories)
        if proprio_dim is None:
            proprio_dim = ds_test[0]['proprio'].numel()
        test_loader = DataLoader(ds_test, batch_size=cfg.batch_size, shuffle=False,
                                 num_workers=cfg.num_workers, pin_memory=(device=='cuda'))

    assert proprio_dim is not None, "Could not infer proprio feature dimension. Provide --train_dir or --test_dir with data."

    return train_loader, val_loader, test_loader, proprio_dim, scaler, action_scaler


@torch.no_grad()
def evaluate(model: BCPolicy, loader: DataLoader, scaler: Standardizer, action_scaler: Standardizer, device: torch.device) -> Dict[str, float]:
    model.eval()
    se_sum = 0.0
    n = 0
    se_per_dim = None
    for batch in loader:
        front = batch['front_rgb'].to(device, non_blocking=True)
        wrist = batch['wrist_rgb'].to(device, non_blocking=True)
        proprio = scaler.transform(batch['proprio']).to(device, non_blocking=True)
        pred = model(front, wrist, proprio)
        pred = action_scaler.inverse_transform(pred) # inverse transform for evaluation
        target = batch['action'].to(device, non_blocking=True)
        diff = pred - target
        se = (diff ** 2)
        if se_per_dim is None:
            se_per_dim = torch.zeros(se.shape[1], device=device)
        se_sum += se.sum().item()
        se_per_dim += se.sum(dim=0)
        n += pred.shape[0]

    mse = se_sum / (n * pred.shape[1])
    per_dim = (se_per_dim / n).tolist()
    return {"mse": float(mse), **{f"mse_dim_{i}": v for i, v in enumerate(per_dim)}}


def train_and_eval():
    set_seed(cfg.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader, val_loader, test_loader, proprio_dim, scaler, action_scaler = build_loaders(cfg)

    # Save standardizer early if training
    os.makedirs(cfg.out_dir, exist_ok=True)
    scaler_path = cfg.scaler_path or os.path.join(cfg.out_dir, 'scaler.pkl')
    action_scaler_path = os.path.join(cfg.out_dir, 'action_scaler.pkl')
    if not cfg.eval_only:
        scaler.save(scaler_path)
        action_scaler.save(action_scaler_path)

    # Build model
    model = BCPolicy(input_dim=proprio_dim,
                     hidden_fusion=cfg.hidden_fusion, 
                     output_dim=cfg.action_dim, 
                     pretrained_backbone=True,
                     dropout=cfg.dropout)
    model.to(device)
    print(f"Model params: {count_parameters(model):,}")

    if cfg.eval_only:
        assert cfg.ckpt and os.path.isfile(cfg.ckpt), "--eval_only requires a valid --ckpt"
        ckpt = torch.load(cfg.ckpt, map_location=device)
        model.load_state_dict(ckpt['model'])
        if cfg.scaler_path:
            scaler = Standardizer.load(cfg.scaler_path)
        if test_loader is not None:
            metrics = evaluate(model, test_loader, scaler, action_scaler=action_scaler, device=device)
            print(json.dumps({"split": "test", **metrics}, indent=2))
        else:
            print("No --test_dir provided; nothing to evaluate.")
        return

    # Optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    scaler_amp = torch.amp.GradScaler('cuda',enabled=cfg.amp)

    best_val = math.inf
    best_path = os.path.join(cfg.out_dir, 'best.pt')

    train_losses = []
    val_losses = []
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running_se_per_dim = torch.zeros(cfg.action_dim, device=device)
        total_samples = 0

        for batch in train_loader:
            front = batch['front_rgb'].to(device)
            wrist = batch['wrist_rgb'].to(device)
            proprio = scaler.transform(batch['proprio']).to(device)
            target = action_scaler.transform(batch['action']).to(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=cfg.amp):
                pred = model(front, wrist, proprio)
                loss = F.mse_loss(pred, target)
            scaler_amp.scale(loss).backward()
            scaler_amp.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler_amp.step(optimizer)
            scaler_amp.update()

            # 累加 per-dim MSE
            se_batch = (pred - target) ** 2            # [batch_size, action_dim]
            running_se_per_dim += se_batch.sum(dim=0)  # 按维度累加
            total_samples += pred.shape[0]

        # epoch 完成后计算整体 MSE 和 per-dim MSE
        train_metrics = {'mse': running_se_per_dim.sum().item() / (total_samples * cfg.action_dim)}
        for i in range(cfg.action_dim):
            train_metrics[f'mse_dim_{i}'] = (running_se_per_dim[i].item() / total_samples)
        train_losses.append(train_metrics)

        print("预测动作：",pred[0].detach().cpu().numpy(), "目标动作：",target[0].detach().cpu().numpy())

        # Validation
        val_metrics = evaluate(model, val_loader, scaler, action_scaler, device) if val_loader is not None else {"mse": float('nan')}
        val_losses.append(val_metrics)

        scheduler.step()

        print(f"Epoch {epoch:03d}/{cfg.epochs}  train_mse={train_metrics['mse']:.6f}  val_mse={val_metrics['mse']:.6f}")

        # Checkpoint best
        if val_metrics['mse'] < best_val:
            best_val = val_metrics['mse']
            torch.save({
                'model': model.state_dict(),
                'model_kwargs': {
                    'input_dim': proprio_dim,
                    'output_dim': cfg.action_dim,
                    'hidden_fusion': cfg.hidden_fusion,
                },
                'epoch': epoch,
                'val_mse': val_metrics['mse']
            }, best_path)

    print(f"Best val MSE: {best_val:.6f}  (saved to {best_path})")
    plot_losses(train_losses=train_losses, val_losses=val_losses, action_dim=cfg.action_dim, out_dir=cfg.out_dir)

    # Evaluate best on test set (if provided)
    if test_loader is not None:
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt['model'])
        metrics = evaluate(model, test_loader, scaler, device)
        print(json.dumps({"split": "test", **metrics}, indent=2))

# ------------------------------
# CLI
def parse_args():
    p = argparse.ArgumentParser(description="Multimodal Behavior Cloning pipeline")
    p.add_argument('--train_dir', type=str, default=None, help='Directory with training .npz files')
    p.add_argument('--test_dir', type=str, default=None, help='Directory with cross-dataset test .npz files')
    p.add_argument('--out_dir', type=str, default='./bc_ckpts')
    p.add_argument('--epochs', type=int, default=70)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--dropout', type=float, default=0.2)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--val_split', type=float, default=0.3)
    p.add_argument('--amp', action='store_true', help='Enable mixed precision (default off unless flag provided)')
    p.add_argument('--expand_trajectories', action='store_true', help='Expand (T,...) arrays into per-timestep samples')

    # Eval-only
    p.add_argument('--eval_only', action='store_true')
    p.add_argument('--ckpt', type=str, default=None)
    p.add_argument('--scaler', dest='scaler_path', type=str, default=None)
    p.add_argument('--hidden_fusion', type=Sequence, default=(512, 512, 256))
    p.add_argument('--action_dim', type=int, default=7,help='Dimension of action space (default 7)')

    args = p.parse_args()

    # By default, AMP is ON per the paper-like defaults if CUDA available
    if torch.cuda.is_available() and not args.amp:
        args.amp = True
    return args


if __name__ == '__main__':
    cfg = parse_args()
    train_and_eval(cfg)

