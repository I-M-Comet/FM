"""
eval_linear_probe_npy.py

Linear-probe evaluation for EEG foundation model checkpoints using cached .npy arrays (memmap).
This script assumes you already extracted each dataset split into:
  <DATASET_CACHE_ROOT>/<split>/eeg.npy
  <DATASET_CACHE_ROOT>/<split>/coords.npy
  <DATASET_CACHE_ROOT>/<split>/label.npy

For TUAB: only train/test exist. We make val by stratified split from train (default 0.2).

Key goals:
  - Avoid loading huge .npz into RAM (use memmap .npy).
  - Use GPU for feature extraction (encoder frozen).
  - Train only a linear classifier head (LP) with early stopping.

Usage example:
  CUDA_VISIBLE_DEVICES=0 python -m eeg_fm.eval \
    --ckpts /data/ssd/checkpoints/A2_1_B_lr3.5e-4/final/teacher /data/ssd/checkpoints/A2_1_B_lr9.5e-5/final/teacher \
    --tasks tuab isruc mi \
    --tuab_root /mnt/e/TUAB_npy \
    --isruc_root /mnt/e/ISRUC_npy \
    --mi_root /mnt/e/PhysioNetMI_npy \
    --seed 42 --num_workers 2 \
    --feat_batch_size 64 --lp_batch_size 1024 \
    --amp --apply_rescale \
    --epochs 50 --patience 10 \
    --lrs 3e-4 1e-3 3e-3 1e-2 --tune_lr_on first_ckpt \
    --wandb_project EEG_FM --wandb_group A2_1 --wandb_name A2_1_B \ 
    --out_csv /data/ssd/eval_results/A2_1_B_eval.csv

CUDA_VISIBLE_DEVICES=0 python -m eeg_fm.eval \
    --ckpts /data/ssd/checkpoints/A2_1_B_lr3.5e-4/final/teacher /data/ssd/checkpoints/A2_1_B_lr9.5e-5/final/teacher \
    --tasks tuab isruc mi \
    --tuab_root /mnt/e/TUAB_npy \
    --isruc_root /mnt/e/ISRUC_npy \
    --mi_root /mnt/e/PhysioNetMI_npy \
    --amp --apply_rescale \
    --lrs 3e-4 1e-3 3e-3 1e-2 --tune_lr_on first_ckpt \
    --wandb_project EEG_FM --wandb_group A2_1 --wandb_name A2_1_B \ 

"""

from __future__ import annotations

import argparse
import csv
import os
import random
import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset

# Import your encoder
# Expected: EEGEncoder.from_pretrained(<dir with config.json + pytorch_model.bin>)
from .model import EEGEncoder
try:
    import wandb
except Exception:
    wandb = None

# -------------------------
# utils
# -------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def add_bool_arg(parser: argparse.ArgumentParser, name: str, default: bool, help_text: str) -> None:
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument(f"--{name}", dest=name, action="store_true", help=help_text)
    group.add_argument(f"--no_{name}", f"--no-{name}", dest=name, action="store_false", help=f"Disable {name.replace('_', ' ')}.")
    parser.set_defaults(**{name: default})


def feature_dim_from_pool(d_model: int, pool: str) -> int:
    pool = str(pool).lower().strip()
    if pool == "mean":
        return int(d_model)
    if pool in {"mean_std", "tc_mean_std", "ct_mean_std"}:
        return 2 * int(d_model)
    if pool == "tc_ct_mean_std":
        return 4 * int(d_model)
    raise ValueError(f"Unsupported pool={pool}")


def resolve_task_roots(task_root: str = "", tuab_root: str = "", isruc_root: str = "", mi_root: str = "") -> Dict[str, str]:
    base = Path(task_root).expanduser() if task_root else None

    def _pick(explicit: str, suffix: str) -> str:
        if explicit:
            return str(Path(explicit).expanduser())
        if base is not None:
            return str(base / suffix)
        return ""

    tuab = _pick(tuab_root, "TUAB_npy")
    isruc = _pick(isruc_root, "ISRUC_npy")
    mi = _pick(mi_root, "PhysioNetMI_npy")

    return {
        "tuab": tuab,
        "isruc": isruc,
        "mi": mi,
        "physionetmi": mi,
        "physionet_mi": mi,
    }


def resolve_ckpt_dirs(args: argparse.Namespace) -> List[str]:
    ckpts = getattr(args, "ckpts", None)
    if ckpts:
        return [str(Path(p).expanduser()) for p in ckpts]
    ckpt = getattr(args, "ckpt", None)
    if ckpt:
        return [str(Path(ckpt).expanduser())]
    raise ValueError("Missing checkpoint path. Provide --ckpt or --ckpts.")


def compute_num_patches(T: int, patch_samples: int, hop_samples: int) -> int:
    if T < patch_samples:
        return 0
    return int((T - patch_samples) // hop_samples + 1)


@torch.no_grad()
def rescale_small_segments(
    x: torch.Tensor,            # (B,C,T)
    target_rms: float = 1.0,
    rms_low: float = 0.5,
    rms_floor: float = 0.05,
    gain_max: float = 8.0,
    clip: float = 15.0,
) -> torch.Tensor:
    x32 = x.float()
    rms = torch.sqrt(torch.mean(x32 * x32, dim=-1, keepdim=True) + 1e-8)  # (B,C,1)
    need = (rms < rms_low).to(x32.dtype)
    gain = (target_rms / rms.clamp_min(rms_floor)).clamp(1.0 / gain_max, gain_max)
    x32 = x32 * (1.0 + need * (gain - 1.0))
    if clip and clip > 0:
        x32 = x32.clamp(-clip, clip)
    return x32.to(dtype=x.dtype)


def stratified_split_indices(y: np.ndarray, val_ratio: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    assert y.ndim == 1
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    train_idx: List[int] = []
    val_idx: List[int] = []
    all_idx = np.arange(len(y))
    for c in classes:
        idx_c = all_idx[y == c]
        rng.shuffle(idx_c)
        n_val = int(round(len(idx_c) * float(val_ratio)))
        n_val = min(max(n_val, 1), max(len(idx_c) - 1, 1))  # keep at least 1 for train/val if possible
        val_idx.extend(idx_c[:n_val].tolist())
        train_idx.extend(idx_c[n_val:].tolist())
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return np.array(train_idx, dtype=np.int64), np.array(val_idx, dtype=np.int64)


def accuracy_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    pred = logits.argmax(dim=-1)
    return float((pred == y).float().mean().item())

@torch.no_grad()
def _confusion_matrix(y_true: torch.Tensor, y_pred: torch.Tensor, n_classes: int, device: torch.device) -> torch.Tensor:
    """Confusion matrix (K,K). Rows=true, Cols=pred."""
    y_true = y_true.to(device=device, dtype=torch.long)
    y_pred = y_pred.to(device=device, dtype=torch.long)
    idx = y_true * n_classes + y_pred
    cm = torch.bincount(idx, minlength=n_classes * n_classes)
    return cm.view(n_classes, n_classes)

@torch.no_grad()
def weighted_f1_from_cm(cm: torch.Tensor) -> float:
    """Weighted F1 from confusion matrix."""
    cm = cm.to(torch.float32)
    tp = torch.diagonal(cm)
    support = cm.sum(dim=1)
    fp = cm.sum(dim=0) - tp
    fn = support - tp
    denom = (2 * tp + fp + fn).clamp_min(1e-12)
    f1 = (2 * tp) / denom
    w = support.clamp_min(0.0)
    return float((f1 * w).sum().div(w.sum().clamp_min(1e-12)).item())

# -------------------------
# Dataset (memmap .npy)
# -------------------------
class NpySplitDataset(Dataset):
    def __init__(self, split_dir: str):
        split_dir = str(split_dir)
        self.split_dir = split_dir
        eeg_path = os.path.join(split_dir, "eeg.npy")
        coords_path = os.path.join(split_dir, "coords.npy")
        label_path = os.path.join(split_dir, "label.npy")

        if not os.path.exists(eeg_path):
            raise FileNotFoundError(eeg_path)
        if not os.path.exists(coords_path):
            raise FileNotFoundError(coords_path)
        if not os.path.exists(label_path):
            raise FileNotFoundError(label_path)

        # mmap: header + lazy pages
        self.eeg = np.load(eeg_path, mmap_mode="r")
        self.coords = np.load(coords_path, mmap_mode="r")
        self.y = np.load(label_path, mmap_mode="r")

        if self.eeg.shape[0] != self.coords.shape[0] or self.eeg.shape[0] != self.y.shape[0]:
            raise ValueError(f"N mismatch: eeg={self.eeg.shape} coords={self.coords.shape} y={self.y.shape}")

        # Expect (N,C,T) and (N,C,3)
        if self.eeg.ndim != 3:
            raise ValueError(f"eeg must be (N,C,T), got {self.eeg.shape}")
        if self.coords.ndim != 3 or self.coords.shape[-1] != 3:
            raise ValueError(f"coords must be (N,C,3), got {self.coords.shape}")

    def __len__(self) -> int:
        return int(self.eeg.shape[0])

    def __getitem__(self, idx: int):
        # Return numpy views; collate will stack
        return self.eeg[idx], self.coords[idx], int(self.y[idx])


def make_collate_eeg(coord_scale: float = 10.0):
    """
    Collate factory so coord scaling can be configured from CLI.
    coord_scale should match your pretraining pipeline.
    """
    coord_scale = float(coord_scale)

    def _collate(batch):
        eegs, coords, ys = zip(*batch)
        eeg = torch.from_numpy(np.stack(eegs, axis=0))
        coord = torch.from_numpy(np.stack(coords, axis=0))
        y = torch.tensor(ys, dtype=torch.long)

        # Match training decode_sample: eeg fp16, coords fp32 and (optionally) scaled.
        if eeg.dtype != torch.float16:
            eeg = eeg.to(torch.float16)
        coord = coord.to(torch.float32) * coord_scale

        return eeg, coord, y

    return _collate



# -------------------------
# Encoder call helpers + pooling
# -------------------------
def _call_embed_from_indices(encoder: EEGEncoder, **kwargs):
    """
    Backward/forward compatible wrapper:
      - Some versions return (tok, pad, rope_pos)
      - Others may return (tok, pad, rope_pos, chan_idx) etc.
    """
    out = encoder.embed_from_indices(**kwargs)
    if isinstance(out, (list, tuple)):
        if len(out) == 3:
            tok, pad, rope_pos = out
            chan_idx = None
            return tok, pad, rope_pos, chan_idx
        if len(out) == 4:
            tok, pad, rope_pos, chan_idx = out
            return tok, pad, rope_pos, chan_idx
    raise RuntimeError(f"Unexpected embed_from_indices return: type={type(out)}")


def _call_encoder_forward(
    encoder: EEGEncoder,
    tok: torch.Tensor,
    pad: torch.Tensor,
    rope_pos: torch.Tensor,
    coords: torch.Tensor,
    chan_idx: Optional[torch.Tensor],
) -> torch.Tensor:
    """
    Call encoder.forward with only the kwargs it supports.
    This avoids signature mismatch across model versions.
    """
    sig = inspect.signature(encoder.forward)
    kwargs = {}
    if "padding_mask" in sig.parameters:
        kwargs["padding_mask"] = pad
    if "rope_pos" in sig.parameters:
        kwargs["rope_pos"] = rope_pos
    # optional args in some variants
    if ("coords" in sig.parameters):
        kwargs["coords"] = coords
    if ("chan_idx" in sig.parameters) and (chan_idx is not None):
        kwargs["chan_idx"] = chan_idx
    return encoder(tok, **kwargs)


def pool_tokens(
    z: torch.Tensor,                 # (B,L,D)
    pad: Optional[torch.Tensor],      # (B,L) True=pad
    pool: str,
    *,
    C: Optional[int] = None,
    P: Optional[int] = None,
) -> torch.Tensor:
    """
    Pool token embeddings into a single feature vector per sample.

    pool:
      - mean: global mean (legacy; can collapse for large L)
      - mean_std: concat(mean, std) over tokens (robust default)
      - tc_mean_std: reshape to (B,P,C,D), pool over C first -> (B,P,D), then concat(time-mean, time-std)
      - ct_mean_std: reshape to (B,P,C,D), pool over P first -> (B,C,D), then concat(chan-mean, chan-std)
      - tc_ct_mean_std: concat(tc_mean_std, ct_mean_std)  (often best for EEG)

    Notes:
      - For tc*/ct* modes, this assumes token order is time-major: for each time t, channels 0..C-1.
        Our extract_features builds indices exactly in that order.
    """
    pool = str(pool).lower().strip()
    B, L, D = z.shape

    if pad is None:
        valid = torch.ones((B, L), device=z.device, dtype=torch.bool)
    else:
        valid = ~pad

    def masked_mean_and_std(x: torch.Tensor, v: torch.Tensor, dim: int):
        # x: (..., dim, ...)
        w = v.to(x.dtype)
        denom = w.sum(dim=dim, keepdim=True).clamp_min(1.0)
        mean = (x * w.unsqueeze(-1)).sum(dim=dim, keepdim=True) / denom
        var = ((x - mean) ** 2 * w.unsqueeze(-1)).sum(dim=dim, keepdim=True) / denom
        std = torch.sqrt(torch.clamp(var, min=0.0) + 1e-6)
        return mean.squeeze(dim), std.squeeze(dim)

    if pool == "mean":
        w = valid.to(z.dtype)
        denom = w.sum(dim=1, keepdim=True).clamp_min(1.0)
        return (z * w[..., None]).sum(dim=1) / denom  # (B,D)

    if pool == "mean_std":
        w = valid.to(z.dtype)
        denom = w.sum(dim=1, keepdim=True).clamp_min(1.0)
        mean = (z * w[..., None]).sum(dim=1) / denom
        mean2 = (z * z * w[..., None]).sum(dim=1) / denom
        var = torch.clamp(mean2 - mean * mean, min=0.0)
        std = torch.sqrt(var + 1e-6)
        return torch.cat([mean, std], dim=-1)  # (B,2D)

    # factorized pooling requires known grid
    if C is None or P is None:
        raise ValueError(f"pool={pool} requires C and P, got C={C}, P={P}")
    if L != int(C) * int(P):
        raise ValueError(f"pool={pool} requires L==C*P. Got L={L}, C={C}, P={P}")

    C = int(C)
    P = int(P)

    # reshape (time-major): (B,P,C,D)
    z_grid = z.view(B, P, C, D)
    v_grid = valid.view(B, P, C)

    # tc: pool over channels -> (B,P,D), valid over time -> (B,P)
    if pool in ("tc_mean_std", "tc_ct_mean_std"):
        # per-time mean across channels
        v_c = v_grid  # (B,P,C)
        w_c = v_c.to(z.dtype)
        denom_c = w_c.sum(dim=2, keepdim=True).clamp_min(1.0)  # (B,P,1)
        z_t = (z_grid * w_c.unsqueeze(-1)).sum(dim=2) / denom_c  # (B,P,D)
        v_t = (v_c.any(dim=2))  # (B,P)
        mean_t, std_t = masked_mean_and_std(z_t, v_t, dim=1)  # (B,D) each
        feat_tc = torch.cat([mean_t, std_t], dim=-1)  # (B,2D)

    if pool == "tc_mean_std":
        return feat_tc

    # ct: pool over time -> (B,C,D), valid over channels -> (B,C)
    if pool in ("ct_mean_std", "tc_ct_mean_std"):
        v_p = v_grid  # (B,P,C)
        w_p = v_p.to(z.dtype)
        denom_p = w_p.sum(dim=1, keepdim=True).clamp_min(1.0)  # (B,1,C)
        z_c = (z_grid * w_p.unsqueeze(-1)).sum(dim=1) / denom_p.squeeze(1).unsqueeze(-1)  # (B,C,D)
        v_c2 = (v_p.any(dim=1))  # (B,C)
        mean_c, std_c = masked_mean_and_std(z_c, v_c2, dim=1)  # (B,D)
        feat_ct = torch.cat([mean_c, std_c], dim=-1)  # (B,2D)

    if pool == "ct_mean_std":
        return feat_ct

    if pool == "tc_ct_mean_std":
        return torch.cat([feat_tc, feat_ct], dim=-1)  # (B,4D)

    raise ValueError(f"Unknown pool: {pool}")

def normalize_features(
    X_train: torch.Tensor,
    X_val: torch.Tensor,
    X_test: torch.Tensor,
    mode: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Feature normalization applied after extraction (on CPU tensors)."""
    mode = (mode or "none").lower().strip()
    if mode == "none":
        return X_train, X_val, X_test
    if mode == "zscore":
        mu = X_train.mean(dim=0, keepdim=True)
        sd = X_train.std(dim=0, keepdim=True).clamp_min(1e-6)
        return (X_train - mu) / sd, (X_val - mu) / sd, (X_test - mu) / sd
    if mode == "l2":
        return F.normalize(X_train, p=2, dim=-1), F.normalize(X_val, p=2, dim=-1), F.normalize(X_test, p=2, dim=-1)
    raise ValueError(f"Unknown feat_norm={mode}")




# -------------------------
# Feature extraction
# -------------------------
@torch.no_grad()
def extract_features(
    encoder: EEGEncoder,
    ds: Dataset,
    device: torch.device,
    feat_batch_size: int,
    num_workers: int,
    pin_memory: bool,
    amp: bool,
    apply_rescale: bool,
    rescale_kwargs: Dict[str, float],
    pool: str,
    coord_scale: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      X: (N,D_out) float32 CPU
      y: (N,) long CPU
    """
    encoder.eval()

    loader_kwargs = dict(
        batch_size=feat_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        collate_fn=make_collate_eeg(coord_scale),
        persistent_workers=(num_workers > 0),
    )
    # prefetch_factor is only valid when num_workers > 0
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = 2

    loader = DataLoader(ds, **loader_kwargs)

    # Determine token grid from first sample
    eeg0, coord0, _ = ds[0]
    C = int(eeg0.shape[0])
    T = int(eeg0.shape[1])
    patch_samples = int(getattr(encoder, "patch_samples"))
    hop_samples = int(getattr(encoder, "hop_samples"))
    P = compute_num_patches(T, patch_samples, hop_samples)
    if P <= 0:
        raise RuntimeError(f"Invalid P={P} from T={T}, patch_samples={patch_samples}, hop_samples={hop_samples}")

    D = int(encoder.cfg.d_model)
    N = len(ds)
    feat_dim = feature_dim_from_pool(D, pool)
    X = torch.empty((N, feat_dim), dtype=torch.float32)  # CPU
    y_all = torch.empty((N,), dtype=torch.long)

    # Precompute packed indices (time-major: t then c) on DEVICE
    # L = C*P
    L = C * P
    c_base = torch.arange(C, device=device, dtype=torch.long).repeat(P)                 # (L,)
    t_base = torch.arange(P, device=device, dtype=torch.long).repeat_interleave(C)      # (L,)

    offset = 0
    autocast = torch.amp.autocast

    for eeg, coord, y in loader:
        B = eeg.shape[0]
        eeg = eeg.to(device, non_blocking=True)
        coord = coord.to(device, non_blocking=True)

        if apply_rescale:
            eeg = rescale_small_segments(eeg, **rescale_kwargs)

        # Build batch packed idx
        c_idx = c_base[None, :].expand(B, L)
        t_idx = t_base[None, :].expand(B, L)
        pad = torch.zeros((B, L), dtype=torch.bool, device=device)

        with autocast(device.type, dtype=torch.bfloat16, enabled=(amp and device.type == "cuda")):
            tok, pad2, rope, chan = _call_embed_from_indices(
                encoder,
                x=eeg,  # keep original dtype; autocast handles bf16
                coords=coord,
                c_idx=c_idx,
                t_idx=t_idx,
                pad=pad,
                freq_mask_bins=None,
                freq_domain_drop=None,
            )
            z = _call_encoder_forward(encoder, tok=tok, pad=pad2, rope_pos=rope, coords=coord, chan_idx=chan)  # (B,L,D)

            # pool token embeddings into a single vector
            feat = pool_tokens(z, pad2, pool=pool, C=C, P=P)

        feat_cpu = feat.detach().float().cpu()
        X[offset : offset + B] = feat_cpu
        y_all[offset : offset + B] = y.cpu()
        offset += B

    assert offset == N
    return X, y_all


# -------------------------
# Linear probe training
# -------------------------
@dataclass
class LPResult:
    best_epoch: int
    best_val_acc: float
    test_acc_at_best: float
    best_val_f1w: float
    test_f1w_at_best: float
    lr: float
    # Debugging / sanity: did the head actually learn?
    train_acc_at_best: float
    train_acc_last: float
    last_epoch: int


def train_linear_probe(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    n_classes: int,
    device: torch.device,
    lr: float,
    epochs: int,
    patience: int,
    batch_size: int,
    weight_decay: float = 0.0,
    class_weight: Optional[str] = None,  # "balanced" or None
) -> LPResult:
    head = nn.Linear(X_train.shape[1], n_classes).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)

    if class_weight == "balanced":
        # inverse frequency weights from train labels
        with torch.no_grad():
            counts = torch.bincount(y_train.cpu(), minlength=n_classes).float()
            w = (counts.sum() / counts.clamp_min(1.0))
            w = (w / w.mean()).to(device)
        loss_fn = nn.CrossEntropyLoss(weight=w)
    else:
        loss_fn = nn.CrossEntropyLoss()

    # DataLoaders over features
    train_ds = TensorDataset(X_train, y_train)
    val_ds = TensorDataset(X_val, y_val)
    test_ds = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True, drop_last=False)

    best_val = -1.0
    best_epoch = -1
    best_state = None
    best_test = -1.0
    best_val_f1w = -1.0
    best_test_f1w = -1.0
    bad = 0

    train_acc_at_best = 0.0
    train_acc_last = 0.0
    last_epoch = 0

    for ep in range(1, epochs + 1):
        head.train()
        correct_tr = 0
        total_tr = 0
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logits = head(xb)
            # train acc (for sanity; computed on-the-fly while weights change)
            pred = logits.argmax(dim=-1)
            correct_tr += int((pred == yb).sum().item())
            total_tr += int(yb.numel())
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()

        train_acc = correct_tr / max(1, total_tr)
        train_acc_last = float(train_acc)
        last_epoch = int(ep)

        # eval val
        head.eval()
        with torch.no_grad():
            # val
            tot = 0
            correct = 0
            cm_val = torch.zeros((n_classes, n_classes), dtype=torch.long, device=device)
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                logits = head(xb)
                pred = logits.argmax(dim=-1)
                correct += int((pred == yb).sum().item())
                tot += int(yb.numel())
                cm_val += _confusion_matrix(yb, pred, n_classes=n_classes, device=device)
            val_acc = correct / max(1, tot)
            val_f1w = weighted_f1_from_cm(cm_val)

            # test
            tot = 0
            correct = 0
            cm_test = torch.zeros((n_classes, n_classes), dtype=torch.long, device=device)
            for xb, yb in test_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                logits = head(xb)
                pred = logits.argmax(dim=-1)
                correct += int((pred == yb).sum().item())
                tot += int(yb.numel())
                cm_test += _confusion_matrix(yb, pred, n_classes=n_classes, device=device)
            test_acc = correct / max(1, tot)
            test_f1w = weighted_f1_from_cm(cm_test)

        if val_acc > best_val + 1e-6:
            best_val = float(val_acc)
            best_epoch = int(ep)
            best_test = float(test_acc)
            best_val_f1w = float(val_f1w)
            best_test_f1w = float(test_f1w)
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
            train_acc_at_best = float(train_acc_last)
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        head.load_state_dict(best_state)

    return LPResult(
        best_epoch=best_epoch,
        best_val_acc=best_val,
        test_acc_at_best=best_test,
        best_val_f1w=best_val_f1w,
        test_f1w_at_best=best_test_f1w,
        lr=float(lr),
        train_acc_at_best=float(train_acc_at_best),
        train_acc_last=float(train_acc_last),
        last_epoch=int(last_epoch),
    )


# -------------------------
# Task wrappers
# -------------------------
def load_task_splits(task: str, root: str, seed: int, tuab_val_ratio: float) -> Tuple[Dataset, Dataset, Dataset, int]:
    """
    Returns (train_ds, val_ds, test_ds, n_classes)
    """
    task = task.lower().strip()
    root = str(root)
    if task == "tuab":
        train = NpySplitDataset(os.path.join(root, "train"))
        test = NpySplitDataset(os.path.join(root, "test"))

        # make val from train
        # Load labels lazily with memmap -> materialize only y (small)
        y_train = np.asarray(train.y, dtype=np.int64)
        tr_idx, va_idx = stratified_split_indices(y_train, val_ratio=tuab_val_ratio, seed=seed)

        train_ds = torch.utils.data.Subset(train, tr_idx.tolist())
        val_ds = torch.utils.data.Subset(train, va_idx.tolist())
        test_ds = test
        n_classes = 2
        return train_ds, val_ds, test_ds, n_classes

    elif task == "isruc":
        train_ds = NpySplitDataset(os.path.join(root, "train"))
        val_ds = NpySplitDataset(os.path.join(root, "val"))
        test_ds = NpySplitDataset(os.path.join(root, "test"))
        n_classes = 5
        return train_ds, val_ds, test_ds, n_classes

    elif task in ("mi", "physionetmi", "physionet_mi"):
        train_ds = NpySplitDataset(os.path.join(root, "train"))
        val_ds = NpySplitDataset(os.path.join(root, "val"))
        test_ds = NpySplitDataset(os.path.join(root, "test"))
        n_classes = 4
        return train_ds, val_ds, test_ds, n_classes

    else:
        raise ValueError(f"Unknown task: {task}")


def build_parser(
    add_help: bool = True,
    *,
    include_ckpt: bool = True,
    include_wandb: bool = True,
    include_seed: bool = True,
) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(add_help=add_help)

    if include_ckpt:
        ckpt_group = ap.add_mutually_exclusive_group(required=True)
        ckpt_group.add_argument("--ckpt", type=str, default=None, help="Single checkpoint dir containing config.json + pytorch_model.bin")
        ckpt_group.add_argument("--ckpts", type=str, nargs="+", default=None, help="One or more checkpoint dirs containing config.json + pytorch_model.bin")

    ap.add_argument("--tasks", type=str, nargs="+", default=["tuab", "isruc", "mi"])
    ap.add_argument("--task_root", type=str, default="", help="Base directory containing TUAB_npy/, ISRUC_npy/, and PhysioNetMI_npy/.")
    ap.add_argument("--tuab_root", type=str, default="", help="TUAB cache root containing train/ and test/ folders")
    ap.add_argument("--isruc_root", type=str, default="", help="ISRUC cache root containing train/val/test folders")
    ap.add_argument("--mi_root", type=str, default="", help="PhysioNetMI cache root containing train/val/test folders")

    ap.add_argument("--tuab_val_ratio", type=float, default=0.2)
    if include_seed:
        ap.add_argument("--seed", type=int, default=42)

    # feature extraction
    ap.add_argument("--feat_batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=2, help="Start with 0 (safe). Increase to 2/4 if stable.")
    ap.add_argument("--pin_memory", action="store_true")
    add_bool_arg(ap, "amp", default=True, help_text="Enable AMP for feature extraction.")

    # pooling / feature normalization
    ap.add_argument(
        "--pool",
        type=str,
        default="mean_std",
        choices=["mean", "mean_std", "tc_mean_std", "ct_mean_std", "tc_ct_mean_std"],
        help="Token pooling for linear probe features. mean_std is more robust than mean for large token counts.",
    )
    ap.add_argument(
        "--feat_norm",
        type=str,
        default="zscore",
        choices=["none", "zscore", "l2"],
        help="Normalize extracted features using train-set statistics (recommended: zscore).",
    )
    ap.add_argument("--coord_scale", type=float, default=10.0, help="Multiply coords by this factor before feeding the model. Must match pretraining.")

    # preprocessing
    add_bool_arg(ap, "apply_rescale", default=True, help_text="Apply same low-RMS rescale as pretraining (recommended).")
    ap.add_argument("--rescale_rms_low", type=float, default=0.5)
    ap.add_argument("--rescale_rms_floor", type=float, default=0.05)
    ap.add_argument("--rescale_gain_max", type=float, default=8.0)
    ap.add_argument("--rescale_clip", type=float, default=15.0)

    # linear probe
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--lp_batch_size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=3e-3, help="Used if --lrs is not provided")
    ap.add_argument("--lrs", type=float, nargs="*", default=None, help="LR grid (AdamW) for tuning")
    ap.add_argument("--tune_lr_on", type=str, default="first_ckpt", choices=["none", "first_ckpt"], help="Tune LR on first ckpt (per task) then reuse.")
    ap.add_argument("--class_weight", type=str, default="balanced", choices=["none", "balanced"])

    if include_wandb:
        ap.add_argument("--no_wandb", action="store_true", help="Disable Weights & Biases logging.")
        ap.add_argument("--wandb_project", type=str, default="EEG_FM")
        ap.add_argument("--wandb_name", type=str, default="")

    ap.add_argument("--out_csv", type=str, default=None, help="Path to output CSV file with results")
    return ap


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def run_eval(args: argparse.Namespace) -> List[Dict[str, object]]:
    seed = int(getattr(args, "seed", 42))
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[eval] device={device} amp={args.amp} num_workers={args.num_workers} pin_memory={args.pin_memory}")

    use_wandb = (wandb is not None) and (not bool(getattr(args, "no_wandb", False)))
    if use_wandb:
        run_name = getattr(args, "wandb_name", "") or None
        wandb.init(
            project=getattr(args, "wandb_project", "EEG_FM"),
            name=run_name,
            config={
                "tasks": args.tasks,
                "seed": seed,
                "feat_batch_size": args.feat_batch_size,
                "lp_batch_size": args.lp_batch_size,
                "epochs": args.epochs,
                "patience": args.patience,
                "lr_grid": (args.lrs if args.lrs else [args.lr]),
                "tune_lr_on": args.tune_lr_on,
                "class_weight": args.class_weight,
            },
        )

    roots = resolve_task_roots(
        task_root=getattr(args, "task_root", ""),
        tuab_root=getattr(args, "tuab_root", ""),
        isruc_root=getattr(args, "isruc_root", ""),
        mi_root=getattr(args, "mi_root", ""),
    )
    ckpt_dirs = resolve_ckpt_dirs(args)

    lr_grid = args.lrs if args.lrs and len(args.lrs) > 0 else [args.lr]
    csv_rows: List[Dict[str, object]] = []
    chosen_lr_per_task: Dict[str, float] = {}

    for ckpt_idx, ckpt_dir in enumerate(ckpt_dirs):
        ckpt_dir = str(ckpt_dir)
        ckpt_name = Path(ckpt_dir).name
        print()
        print(f"[ckpt] {ckpt_idx + 1}/{len(ckpt_dirs)} name={ckpt_name}")
        print(f"  path={ckpt_dir}")

        encoder = EEGEncoder.from_pretrained(ckpt_dir, map_location="cpu")
        encoder.to(device)
        encoder.eval()

        rescale_kwargs = dict(
            target_rms=1.0,
            rms_low=args.rescale_rms_low,
            rms_floor=args.rescale_rms_floor,
            gain_max=args.rescale_gain_max,
            clip=args.rescale_clip,
        )

        for task in args.tasks:
            t = task.lower().strip()
            root = roots.get(t, "")
            if not root:
                raise ValueError(f"Missing task root for task={t}. Use --task_root or the per-task root argument.")

            print()
            print(f"[task={t}] loading splits from {root}")
            train_ds, val_ds, test_ds, n_classes = load_task_splits(
                task=t,
                root=root,
                seed=seed,
                tuab_val_ratio=args.tuab_val_ratio,
            )

            print(f"[task={t}] extracting features: train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")
            Xtr, ytr = extract_features(
                encoder=encoder,
                ds=train_ds,
                device=device,
                feat_batch_size=args.feat_batch_size,
                num_workers=args.num_workers,
                pin_memory=args.pin_memory,
                amp=args.amp,
                apply_rescale=args.apply_rescale,
                rescale_kwargs=rescale_kwargs,
                pool=args.pool,
                coord_scale=args.coord_scale,
            )
            Xva, yva = extract_features(
                encoder=encoder,
                ds=val_ds,
                device=device,
                feat_batch_size=args.feat_batch_size,
                num_workers=args.num_workers,
                pin_memory=args.pin_memory,
                amp=args.amp,
                apply_rescale=args.apply_rescale,
                rescale_kwargs=rescale_kwargs,
                pool=args.pool,
                coord_scale=args.coord_scale,
            )
            Xte, yte = extract_features(
                encoder=encoder,
                ds=test_ds,
                device=device,
                feat_batch_size=args.feat_batch_size,
                num_workers=args.num_workers,
                pin_memory=args.pin_memory,
                amp=args.amp,
                apply_rescale=args.apply_rescale,
                rescale_kwargs=rescale_kwargs,
                pool=args.pool,
                coord_scale=args.coord_scale,
            )

            Xtr, Xva, Xte = normalize_features(Xtr, Xva, Xte, mode=args.feat_norm)
            with torch.no_grad():
                counts = torch.bincount(ytr.cpu(), minlength=n_classes).float()
                maj = float((counts.max() / counts.sum().clamp_min(1.0)).item())
                feat_std_mean = float(Xtr.std(dim=0).mean().item())
                feat_abs_mean = float(Xtr.abs().mean().item())
            print(f"[task={t}] pool={args.pool} feat_norm={args.feat_norm} |X|mean={feat_abs_mean:.4f} std_dim_mean={feat_std_mean:.4f} maj={maj:.3f} counts={counts.tolist()}")

            if args.tune_lr_on == "first_ckpt" and t in chosen_lr_per_task:
                use_lrs = [chosen_lr_per_task[t]]
                print(f"[task={t}] using pre-chosen lr={use_lrs[0]:.2e}")
            else:
                use_lrs = lr_grid

            best: Optional[LPResult] = None
            for lr in use_lrs:
                r = train_linear_probe(
                    X_train=Xtr,
                    y_train=ytr,
                    X_val=Xva,
                    y_val=yva,
                    X_test=Xte,
                    y_test=yte,
                    n_classes=n_classes,
                    device=device,
                    lr=float(lr),
                    epochs=args.epochs,
                    patience=args.patience,
                    batch_size=args.lp_batch_size,
                    weight_decay=0.0,
                    class_weight=None if args.class_weight == "none" else "balanced",
                )
                print(
                    f"[task={t}] lr={lr:.2e} best_val={r.best_val_acc:.4f} test@best={r.test_acc_at_best:.4f} "
                    f"best_ep={r.best_epoch} train_last={r.train_acc_last:.4f} train@best={r.train_acc_at_best:.4f} stop_ep={r.last_epoch}"
                )
                if (best is None) or (r.best_val_acc > best.best_val_acc + 1e-6):
                    best = r

            assert best is not None
            if args.tune_lr_on == "first_ckpt" and ckpt_idx == 0:
                chosen_lr_per_task[t] = float(best.lr)
                print(f"[task={t}] chosen lr for reuse = {best.lr:.2e}")

            row = dict(
                ckpt=ckpt_name,
                ckpt_path=ckpt_dir,
                task=t,
                n_classes=n_classes,
                lr=float(best.lr),
                best_epoch=int(best.best_epoch),
                stop_epoch=int(best.last_epoch),
                train_acc_last=float(best.train_acc_last),
                train_acc_at_best=float(best.train_acc_at_best),
                val_acc=float(best.best_val_acc),
                val_f1w=float(best.best_val_f1w),
                test_acc=float(best.test_acc_at_best),
                test_f1w=float(best.test_f1w_at_best),
            )
            csv_rows.append(row)
            print(
                f"[task={t}] BEST: val_acc={best.best_val_acc:.4f} val_f1w={best.best_val_f1w:.4f} "
                f"test_acc={best.test_acc_at_best:.4f} test_f1w={best.test_f1w_at_best:.4f} "
                f"lr={best.lr:.2e} train_last={best.train_acc_last:.4f} train@best={best.train_acc_at_best:.4f} stop_ep={best.last_epoch}"
            )

    if use_wandb and csv_rows:
        cols = list(csv_rows[0].keys())
        table = wandb.Table(columns=cols)
        for r in csv_rows:
            table.add_data(*[r.get(c, None) for c in cols])
        wandb.log({"lp_results": table})
        wandb.finish()

    if args.out_csv and csv_rows:
        outp = Path(args.out_csv).expanduser()
        outp.parent.mkdir(parents=True, exist_ok=True)
        with open(outp, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            for r in csv_rows:
                writer.writerow(r)
        print()
        print(f"[eval] wrote csv: {outp}")

    print()
    print("[eval] done.")
    return csv_rows


def main(argv: Optional[Sequence[str]] = None) -> List[Dict[str, object]]:
    return run_eval(parse_args(argv))


if __name__ == "__main__":
    main()
