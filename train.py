# eeg_fm/train.py
from __future__ import annotations

import argparse
import copy
import json
import csv
import math
import os
import re
import random
import time
from pathlib import Path
from contextlib import ExitStack, nullcontext
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm

from .config import EEGModelConfig, TrainConfig
from .model import EEGEncoder, CrossAttentionPredictor
from .masking import (
    sample_jepa_target_mask,
    sample_freq_bin_mask,
    dilate_time_mask,
)
from .augment import apply_student_augmentations
from .data import find_shards, build_webdataset, ShapeBatcher #, AdaptiveTokenBucketBatcher

try:
    from accelerate import Accelerator
    from accelerate.utils import set_seed
except Exception:
    Accelerator = None


class MetricsWriter:
    """Write training metrics to disk when W&B is disabled.

    - metrics.jsonl: full logs as JSON lines (lossless)
    - metrics.csv: first-seen keys only (stable, lightweight)
    """

    def __init__(self, out_dir: str):
        os.makedirs(out_dir, exist_ok=True)
        self.jsonl_path = os.path.join(out_dir, "metrics.jsonl")
        self.csv_path = os.path.join(out_dir, "metrics.csv")
        self._jsonl_f = open(self.jsonl_path, "a", encoding="utf-8")
        self._csv_f = open(self.csv_path, "a", encoding="utf-8", newline="")
        self._csv_writer = None
        self._csv_fieldnames = None

    def write(self, step: int, logs: dict):
        row = dict(logs)
        row["step"] = int(step)
        row["wall_time"] = float(time.time())

        self._jsonl_f.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._jsonl_f.flush()

        if self._csv_writer is None:
            self._csv_fieldnames = sorted(row.keys())
            self._csv_writer = csv.DictWriter(self._csv_f, fieldnames=self._csv_fieldnames)
            if self._csv_f.tell() == 0:
                self._csv_writer.writeheader()

        csv_row = {k: row.get(k, "") for k in self._csv_fieldnames}
        self._csv_writer.writerow(csv_row)
        self._csv_f.flush()

    def close(self):
        try:
            self._jsonl_f.close()
        finally:
            self._csv_f.close()


def set_torch_flags_for_sdp():
    # flash / mem-efficient attention if available
    try:
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
    except Exception:
        pass


def cosine_warmup(step: int, warmup: int, total: int, base_lr: float) -> float:
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def token_wcc_lr(tokens_next: int, warmup_tokens: int, total_tokens: int, cooldown_tokens: int, base_lr: float, min_lr: float = 0.0) -> float:
    # Token-based warmup-constant-cooldown schedule.
    # Use tokens_next (tokens after the upcoming step) to mimic (step+1)/warmup behavior.
    t = max(0, int(tokens_next))
    T = max(1, int(total_tokens))
    W = max(0, int(warmup_tokens))
    C = max(0, int(cooldown_tokens))

    if W > T:
        W = T
    if C > (T - W):
        C = max(0, T - W)

    stable = max(0, T - W - C)

    if W > 0 and t < W:
        return float(base_lr) * float(t) / float(max(1, W))

    if t < (W + stable):
        return float(base_lr)

    if C > 0 and t < T:
        td = t - (W + stable)
        frac = float(td) / float(max(1, C))
        return float(base_lr) + (float(min_lr) - float(base_lr)) * frac

    return float(min_lr)


def vicreg_var_cov_loss(
    x: torch.Tensor,
    gamma: float = 1.0,
    var_weight: float = 1.0,
    cov_weight: float = 1.0,
    eps: float = 1e-4,
):
    # VICReg-style variance + covariance regularizer (no invariance term here).
    # x: (N,D)
    assert x.dim() == 2, f"expected (N,D), got {tuple(x.shape)}"
    N, D = x.shape
    if N <= 1:
        z = torch.zeros((), device=x.device, dtype=x.dtype)
        return z, z, z

    x = x - x.mean(dim=0, keepdim=True)

    std = torch.sqrt(x.var(dim=0, unbiased=False) + eps)
    var = torch.mean(F.relu(float(gamma) - std))

    cov_m = (x.T @ x) / float(max(1, N - 1))
    eye = torch.eye(D, device=x.device, dtype=torch.bool)
    off = cov_m.masked_select(~eye)
    cov = (off ** 2).mean()

    tot = float(var_weight) * var + float(cov_weight) * cov
    return tot, var, cov


def ema_momentum_schedule(step: int, total: int, m0: float, m1: float) -> float:
    progress = step / max(1, total)
    m = m1 - (m1 - m0) * (0.5 * (1.0 + math.cos(math.pi * progress)))
    return float(m)


@torch.no_grad()
def update_ema(teacher: torch.nn.Module, student: torch.nn.Module, m: float):
    for p_t, p_s in zip(teacher.parameters(), student.parameters()):
        p_t.data.mul_(m).add_(p_s.data, alpha=(1.0 - m))


@torch.no_grad()
def mask_to_packed_indices(mask: torch.Tensor, valid: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    mask/valid: (B,C,P) bool

    returns:
      c_idx: (B,Lmax) long, channel index (>=0 for valid positions; 0 for PAD)
      t_idx: (B,Lmax) long, time-patch index (>=0; 0 for PAD)
      pad:   (B,Lmax) bool, True=PAD
    Ordering: sort by (t, c) for determinism.
    """
    device = mask.device
    mask = mask & valid
    B, C, P = mask.shape
    lengths = mask.flatten(1).sum(dim=1)  # (B,)
    Lmax = int(lengths.max().item())
    if Lmax <= 0:
        # should not happen if masks are sane; fall back to 1 dummy token
        Lmax = 1

    c_idx = torch.zeros((B, Lmax), dtype=torch.long, device=device)
    t_idx = torch.zeros((B, Lmax), dtype=torch.long, device=device)
    pad = torch.ones((B, Lmax), dtype=torch.bool, device=device)

    for b in range(B):
        idx = torch.nonzero(mask[b], as_tuple=False)  # (L,2): [c,t]
        if idx.numel() == 0:
            continue
        # sort by time then channel
        key = idx[:, 1] * C + idx[:, 0]
        idx = idx[key.argsort()]
        l = idx.shape[0]
        c_idx[b, :l] = idx[:, 0]
        t_idx[b, :l] = idx[:, 1]
        pad[b, :l] = False

    return c_idx, t_idx, pad


@torch.no_grad()
def rescale_small_segments(
    x: torch.Tensor,            # (B,C,T) fp16/bf16/fp32
    target_rms: float = 1.0,
    rms_low: float = 0.5,
    rms_floor: float = 0.05,
    gain_max: float = 8.0,
    clip: float = 15.0,
) -> torch.Tensor:
    # fp32에서 통계 계산(안정)
    x32 = x.float()
    # rms = torch.sqrt(torch.mean(x32 * x32, dim=(1,2), keepdim=True) + 1e-8)  # (B,1,1)
    rms = torch.sqrt(torch.mean(x32 * x32, dim=-1, keepdim=True) + 1e-8)  # (B,C,1)

    # rms가 너무 작은 것만 보정
    need = (rms < rms_low).to(x32.dtype)  # (B,1,1) 0/1
    gain = (target_rms / rms.clamp_min(rms_floor)).clamp(1.0 / gain_max, gain_max)

    # need==1인 샘플만 스케일 적용
    x32 = x32 * (1.0 + need * (gain - 1.0))

    if clip and clip > 0:
        x32 = x32.clamp(-clip, clip)

    return x32.to(dtype=x.dtype)


def gather_channel_embeddings(x: torch.Tensor, c_idx: torch.Tensor, pad: torch.Tensor) -> torch.Tensor:
    """
    x: (B,C,D)
    c_idx: (B,L) long (>=0)
    pad: (B,L) bool
    returns: (B,L,D) with pad->0
    """
    B, C, D = x.shape
    B2, L = c_idx.shape
    assert B == B2
    idx = c_idx[..., None].expand(B, L, D)
    out = x.gather(dim=1, index=idx)
    return out.masked_fill(pad[..., None], 0.0)


def auto_tune_tokens_per_batch(
    student: EEGEncoder,
    predictor: CrossAttentionPredictor,
    model_cfg: EEGModelConfig,
    train_cfg: TrainConfig,
    accelerator: Accelerator,
) -> int:
    """
    NEW(1): Synthetic probe to estimate a safe tokens_per_batch for the *worst bucket* (seq_len ~= max_tokens).
    This is a heuristic. It tends to be conservative and avoids OOM surprises.

    Returns the (possibly updated) tokens_per_batch.
    """
    if not train_cfg.auto_tune_tokens_per_batch:
        return train_cfg.tokens_per_batch
    if accelerator.device.type != "cuda":
        return train_cfg.tokens_per_batch

    device = accelerator.device
    # worst-case per-sample tokens (bounded by model_cfg.max_tokens)
    L = int(min(train_cfg.auto_tune_probe_seq_len, model_cfg.max_tokens))
    L = max(256, L)
    Bp = int(max(1, train_cfg.auto_tune_probe_batch))

    # worst-case (memory) happens when context is largest -> target fraction is smallest.
    tgt_frac_min = float(min(train_cfg.time_mask_ratio_min, train_cfg.spatial_mask_ratio_min))
    ctx_frac = float(max(0.50, min(0.95, 1.0 - tgt_frac_min)))

    Lc = int(max(64, round(L * ctx_frac)))
    Lt = int(max(32, L - Lc))

    dtype = torch.bfloat16 if train_cfg.mixed_precision == "bf16" else torch.float16

    # synthetic tensors (no patch embed / FFT) just to measure transformer activations
    ctx = torch.randn((Bp, Lc, model_cfg.d_model), device=device, dtype=dtype)
    ctx_pad = torch.zeros((Bp, Lc), device=device, dtype=torch.bool)
    rope_ctx = torch.arange(Lc, device=device, dtype=torch.long)[None, :].expand(Bp, Lc)

    tgt_coord = torch.randn((Bp, Lt, model_cfg.d_model), device=device, dtype=dtype)
    tgt_pad = torch.zeros((Bp, Lt), device=device, dtype=torch.bool)
    rope_tgt = torch.arange(Lt, device=device, dtype=torch.long)[None, :].expand(Bp, Lt)

    # reset peak stats
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    student.train()
    predictor.train()

    # forward/backward
    # dummy chan/coords for hybrid blocks
    chan_ctx = torch.zeros((Bp, Lc), device=device, dtype=torch.long)
    coords_dummy = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype)[None, None, :].expand(Bp, 1, 3)

    z = student(ctx, padding_mask=ctx_pad, rope_pos=rope_ctx, chan_idx=chan_ctx, coords=coords_dummy)
    pred = predictor(z, ctx_pad, rope_ctx, tgt_coord, tgt_pad, rope_tgt)
    loss = (pred ** 2).mean()
    loss.backward()

    mem_used = torch.cuda.max_memory_allocated(device)
    total_mem = torch.cuda.get_device_properties(device).total_memory
    target_mem = int(total_mem * float(train_cfg.auto_tune_target_mem_frac))

    # clear grads to avoid later surprises
    student.zero_grad(set_to_none=True)
    predictor.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()

    tokens_probe = Bp * L
    if mem_used <= 0:
        return train_cfg.tokens_per_batch

    scale = target_mem / float(mem_used)
    tokens_safe = int(tokens_probe * scale * 0.90)  # extra safety margin
    tokens_safe = max(1024, tokens_safe)

    # round down for stability
    round_to = 256
    tokens_safe = (tokens_safe // round_to) * round_to

    if accelerator.is_main_process:
        print(f"[auto_tune] probe: B={Bp}, L={L} (Lc={Lc}, Lt={Lt}) mem_used={mem_used/1e9:.2f}GB "
              f"target_mem={target_mem/1e9:.2f}GB -> tokens_per_batch~{tokens_safe}")

    # only reduce (avoid unexpected huge changes)
    if train_cfg.tokens_per_batch > tokens_safe:
        return tokens_safe
    return train_cfg.tokens_per_batch


def build_parser(add_help: bool = True) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(add_help=add_help)
    p.add_argument("--model_cfg", type=str, default="")
    p.add_argument("--train_cfg", type=str, default="")

    # overrides
    p.add_argument("--data_root", type=str, default=None)
    p.add_argument("--cache_dir", type=str, default=None)
    p.add_argument("--output_dir", type=str, default=None)

    p.add_argument("--d_model", type=int, default=None)
    p.add_argument("--n_layers", type=int, default=None)
    p.add_argument("--n_heads", type=int, default=None)

    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight_decay", type=float, default=None)
    p.add_argument("--tokens_per_batch", type=int, default=None)
    p.add_argument("--grad_accum_steps", type=int, default=None)
    p.add_argument("--tokens_per_update", type=int, default=None)
    p.add_argument("--accum_tokens_basis", type=str, default=None)
    p.add_argument("--max_steps", type=int, default=None)
    p.add_argument("--warmup_steps", type=int, default=None)

    p.add_argument("--ema_momentum", type=float, default=None)

    # model-side ablations
    p.add_argument("--attn_qk_norm", type=str, default=None)  # off | l2 | rms | layernorm")
    p.add_argument("--mlp_type", type=str, default=None)
    p.add_argument("--norm_type", type=str, default=None)
    p.add_argument("--layerscale_init", type=float, default=None)
    p.add_argument("--full_attn_every", type=int, default=None)
    p.add_argument("--spatial_bias_degree", type=int, default=None)

    # masking ablations
    p.add_argument("--time_mask_style", type=int, default=None)  # 0: "single", 1: "multi", 2: "ssp"
    p.add_argument("--time_mask_ratio_min", type=float, default=None)
    p.add_argument("--time_mask_ratio_max", type=float, default=None)
    p.add_argument("--spatial_mask_ratio_min", type=float, default=None)
    p.add_argument("--spatial_mask_ratio_max", type=float, default=None)

    # repro
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--torch_deterministic", action="store_true")
    p.add_argument("--no_cudnn_benchmark", action="store_true")

    p.add_argument("--use_wandb", action="store_true")
    p.add_argument("--no_wandb", action="store_true")
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--wandb_project", type=str, default=None)

    # resume / init
    p.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Path to an Accelerate state directory (or a step_* dir containing accelerator_state_*).",
    )
    p.add_argument(
        "--init_from",
        type=str,
        default=None,
        help="Initialize weights from a step_* checkpoint directory (student/teacher/predictor), but reset optimizer.",
    )
    return p


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def _build_train_artifacts(train_cfg: TrainConfig) -> Dict[str, object]:
    final_dir = os.path.join(train_cfg.output_dir, "final")
    return {
        "output_dir": str(train_cfg.output_dir),
        "final_dir": final_dir,
        "student_dir": os.path.join(final_dir, "student"),
        "teacher_dir": os.path.join(final_dir, "teacher"),
        "predictor_path": os.path.join(final_dir, "predictor.pt"),
        "run_name": getattr(train_cfg, "run_name", None),
        "wandb_project": getattr(train_cfg, "wandb_project", None),
        "seed": int(getattr(train_cfg, "seed", 42)),
        "use_wandb": bool(getattr(train_cfg, "use_wandb", False)),
    }

def _resolve_accelerator_state_dir(p: Optional[str]) -> Optional[str]:
    if not p:
        return None
    path = Path(p).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"resume_from path does not exist: {path}")

    # If user points to a step_* dir, auto-pick accelerator_state_* inside.
    if path.is_dir():
        # direct state dir
        if path.name.startswith("accelerator_state"):
            return str(path)
        # search inside
        cand = [d for d in path.iterdir() if d.is_dir() and d.name.startswith("accelerator_state")]
        if cand:
            cand = sorted(cand, key=lambda x: x.name)
            return str(cand[-1])

    # If it's a file or something else, just return as-is (accelerate will complain).
    return str(path)


def _parse_step_from_any_path(text: str) -> Optional[int]:
    m = re.search(r"step_(\d+)", text)
    if m:
        return int(m.group(1))
    m = re.search(r"accelerator_state_(\d+)", text)
    if m:
        return int(m.group(1))
    return None


def _resume_step_from_trainer_state(state_dir: str) -> int:
    p = Path(state_dir).expanduser().resolve()
    # trainer_state.json is saved at step directory level
    candidates = [
        p / "trainer_state.json",
        p.parent / "trainer_state.json",
        p.parent.parent / "trainer_state.json",
    ]
    for c in candidates:
        if c.exists() and c.is_file():
            try:
                with open(c, "r", encoding="utf-8") as f:
                    d = json.load(f)
                if "next_global_step" in d:
                    return int(d["next_global_step"])
                if "global_step" in d:
                    return int(d["global_step"]) + 1
            except Exception:
                pass
    s = _parse_step_from_any_path(str(p))
    return int(s + 1) if s is not None else 0


def _resolve_init_step_dir(p: Optional[str]) -> Optional[str]:
    if not p:
        return None
    path = Path(p).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"init_from path does not exist: {path}")
    if not path.is_dir():
        raise ValueError(f"init_from must be a directory (step_*), got: {path}")
    return str(path)


def _resolve_weights_in_step_dir(step_dir: str) -> Tuple[str, Optional[str], str]:
    """Return (student_dir, teacher_dir_or_None, predictor_pt)."""
    p = Path(step_dir).expanduser().resolve()
    if not p.is_dir():
        raise ValueError(f"step_dir is not a directory: {p}")

    student_dirs = [d for d in p.iterdir() if d.is_dir() and d.name.startswith("student")]
    teacher_dirs = [d for d in p.iterdir() if d.is_dir() and d.name.startswith("teacher")]
    pred_pts = [f for f in p.iterdir() if f.is_file() and f.name.startswith("predictor") and f.suffix == ".pt"]

    if not student_dirs:
        raise FileNotFoundError(f"No student_* directory found in {p}")
    if not pred_pts:
        raise FileNotFoundError(f"No predictor_*.pt found in {p}")

    student_dirs = sorted(student_dirs, key=lambda x: x.name)
    pred_pts = sorted(pred_pts, key=lambda x: x.name)
    teacher_dir = sorted(teacher_dirs, key=lambda x: x.name)[-1] if teacher_dirs else None

    return str(student_dirs[-1]), (str(teacher_dir) if teacher_dir else None), str(pred_pts[-1])


def run_train(args: argparse.Namespace) -> Dict[str, object]:

    if Accelerator is None:
        raise RuntimeError("accelerate is not installed. pip install accelerate")

    model_cfg = EEGModelConfig() if not args.model_cfg else EEGModelConfig.from_json(args.model_cfg)
    train_cfg = TrainConfig() if not args.train_cfg else TrainConfig.from_json(args.train_cfg)

    # ---- overrides that affect seeding / determinism ----
    if args.seed is not None:
        train_cfg.seed = int(args.seed)
    if args.torch_deterministic:
        train_cfg.torch_deterministic = True
    if args.no_cudnn_benchmark:
        train_cfg.cudnn_benchmark = False

    # overrides
    if args.data_root is not None: train_cfg.data_root = args.data_root
    if args.cache_dir is not None: train_cfg.cache_dir = args.cache_dir
    if args.output_dir is not None: train_cfg.output_dir = args.output_dir

    if args.d_model is not None: model_cfg.d_model = args.d_model
    if args.n_layers is not None: model_cfg.n_layers = args.n_layers
    if args.n_heads is not None: model_cfg.n_heads = args.n_heads

    if args.attn_qk_norm is not None:
        model_cfg.attn_qk_norm = str(args.attn_qk_norm)
    if args.mlp_type is not None:
        model_cfg.mlp_type = str(args.mlp_type)
    if args.norm_type is not None:
        model_cfg.norm_type = str(args.norm_type)
    if args.layerscale_init is not None:
        model_cfg.layerscale_init = float(args.layerscale_init)

    if args.full_attn_every is not None:
        model_cfg.full_attn_every = int(args.full_attn_every)
    if args.spatial_bias_degree is not None:
        model_cfg.spatial_bias_degree = int(args.spatial_bias_degree)

    if args.lr is not None: train_cfg.lr = args.lr
    if args.weight_decay is not None: train_cfg.weight_decay = args.weight_decay
    if args.tokens_per_batch is not None: train_cfg.tokens_per_batch = args.tokens_per_batch
    if args.grad_accum_steps is not None: train_cfg.grad_accum_steps = args.grad_accum_steps
    if args.tokens_per_update is not None: train_cfg.tokens_per_update = int(args.tokens_per_update)
    if args.accum_tokens_basis is not None: train_cfg.accum_tokens_basis = str(args.accum_tokens_basis)
    if args.max_steps is not None: train_cfg.max_steps = args.max_steps
    if args.warmup_steps is not None: train_cfg.warmup_steps = int(args.warmup_steps)

    if args.ema_momentum is not None: train_cfg.ema_momentum = float(args.ema_momentum)

    if args.time_mask_style is not None: train_cfg.time_mask_style = int(args.time_mask_style)
    if args.time_mask_ratio_min is not None: train_cfg.time_mask_ratio_min = float(args.time_mask_ratio_min)
    if args.time_mask_ratio_max is not None: train_cfg.time_mask_ratio_max = float(args.time_mask_ratio_max)
    if args.spatial_mask_ratio_min is not None: train_cfg.spatial_mask_ratio_min = float(args.spatial_mask_ratio_min)
    if args.spatial_mask_ratio_max is not None: train_cfg.spatial_mask_ratio_max = float(args.spatial_mask_ratio_max)

    if args.use_wandb: train_cfg.use_wandb = True
    if args.no_wandb: train_cfg.use_wandb = False
    if args.run_name is not None: train_cfg.run_name = args.run_name
    if args.wandb_project is not None: train_cfg.wandb_project = args.wandb_project

    # resume / init overrides
    if getattr(args, 'resume_from', None) is not None:
        train_cfg.resume_from = args.resume_from
    if getattr(args, 'init_from', None) is not None:
        train_cfg.init_from = args.init_from

    resume_state_dir = _resolve_accelerator_state_dir(getattr(train_cfg, 'resume_from', None))
    init_step_dir = _resolve_init_step_dir(getattr(train_cfg, 'init_from', None))
    if resume_state_dir and init_step_dir:
        raise ValueError('Cannot use both resume_from and init_from at the same time')

    resume_step = _resume_step_from_trainer_state(resume_state_dir) if resume_state_dir else 0

    use_token_budget = bool(getattr(train_cfg, "tokens_per_update", 0) and int(train_cfg.tokens_per_update) > 0)
    grad_accum_steps = 1 if use_token_budget else int(train_cfg.grad_accum_steps)

    accelerator = Accelerator(
        # NOTE: we implement manual accumulation (token-budget or micro) in the training loop.
        # Keep accelerate's internal grad-accum at 1 to avoid accidental double-scaling.
        gradient_accumulation_steps=1,
        mixed_precision=train_cfg.mixed_precision,
        log_with="wandb" if train_cfg.use_wandb else None,
        project_dir=train_cfg.output_dir,
    )

    if accelerator.is_main_process:
        os.makedirs(train_cfg.output_dir, exist_ok=True)
        model_cfg.save_json(os.path.join(train_cfg.output_dir, "model_config.json"))
        train_cfg.save_json(os.path.join(train_cfg.output_dir, "train_config.json"))

    metrics_writer = None
    if train_cfg.use_wandb:
        accelerator.init_trackers(
            train_cfg.wandb_project,
            config={**model_cfg.to_dict(), **train_cfg.to_dict()},
            init_kwargs={"wandb": {"name": train_cfg.run_name}} if train_cfg.run_name else None,
        )
    else:
        metrics_writer = MetricsWriter(train_cfg.output_dir)

    set_torch_flags_for_sdp()

    # ---- seed fixing (recommended for tuning/ablations) ----
    seed = int(getattr(train_cfg, "seed", 42))
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed, device_specific=True)

    if bool(getattr(train_cfg, "torch_deterministic", False)):
        # WARNING: may reduce throughput and can raise errors for some ops.
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = bool(getattr(train_cfg, "cudnn_benchmark", True))

    # models (move to device before dataloader to allow auto-tune)
    student = EEGEncoder(model_cfg).to(accelerator.device)
    predictor = CrossAttentionPredictor(model_cfg).to(accelerator.device)

    # optional: auto-tune tokens_per_batch
    tuned = auto_tune_tokens_per_batch(student, predictor, model_cfg, train_cfg, accelerator)
    train_cfg.tokens_per_batch = tuned

    # optional: init weights from an existing step directory (weights-only)
    teacher_init_dir: Optional[str] = None
    if init_step_dir:
        student_dir, teacher_dir, predictor_pt = _resolve_weights_in_step_dir(init_step_dir)
        if accelerator.is_main_process:
            print(f"[init_from] student={student_dir} teacher={teacher_dir} predictor={predictor_pt}")
        # student
        sd = torch.load(os.path.join(student_dir, "pytorch_model.bin"), map_location="cpu")
        student.load_state_dict(sd, strict=True)
        # predictor
        pd = torch.load(predictor_pt, map_location="cpu")
        predictor.load_state_dict(pd, strict=True)
        # teacher (optional)
        teacher_init_dir = teacher_dir

    # EMA teacher (after potential tuning)
    teacher = copy.deepcopy(student)
    for p in teacher.parameters():
        p.requires_grad = False
    teacher.eval()

    if teacher_init_dir:
        tsd = torch.load(os.path.join(teacher_init_dir, 'pytorch_model.bin'), map_location='cpu')
        teacher.load_state_dict(tsd, strict=True)

    # data
    shards = find_shards(train_cfg.data_root, train_cfg.shard_glob)
    if len(shards) == 0:
        raise RuntimeError(f"No shards found under {train_cfg.data_root} with {train_cfg.shard_glob}")

    patch_samples = int(round(model_cfg.sample_rate * model_cfg.patch_seconds))
    hop_sec = float(getattr(model_cfg, "patch_hop_seconds", model_cfg.patch_seconds))
    if hop_sec <= 0:
        hop_sec = float(model_cfg.patch_seconds)
    hop_samples = int(round(model_cfg.sample_rate * hop_sec))
    hop_samples = max(1, hop_samples)

    ds = build_webdataset(
        shards=shards,
        cache_dir=train_cfg.cache_dir,
        shard_shuffle=train_cfg.shard_shuffle,
        sample_shuffle=train_cfg.sample_shuffle,
        long_crop_prob=getattr(train_cfg, "long_crop_prob", 0.2),
        long_crop_30_prob=getattr(train_cfg, "long_crop_30_prob", 0.5),
        max_tokens=model_cfg.max_tokens,
        patch_samples=patch_samples,
        hop_samples=hop_samples,
        enable_channel_grouping=train_cfg.enable_channel_grouping,
        limit_num_samples=train_cfg.limit_num_samples,
        cache_max_bytes=train_cfg.cache_max_bytes,
        post_split_shuffle=train_cfg.post_split_shuffle,
        eviction_interval=train_cfg.eviction_interval,
        data_mode=getattr(train_cfg, "data_mode", "finite"),
        seed=train_cfg.seed,
    )

    # bucket batching
    ds_batched = ShapeBatcher(
        dataset=ds,
        tokens_per_batch=train_cfg.tokens_per_batch,
        max_samples_per_batch=train_cfg.max_samples_per_batch,
        patch_samples=patch_samples,
        hop_samples=hop_samples,

        # flush 정책(추천 시작값)
        max_wait_samples=5000,          # 희귀 shape가 5000샘플 동안 배치 못 만들면 방출
        flush_check_every=256,          # 256샘플마다 expired 검사
        max_pending_samples=512,        # CPU 메모리 보호(중요!)
        max_pending_tokens=0,           # 필요하면 활성화(예: 2_000_000)

        shuffle_within_bucket=True,
        yield_incomplete=True,
    )

    import webdataset as wds
    loader = wds.WebLoader(ds_batched, batch_size=None, num_workers=train_cfg.num_workers)

    # optimizer
    opt = AdamW(
        list(student.parameters()) + list(predictor.parameters()),
        lr=train_cfg.lr,
        betas=train_cfg.betas,
        weight_decay=train_cfg.weight_decay,
    )

    # accelerate prepare
    student, teacher, predictor, opt, loader = accelerator.prepare(student, teacher, predictor, opt, loader)

    # resume full training state (models/optimizer/rng)
    if resume_state_dir:
        accelerator.wait_for_everyone()
        accelerator.load_state(resume_state_dir)
        # ensure teacher stays frozen
        for p in teacher.parameters():
            p.requires_grad = False
        teacher.eval()

    # freq bin centers for masking
    bin_centers = accelerator.unwrap_model(student).freq_feat.bin_centers_hz.detach().cpu()
    student.train()
    predictor.train()
    teacher.eval()

    pbar = tqdm(total=train_cfg.max_steps, disable=not accelerator.is_local_main_process)
    if resume_state_dir and int(resume_step) >= int(train_cfg.max_steps):
        if accelerator.is_main_process:
            print(f"[resume] resume_step ({resume_step}) >= max_steps ({train_cfg.max_steps}). Nothing to do.")
        return
    it = iter(loader)
    global_step = int(resume_step) if resume_state_dir else 0
    if global_step > 0:
        pbar.update(global_step)

    # accumulation state (NEW)
    # - token-budget (preferred) OR fallback to fixed grad_accum_steps
    manual_accum = bool(use_token_budget or (grad_accum_steps and grad_accum_steps > 1))

    budget_tokens = int(getattr(train_cfg, "tokens_per_update", 0) or 0)
    accum_basis = str(getattr(train_cfg, "accum_tokens_basis", "target")).lower()
    if not use_token_budget:
        budget_tokens = int(max(1, grad_accum_steps))
        accum_basis = "micro"
    accum_tokens = 0
    accum_micro = 0
    accum_bucket_keys = set()

    # token-based LR schedule state (optional)
    lr_sched = str(getattr(train_cfg, "lr_schedule", "cosine")).lower().strip()
    if lr_sched == "token_wcc" and (not use_token_budget or budget_tokens <= 0):
        if accelerator.is_main_process:
            print("[warn] lr_schedule=token_wcc requires token-budget accumulation (tokens_per_update>0). Falling back to cosine.")
        lr_sched = "cosine"

    tokens_seen_total = int(resume_step) * int(budget_tokens) if (resume_state_dir and lr_sched == "token_wcc") else 0
    accum_tokens_eff = 0  # effective tokens accumulated in the current optimizer update

    # cheap proxy accumulators (per optimizer step)
    proxy_enabled = bool(getattr(train_cfg, "log_proxies", True))
    if proxy_enabled:
        dev = accelerator.device
        proxy_n = torch.zeros((), device=dev, dtype=torch.float32)
        proxy_cos_sum = torch.zeros((), device=dev, dtype=torch.float32)
        proxy_pred_norm_sum = torch.zeros((), device=dev, dtype=torch.float32)
        proxy_pred_norm_sumsq = torch.zeros((), device=dev, dtype=torch.float32)
        proxy_tgt_norm_sum = torch.zeros((), device=dev, dtype=torch.float32)
        proxy_tgt_norm_sumsq = torch.zeros((), device=dev, dtype=torch.float32)
        proxy_last_pred_feat_std = torch.zeros((), device=dev, dtype=torch.float32)
        proxy_last_tgt_feat_std = torch.zeros((), device=dev, dtype=torch.float32)
        proxy_last_cos_mean = torch.zeros((), device=dev, dtype=torch.float32)

    # VICReg logging accumulators (per optimizer step)
    vic_log = float(getattr(train_cfg, "vicreg_weight", 0.0) or 0.0) > 0
    if vic_log:
        dev = accelerator.device
        vic_loss_sum = torch.zeros((), device=dev, dtype=torch.float32)
        vic_var_sum = torch.zeros((), device=dev, dtype=torch.float32)
        vic_cov_sum = torch.zeros((), device=dev, dtype=torch.float32)
        vic_w_sum = torch.zeros((), device=dev, dtype=torch.float32)

    # ------------------------------------------------------------
    # Optional: "true" multi-view training for long multi-crops
    # ------------------------------------------------------------
    mv_mode = str(getattr(train_cfg, "long_multicrop_mode", "off")).lower().strip()
    mv_weight = float(getattr(train_cfg, "multiview_loss_weight", 0.0) or 0.0)
    mv_only_mc = bool(getattr(train_cfg, "multiview_only_for_multicrop", True))
    mv_enabled = (mv_mode == "multiview") and (mv_weight > 0)

    mv_bank_any = {}      # uid -> teacher embedding (normalized, detached)
    mv_bank_global = {}   # uid -> teacher embedding from global view only
    if mv_enabled:
        dev = accelerator.device
        mv_pairs_sum = torch.zeros((), device=dev, dtype=torch.float32)
        mv_loss_pairs_sum = torch.zeros((), device=dev, dtype=torch.float32)  # sum(loss_i) over paired samples

    passes_completed = 0
    start_time = time.time()
    while global_step < train_cfg.max_steps:
        try:
            batch = next(it)
        except StopIteration:
            passes_completed += 1
            if accelerator.is_main_process:
                pbar.write(f"[data] pass {passes_completed} completed. Restarting data loader.")
            it = iter(loader)
            batch = next(it)

        x = batch["eeg"].to(dev)          # (B,C,T)
        coords = batch["coord"]   # (B,C,3)
        metas = batch.get("meta", None)  # optional list[dict]
        x = rescale_small_segments(x, target_rms=1.0, rms_low=0.5, rms_floor=0.05, gain_max=8.0, clip=15.0)
        n_channels = batch["n_channels"].to(x.device)
        n_patches = batch["n_patches"].to(x.device)
        B, C_max, _ = x.shape
        P_max = int(n_patches.max().item())

        # valid token mask (B,C,P)
        chan_ok = (torch.arange(C_max, device=x.device)[None, :, None] < n_channels[:, None, None])
        time_ok = (torch.arange(P_max, device=x.device)[None, None, :] < n_patches[:, None, None])
        valid = chan_ok & time_ok  # (B,C,P)

        # ---------------------------------
        # 0) decide whether to sync gradients this micro-batch
        # ---------------------------------
        bucket_key = (int(x.shape[1]), int(P_max))

        # We decide sync *after* we know token counts (needs mask),
        # but to avoid duplicating logic, we default to no_sync context and
        # override later.

        # 1) JEPA target mask (B,C,P)
        target_mask = sample_jepa_target_mask(
            coords=coords.to(x.device),
            n_channels=n_channels,
            n_patches=n_patches,
            mask_time_prob=train_cfg.mask_time_prob,
            mask_spatial_prob=train_cfg.mask_spatial_prob,
            time_ratio_range=(train_cfg.time_mask_ratio_min, train_cfg.time_mask_ratio_max),
            spatial_ratio_range=(train_cfg.spatial_mask_ratio_min, train_cfg.spatial_mask_ratio_max),
            time_mask_style=train_cfg.time_mask_style,
            time_mask_num_blocks=train_cfg.time_mask_num_blocks,
            time_mask_min_block_patches=train_cfg.time_mask_min_block_patches,
            time_ssp_keep_blocks=train_cfg.time_ssp_keep_blocks,
            time_ssp_min_keep_patches=train_cfg.time_ssp_min_keep_patches,
        )
        # optional dilation to reduce overlap leakage
        if train_cfg.mask_dilate_time and train_cfg.mask_dilate_time > 0:
            target_mask = dilate_time_mask(target_mask, dilation=int(train_cfg.mask_dilate_time))

        target_mask = target_mask & valid
        context_mask = (~target_mask) & valid

        # token counts for token-budget accumulation
        if manual_accum:
            if use_token_budget:
                if accum_basis == "valid":
                    tokens_this = int(B) * int(x.shape[1]) * int(P_max)
                elif accum_basis == "context":
                    tokens_this = int(context_mask.sum().item())
                else:  # default: "target"
                    tokens_this = int(target_mask.sum().item())
            else:
                tokens_this = 1
        else:
            tokens_this = 0

        if manual_accum:
            if use_token_budget:
                # always do at least 1 micro-batch per optimizer step
                if accum_tokens == 0 and tokens_this <= 0:
                    will_step = True
                else:
                    will_step = (accum_tokens + tokens_this) >= max(1, budget_tokens)
            else:
                will_step = (accum_micro + 1) >= max(1, budget_tokens)
        else:
            will_step = True

        # DDP gradient sync control
        no_sync = manual_accum and (not will_step) and (accelerator.num_processes > 1)

        # effective tokens contributed by this micro-batch (used for token-based LR schedule + regularizers)
        tokens_eff_this = int(tokens_this)
        if manual_accum and use_token_budget and (budget_tokens > 0) and will_step:
            remain = max(0, int(budget_tokens) - int(accum_tokens))
            if remain > 0 and int(tokens_this) > remain:
                tokens_eff_this = int(remain)

        with ExitStack() as stack:
            if no_sync:
                # avoid all-reduce on non-final micro-batches
                if hasattr(student, "no_sync"):
                    stack.enter_context(student.no_sync())
                if hasattr(predictor, "no_sync"):
                    stack.enter_context(predictor.no_sync())

            # 2) student augmentations (time alignment preserving)
            x_aug = apply_student_augmentations(
                x,
                gain_min=train_cfg.aug_gain_min,
                gain_max=train_cfg.aug_gain_max,
                channel_gain_std=train_cfg.aug_channel_gain_std,
                noise_std_min=train_cfg.aug_noise_std_min,
                noise_std_max=train_cfg.aug_noise_std_max,
                channel_drop_prob=train_cfg.aug_channel_drop_prob,
            )

            # 3) freq corruption (student context only)
            freq_domain_drop = (torch.rand((B,), device=x.device) < train_cfg.freq_domain_drop_prob)
            freq_mask_bins = sample_freq_bin_mask(
                B=B,
                K=model_cfg.freq_bins,
                bin_centers_hz=bin_centers,
                physio_prob=train_cfg.freq_physio_mask_prob,
                num_bands_min=train_cfg.freq_num_bands_min,
                num_bands_max=train_cfg.freq_num_bands_max,
                random_width_min=train_cfg.freq_random_width_min,
                random_width_max=train_cfg.freq_random_width_max,
                device=x.device,
            )

            # 4) pack indices
            c_ctx, t_ctx, pad_ctx = mask_to_packed_indices(context_mask, valid)
            c_tgt, t_tgt, pad_tgt = mask_to_packed_indices(target_mask, valid)

            with accelerator.autocast():
                # 5) student: embed+encode context only
                tok_ctx, pad_ctx, rope_ctx, chan_ctx = accelerator.unwrap_model(student).embed_from_indices(
                    x=x_aug,
                    coords=coords.to(x.device),
                    c_idx=c_ctx,
                    t_idx=t_ctx,
                    pad=pad_ctx,
                    freq_mask_bins=freq_mask_bins,
                    freq_domain_drop=freq_domain_drop,
                )
                # NOTE: embed_from_indices called on unwrap_model(student) to avoid DDP wrapper issues with .unfold()
                # Then we pass tokens through the wrapped model for proper distributed behavior.
                z_ctx = student(tok_ctx, padding_mask=pad_ctx, rope_pos=rope_ctx, chan_idx=chan_ctx, coords=coords.to(x.device))  # (B,Lc,D)

                # 6) teacher: embed+encode targets only (no corruption)
                with torch.no_grad():
                    tok_tgt, pad_tgt2, rope_tgt, chan_tgt = accelerator.unwrap_model(teacher).embed_from_indices(
                        x=x,
                        coords=coords.to(x.device),
                        c_idx=c_tgt,
                        t_idx=t_tgt,
                        pad=pad_tgt,
                        freq_mask_bins=None,
                        freq_domain_drop=None,
                    )
                    z_tgt = teacher(tok_tgt, padding_mask=pad_tgt2, rope_pos=rope_tgt, chan_idx=chan_tgt, coords=coords.to(x.device))  # (B,Lt,D)

                # 7) predictor: build target queries from coord embeddings + cross-attend to ctx
                coord_ch = accelerator.unwrap_model(student).coord_embed(coords.to(x.device))  # (B,C,D)
                coord_tgt = gather_channel_embeddings(coord_ch, c_tgt.clamp(min=0), pad_tgt)   # (B,Lt,D)

                pred_tgt = predictor(
                    ctx=z_ctx,
                    ctx_pad=pad_ctx,
                    rope_ctx=rope_ctx,
                    tgt_coord_emb=coord_tgt,
                    tgt_pad=pad_tgt,
                    rope_tgt=rope_tgt,
                )  # (B,Lt,D)

                # ------------------------------------------------------------
                # (Optional) multi-view consistency loss (DINO-style multi-crop)
                # ------------------------------------------------------------
                mv_loss = None
                mv_pairs = 0
                if mv_enabled and isinstance(metas, list) and len(metas) == B:
                    # pool per-sample representations from target latents (cheap; already computed)
                    valid_tgt_tmp = ~pad_tgt  # (B,Lt)
                    denom = valid_tgt_tmp.sum(dim=1).clamp_min(1).to(pred_tgt.dtype)  # (B,)

                    # student pooled rep: mean(pred_tgt)
                    p_pool = (pred_tgt * valid_tgt_tmp[..., None]).sum(dim=1) / denom[:, None]  # (B,D)
                    p_pool = F.normalize(p_pool.float(), dim=-1)  # fp32

                    # teacher pooled rep: mean(z_tgt)
                    t_pool = (z_tgt * valid_tgt_tmp[..., None]).sum(dim=1) / denom[:, None]  # (B,D)
                    t_pool = F.normalize(t_pool.float(), dim=-1)  # fp32

                    losses = []
                    for i in range(B):
                        mi = metas[i] if isinstance(metas[i], dict) else {}
                        if mv_only_mc and (not bool(mi.get("multicrop", False))):
                            continue
                        uid = str(mi.get("parent_uid", ""))
                        if not uid:
                            # fallback: webdataset key
                            uid = str(mi.get("__key__", ""))
                        if not uid:
                            continue
                        vtype = str(mi.get("view_type", ""))

                        # DINO-like: local(student) matches bank(global/any)
                        if vtype == "local":
                            target = mv_bank_global.get(uid)
                            if target is None:
                                target = mv_bank_any.get(uid)
                            if target is not None:
                                # cosine distance
                                losses.append(1.0 - (p_pool[i] * target).sum())
                                mv_pairs += 1

                    if len(losses) > 0:
                        mv_loss = torch.stack(losses).mean().float()
                    else:
                        mv_loss = torch.zeros((), device=pred_tgt.device, dtype=torch.float32)

                    # update banks with teacher reps (stop-grad)
                    # - global views populate mv_bank_global
                    for i in range(B):
                        mi = metas[i] if isinstance(metas[i], dict) else {}
                        if mv_only_mc and (not bool(mi.get("multicrop", False))):
                            continue
                        uid = str(mi.get("parent_uid", "")) or str(mi.get("__key__", ""))
                        if not uid:
                            continue
                        vtype = str(mi.get("view_type", ""))
                        vec = t_pool[i].detach()
                        mv_bank_any[uid] = vec
                        if vtype == "global":
                            mv_bank_global[uid] = vec

                    # logging accumulators (per optimizer step)
                    if mv_pairs > 0:
                        mv_pairs_sum += float(mv_pairs)
                        mv_loss_pairs_sum += mv_loss.detach().float() * float(mv_pairs)

                # 8) loss on non-pad targets
                valid_tgt = ~pad_tgt
                num_tgt = int(valid_tgt.sum().item())
                if num_tgt <= 0:
                    # extremely unlikely, but skip safely
                    continue

                # sum reduction for token-budget scaling
                loss_sum = F.l1_loss(pred_tgt[valid_tgt], z_tgt[valid_tgt], reduction="sum").float()

                # scale loss so that one optimizer step ~= (tokens_per_update) worth of supervision
                weight = 1.0
                if manual_accum:
                    denom_tokens = max(1, budget_tokens)
                    if use_token_budget:
                        if accum_basis != "target":
                            # If you set accum_basis != target, you're effectively optimizing "loss per (basis) token".
                            # This is allowed, but note that supervision exists only on target tokens.
                            pass
                        # fractional weight for the last micro-batch to hit the token budget more tightly
                        if will_step and denom_tokens > 0:
                            remain = max(0, denom_tokens - int(accum_tokens))
                            if tokens_this > 0 and remain > 0 and tokens_this > remain:
                                weight = float(remain) / float(tokens_this)
                            elif remain == 0:
                                weight = 1.0
                        loss_scaled = (loss_sum * float(weight)) / (float(denom_tokens) * float(model_cfg.d_model))
                    else:
                        # fixed micro-batch accumulation: average loss across micro-batches
                        loss_scaled = (loss_sum / (float(num_tgt) * float(model_cfg.d_model))) / float(denom_tokens)
                else:
                    # single-step: standard mean loss
                    loss_scaled = loss_sum / (float(num_tgt) * float(model_cfg.d_model))

                loss_log = loss_sum / (float(num_tgt) * float(model_cfg.d_model))

                # Add multi-view loss (scaled to match token-budget accumulation)
                if mv_enabled and (mv_loss is not None) and (mv_weight > 0):
                    mv_scale = 1.0
                    if manual_accum:
                        if use_token_budget:
                            mv_scale = (float(tokens_this) * float(weight)) / float(max(1, budget_tokens))
                        else:
                            mv_scale = float(weight) / float(max(1, budget_tokens))
                    loss_scaled = loss_scaled + (mv_weight * mv_loss * float(mv_scale))

                # ------------------------------------------------------------
                # (Optional) VICReg-style variance/cov regularizer
                # ------------------------------------------------------------
                vic_w = float(getattr(train_cfg, "vicreg_weight", 0.0) or 0.0)
                vic_apply = str(getattr(train_cfg, "vicreg_apply_to", "pred")).lower().strip()
                vic_gamma = float(getattr(train_cfg, "vicreg_gamma", 1.0) or 1.0)
                vic_var_w = float(getattr(train_cfg, "vicreg_var_weight", 1.0) or 1.0)
                vic_cov_w = float(getattr(train_cfg, "vicreg_cov_weight", 1.0) or 1.0)
                vic_max = int(getattr(train_cfg, "vicreg_max_tokens", 0) or 0)

                if vic_w > 0:
                    if vic_apply == "ctx":
                        valid_ctx_tmp = ~pad_ctx
                        denom_c = valid_ctx_tmp.sum(dim=1).clamp_min(1).to(z_ctx.dtype)
                        feats = (z_ctx * valid_ctx_tmp[..., None]).sum(dim=1) / denom_c[:, None]  # (B,D)
                        feats = feats.float()
                    else:
                        feats = pred_tgt[valid_tgt].float()  # (Nt,D)

                    if vic_max > 0 and feats.shape[0] > vic_max:
                        idx = torch.randperm(feats.shape[0], device=feats.device)[:vic_max]
                        feats = feats.index_select(0, idx)

                    base, vvar, vcov = vicreg_var_cov_loss(feats, gamma=vic_gamma, var_weight=vic_var_w, cov_weight=vic_cov_w)
                    vic_loss = vic_w * base

                    # scale to be invariant to #micro-batches per optimizer update
                    reg_scale = 1.0
                    if manual_accum:
                        if use_token_budget:
                            reg_scale = float(tokens_eff_this) / float(max(1, budget_tokens))
                        else:
                            reg_scale = float(weight) / float(max(1, budget_tokens))

                    loss_scaled = loss_scaled + vic_loss * float(reg_scale)

                    if 'vic_log' in locals() and vic_log:
                        vic_loss_sum += vic_loss.detach().float() * float(reg_scale)
                        vic_var_sum += vvar.detach().float() * float(reg_scale)
                        vic_cov_sum += vcov.detach().float() * float(reg_scale)
                        vic_w_sum += float(reg_scale)

                # ---------------------------------
                # Cheap proxy stats (logged on optimizer step)
                # ---------------------------------
                if proxy_enabled:
                    with torch.no_grad():
                        pred_f = pred_tgt[valid_tgt].detach().float()
                        tgt_f = z_tgt[valid_tgt].detach().float()

                        # subsample tokens to keep this cheap
                        max_proxy = int(getattr(train_cfg, "proxy_max_tokens", 0) or 0)
                        if max_proxy > 0 and pred_f.shape[0] > max_proxy:
                            idx = torch.linspace(0, pred_f.shape[0] - 1, steps=max_proxy, device=pred_f.device).long()
                            pred_f = pred_f.index_select(0, idx)
                            tgt_f = tgt_f.index_select(0, idx)

                        w = float(weight)
                        n_proxy = float(pred_f.shape[0])
                        if n_proxy > 0:
                            cos = F.cosine_similarity(pred_f, tgt_f, dim=-1)
                            proxy_last_cos_mean = cos.mean()
                            proxy_cos_sum += cos.sum() * w
                            proxy_n += n_proxy * w

                            pn = pred_f.norm(dim=-1)
                            tn = tgt_f.norm(dim=-1)
                            proxy_pred_norm_sum += pn.sum() * w
                            proxy_pred_norm_sumsq += (pn ** 2).sum() * w
                            proxy_tgt_norm_sum += tn.sum() * w
                            proxy_tgt_norm_sumsq += (tn ** 2).sum() * w

                            proxy_last_pred_feat_std = pred_f.std(unbiased=False)
                            proxy_last_tgt_feat_std = tgt_f.std(unbiased=False)

            accelerator.backward(loss_scaled)

        # update accumulation stats
        if manual_accum:
            accum_tokens += int(tokens_this)
            accum_tokens_eff += int(tokens_eff_this)
            accum_micro += 1
            accum_bucket_keys.add(bucket_key)

        # ---------------------------------
        # Optimizer step boundary
        # ---------------------------------
        do_step = bool(will_step)

        if do_step:
            grad_norm = None
            if train_cfg.grad_clip and train_cfg.grad_clip > 0:
                grad_norm = accelerator.clip_grad_norm_(
                    list(student.parameters()) + list(predictor.parameters()),
                    train_cfg.grad_clip,
                )

            # LR schedule (baseline: cosine over steps; optional: token-based WCC)
            if lr_sched == "token_wcc":
                tpu = int(budget_tokens)
                total_toks = int(getattr(train_cfg, "lr_total_tokens", 0) or 0)
                warm_toks = int(getattr(train_cfg, "lr_warmup_tokens", 0) or 0)
                cool_toks = int(getattr(train_cfg, "lr_cooldown_tokens", 0) or 0)
                min_lr = float(getattr(train_cfg, "min_lr", 0.0) or 0.0)

                if total_toks <= 0:
                    total_toks = int(train_cfg.max_steps) * max(1, tpu)
                if warm_toks <= 0:
                    warm_toks = int(train_cfg.warmup_steps) * max(1, tpu)
                if cool_toks <= 0:
                    frac = float(getattr(train_cfg, "lr_cooldown_frac", 0.10) or 0.10)
                    cool_toks = int(frac * float(total_toks))

                tokens_next = int(tokens_seen_total) + int(accum_tokens_eff)
                lr = token_wcc_lr(tokens_next, warm_toks, total_toks, cool_toks, train_cfg.lr, min_lr)
            else:
                lr = cosine_warmup(global_step, train_cfg.warmup_steps, train_cfg.max_steps, train_cfg.lr)
            for pg in opt.param_groups:
                pg["lr"] = lr

            opt.step()
            opt.zero_grad(set_to_none=True)
            if lr_sched == "token_wcc":
                tokens_seen_total = int(tokens_seen_total) + int(accum_tokens_eff)

            m = ema_momentum_schedule(global_step, train_cfg.max_steps, train_cfg.ema_momentum, train_cfg.ema_momentum_final)
            update_ema(teacher=accelerator.unwrap_model(teacher), student=accelerator.unwrap_model(student), m=m)

            if accelerator.is_main_process and (global_step % train_cfg.log_every == 0):
                # bucket key (shape) for this micro-batch (ShapeBatcher guarantees same (C,P) inside micro-batch)
                log_bucket = bool(getattr(train_cfg, "log_bucket_key", True))
                if log_bucket:
                    bucket_C = int(n_channels[0].detach().cpu().item())
                    bucket_P = int(n_patches[0].detach().cpu().item())
                    bucket_tok_per_sample = int(bucket_C * bucket_P)
                else:
                    bucket_C = 0
                    bucket_P = 0
                    bucket_tok_per_sample = 0

                logs = {
                    "loss_tgt_mean": float(loss_log.detach().cpu().item()),
                    "loss_scaled": float(loss_scaled.detach().cpu().item()),
                    "lr": lr,
                    "tokens_seen_total": int(tokens_seen_total) if lr_sched == "token_wcc" else 0,
                    "ema_m": m,
                    "grad_norm": float(grad_norm.detach().cpu().item()) if grad_norm is not None else 0.0,
                    "vicreg_loss": float((vic_loss_sum / vic_w_sum.clamp_min(1e-9)).detach().cpu().item()) if vic_log else 0.0,
                    "vicreg_var": float((vic_var_sum / vic_w_sum.clamp_min(1e-9)).detach().cpu().item()) if vic_log else 0.0,
                    "vicreg_cov": float((vic_cov_sum / vic_w_sum.clamp_min(1e-9)).detach().cpu().item()) if vic_log else 0.0,
                    "ctx_tokens_max": int((~pad_ctx).sum(dim=1).max().detach().cpu().item()),
                    "tgt_tokens_max": int((~pad_tgt).sum(dim=1).max().detach().cpu().item()),
                    "ctx_tokens_sum": int((~pad_ctx).sum().detach().cpu().item()),
                    "tgt_tokens_sum": int((~pad_tgt).sum().detach().cpu().item()),
                    "batch_size_samples": int(B),
                    "bucket_C": bucket_C,
                    "bucket_P": bucket_P,
                    "bucket_tok_per_sample": bucket_tok_per_sample,
                    "tokens_per_batch_cfg": int(train_cfg.tokens_per_batch),
                    "steps_time": float((time.time() - start_time) / train_cfg.log_every),
                }

                # cheap proxy metrics (representation health checks)
                if proxy_enabled and (proxy_n is not None):
                    n = float(proxy_n.detach().cpu().item())
                    if n > 0:
                        cos_mean = float((proxy_cos_sum / proxy_n.clamp_min(1.0)).detach().cpu().item())
                        pred_norm_mean = float((proxy_pred_norm_sum / proxy_n.clamp_min(1.0)).detach().cpu().item())
                        tgt_norm_mean = float((proxy_tgt_norm_sum / proxy_n.clamp_min(1.0)).detach().cpu().item())
                        pred_norm_var = (proxy_pred_norm_sumsq / proxy_n.clamp_min(1.0)) - (proxy_pred_norm_sum / proxy_n.clamp_min(1.0)) ** 2
                        tgt_norm_var = (proxy_tgt_norm_sumsq / proxy_n.clamp_min(1.0)) - (proxy_tgt_norm_sum / proxy_n.clamp_min(1.0)) ** 2
                        pred_norm_std = float(torch.clamp(pred_norm_var, min=0.0).sqrt().detach().cpu().item())
                        tgt_norm_std = float(torch.clamp(tgt_norm_var, min=0.0).sqrt().detach().cpu().item())

                        logs.update({
                            "proxy/cos_mean": cos_mean,
                            "proxy/pred_norm_mean": pred_norm_mean,
                            "proxy/pred_norm_std": pred_norm_std,
                            "proxy/tgt_norm_mean": tgt_norm_mean,
                            "proxy/tgt_norm_std": tgt_norm_std,
                            "proxy/pred_feat_std_last": float(proxy_last_pred_feat_std.detach().cpu().item()),
                            "proxy/tgt_feat_std_last": float(proxy_last_tgt_feat_std.detach().cpu().item()),
                            "proxy/cos_mean_last": float(proxy_last_cos_mean.detach().cpu().item()),
                            "proxy/n_tokens": n,
                        })
                if manual_accum:
                    logs.update({
                        "accum_tokens": int(accum_tokens),
                        "accum_tokens_overshoot": float(max(0, int(accum_tokens) - int(budget_tokens)) / int(budget_tokens)),
                        "accum_micro": int(accum_micro),
                        "accum_unique_buckets": int(len(accum_bucket_keys)),
                        "accum_basis": accum_basis,
                        "tokens_per_update": int(budget_tokens),
                    })

                # multi-view stats (only meaningful when long_multicrop_mode == "multiview")
                if mv_enabled:
                    pairs = float(mv_pairs_sum.detach().cpu().item())
                    if pairs > 0:
                        mv_mean = float((mv_loss_pairs_sum / mv_pairs_sum.clamp_min(1.0)).detach().cpu().item())
                    else:
                        mv_mean = 0.0
                    logs.update({
                        "multiview/pairs": pairs,
                        "multiview/loss_mean": mv_mean,
                        "multiview/weight": float(mv_weight),
                    })
                if metrics_writer is not None:
                    metrics_writer.write(global_step, logs)
                accelerator.log(logs, step=global_step)
                start_time = time.time()

            if (train_cfg.save_every > 0) and ((global_step + 1) % train_cfg.save_every == 0):
            # if accelerator.is_main_process and (global_step % train_cfg.save_every == 0):
                accelerator.wait_for_everyone()
                ckpt_dir = os.path.join(train_cfg.output_dir, f"step_{global_step:07d}")
                os.makedirs(ckpt_dir, exist_ok=True)
                if accelerator.is_main_process:
                    accelerator.unwrap_model(student).save_pretrained(os.path.join(ckpt_dir, f"student_{global_step:07d}"))
                    torch.save(accelerator.unwrap_model(predictor).state_dict(), os.path.join(ckpt_dir, f"predictor_{global_step:07d}.pt"))
                    accelerator.unwrap_model(teacher).save_pretrained(os.path.join(ckpt_dir, f"teacher_{global_step:07d}"))
                accelerator.save_state(os.path.join(ckpt_dir, f"accelerator_state_{global_step:07d}"))
                accelerator.wait_for_everyone()

            # reset accumulation window
            accum_tokens = 0
            accum_micro = 0
            accum_bucket_keys = set()
            accum_tokens_eff = 0

            if proxy_enabled:
                proxy_n.zero_()
                proxy_cos_sum.zero_()
                proxy_pred_norm_sum.zero_()
                proxy_pred_norm_sumsq.zero_()
                proxy_tgt_norm_sum.zero_()
                proxy_tgt_norm_sumsq.zero_()
                proxy_last_pred_feat_std.zero_()
                proxy_last_tgt_feat_std.zero_()
                proxy_last_cos_mean.zero_()

            if vic_log:
                vic_loss_sum.zero_()
                vic_var_sum.zero_()
                vic_cov_sum.zero_()
                vic_w_sum.zero_()

            if mv_enabled:
                mv_pairs_sum.zero_()
                mv_loss_pairs_sum.zero_()
                mv_bank_any.clear()
                mv_bank_global.clear()

            global_step += 1
            pbar.update(1)
            pbar.set_postfix({"loss": f"{float(loss_log.detach().cpu()):.4f}", "lr": f"{lr:.2e}"})

    pbar.close()

    accelerator.wait_for_everyone()  # ensure all processes have finished saving before we write the final checkpoint
    final_dir = os.path.join(train_cfg.output_dir, "final")
    if accelerator.is_main_process:
        os.makedirs(final_dir, exist_ok=True)
        accelerator.unwrap_model(student).save_pretrained(os.path.join(final_dir, "student"))
        torch.save(accelerator.unwrap_model(predictor).state_dict(), os.path.join(final_dir, "predictor.pt"))
        accelerator.unwrap_model(teacher).save_pretrained(os.path.join(final_dir, "teacher"))
    accelerator.save_state(os.path.join(final_dir, "accelerator_state"))

    # Close local metrics writer (if any)
    if metrics_writer is not None and accelerator.is_main_process:
        metrics_writer.close()

    accelerator.end_training()
    return _build_train_artifacts(train_cfg)


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, object]:
    return run_train(parse_args(argv))


if __name__ == "__main__":
    main()
