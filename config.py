# eeg_fm/config.py
from __future__ import annotations

from dataclasses import dataclass, asdict, fields
import json
import os
from typing import Any, Dict, Optional, Tuple


@dataclass
class EEGModelConfig:
    # --------- Signal / tokenization ----------
    sample_rate: int = 200  # fixed in preprocessing
    patch_seconds: float = 1.0
    patch_hop_seconds: float = 1.0  # hop < patch_seconds => overlap
    max_tokens: int = 4096

    # --------- Model options ----------
    mlp_type: str = "swiglu"  # "gelu", "swiglu", or "geglu"
    norm_type: str = "rmsnorm"  # "layernorm" or "rmsnorm"
    layerscale_init: float = 0.0  # if >0, use LayerScale with this init value; else disable

    # --------- Model size ----------
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 12
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    attn_dropout: float = 0.0

    # --------- RoPE (time only) ----------
    rope_theta: float = 10000.0
    rotary_pct: float = 1.0

    # --------- Attention stabilization ----------
    # Optional QK normalization inside attention; independent of MLP/RMSNorm/LN choices.
    # - off       : standard scaled dot-product attention
    # - l2        : L2-normalize q/k vectors (cosine attention) with rescale
    # - rms       : RMS-normalize q/k vectors
    # - layernorm : LayerNorm on q/k vectors
    attn_qk_norm: str = "off"  # off | l2 | rms | layernorm
    attn_qk_norm_eps: float = 1e-6

    # --------- Encoder attention architecture ----------
    # "full"      : standard 1D packed-token self-attention
    # "divided"   : TimeSformer-like divided attention (temporal pass then spatial pass)
    # "hybrid"    : mostly divided, but every N blocks use a full-attention block
    encoder_arch: str = "hybrid"
    full_attn_every: int = 4  # for hybrid; 0 disables full-attn blocks
    full_attn_use_spatial_bias: bool = True  # may disable flash SDP depending on PyTorch

    # --------- Spatial attention bias ----------
    # Additive attention-logits bias computed from electrode coords.
    # - "legendre": learnable Legendre series in cos(gamma)=u·v (unit-sphere)
    # - "none"    : disable
    # NOTE: optional MLP bias was removed; deprecated fields are kept for backward compatibility.
    spatial_bias_type: str = "legendre"  # "legendre" | "none"
    spatial_bias_degree: int = 8
    spatial_bias_use_unit_sphere: bool = True
    spatial_bias_scale: float = 1.0
    # (deprecated)
    spatial_bias_mlp_hidden: int = 64
    spatial_bias_mlp_depth: int = 2

    # --------- Spatial embedding (token-level coord embedding; distinct from spatial bias) ----------
    coord_num_freqs: int = 6
    coord_max_freq: float = 2.0
    coord_include_raw: bool = True
    coord_jitter_std: float = 0.05           # applied with prob coord_jitter_prob
    coord_jitter_prob: float = 0.5
    coord_renormalize: bool = False           # renormalize coords after jitter
    coord_w_init: float = 0.0                      # initial value for learnable spatial bias weight
    isFourier: bool = False

    # --------- Frequency features ----------
    freq_min_hz: float = 1.0
    freq_max_hz: float = 45.0
    freq_bins: int = 32
    freq_spacing: str = "log"          # "log" or "linear"
    freq_use_scale: bool = True        # append mean(log-power) as scale dim
    freq_eps: float = 1e-6
    fft_dtype: str = "float32"         # float32 recommended for FFT stability

    # --------- FiLM fusion ----------
    film_hidden: int = 512

    # --------- JEPA predictor ----------
    predictor_hidden: int = 2048
    predictor_depth: int = 2  # for MLP predictor
    predictor_type: str = "cross_attn"  # "cross_attn" (I-JEPA) or "mlp"
    predictor_layers: int = 2            # cross-attn predictor depth
    predictor_n_heads: int = 8
    predictor_mlp_ratio: float = 4.0

    # --------- Mask token ----------
    mask_token_init_std: float = 0.02

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_json(path: str) -> "EEGModelConfig":
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        # Robust to extra keys (older/newer templates).
        allowed = {fd.name for fd in fields(EEGModelConfig)}
        d_f = {k: v for k, v in d.items() if k in allowed}
        return EEGModelConfig(**d_f)

    def save_json(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)


@dataclass
class TrainConfig:
    # --------- Repro / determinism ----------
    # NOTE:
    # - For throughput: cudnn_benchmark=True and deterministic=False is typical.
    # - For tuning/ablations: fix seeds.
    seed: int = 42
    torch_deterministic: bool = False
    cudnn_benchmark: bool = True

    # --------- Data ----------
    data_root: str = "/mnt/e/open_eeg"
    cache_dir: str = "/mnt/c/wds_cache"
    shard_glob: str = "**/*.tar"

    shard_shuffle: int = 500
    sample_shuffle: int = 256
    cache_max_bytes: int = int(200 * 1024**3)
    post_split_shuffle: int = 128
    eviction_interval: int = 8

    # Long-segment (60s) handling
    long_crop_prob: float = 0.2
    long_crop_30_prob: float = 0.5
    enable_channel_grouping: bool = True

    # bucket batching
    tokens_per_batch: int = 16384
    max_samples_per_batch: int = 256

    # auto-tune tokens_per_batch with a synthetic probe (optional)
    auto_tune_tokens_per_batch: bool = False
    auto_tune_target_mem_frac: float = 0.85
    auto_tune_probe_seq_len: int = 4096
    auto_tune_probe_batch: int = 2

    num_workers: int = 8

    # --------- JEPA masking ----------
    mask_time_prob: float = 0.8
    mask_spatial_prob: float = 0.8
    mask_dilate_time: int = 0
    is_SSP: bool = False 
    time_mask_ratio_min: float = 0.15
    time_mask_ratio_max: float = 0.35
    spatial_mask_ratio_min: float = 0.10
    spatial_mask_ratio_max: float = 0.30

    # --------- Student augmentations (time-alignment preserving) ----------
    aug_gain_min: float = 0.8
    aug_gain_max: float = 1.2
    aug_channel_gain_std: float = 0.0
    aug_noise_std_min: float = 0.00
    aug_noise_std_max: float = 0.03
    aug_channel_drop_prob: float = 0.05

    # --------- Freq corruption (student only) ----------
    freq_domain_drop_prob: float = 0.25
    freq_physio_mask_prob: float = 0.6
    freq_num_bands_min: int = 1
    freq_num_bands_max: int = 2
    freq_random_width_min: float = 0.10
    freq_random_width_max: float = 0.25

    # --------- Optimization ----------
    lr: float = 2e-4
    weight_decay: float = 0.05
    betas: Tuple[float, float] = (0.9, 0.95)
    grad_clip: float = 1.0
    grad_accum_steps: int = 1

    # token-budget accumulation (dynamic)
    accum_tokens_basis: str = "target"  # "target" | "context" | "valid" | "micro"(internal)
    tokens_per_update: int = 131_072

    max_steps: int = 200_000
    warmup_steps: int = 5_000

    # --------- LR schedule ----------
    # Baseline: cosine warmup over steps (existing behavior)
    # Optional: token-based WCC (warmup-constant-cooldown trapezoid) over *effective tokens*.
    lr_schedule: str = "token_wcc"  # cosine | token_wcc
    min_lr: float = 0.0

    # Token-based WCC params (only used when lr_schedule=="token_wcc")
    # If 0, they default to warmup_steps/max_steps scaled by tokens_per_update.
    lr_total_tokens: int = 0
    lr_warmup_tokens: int = 0
    lr_cooldown_tokens: int = 0
    lr_cooldown_frac: float = 0.10

    # --------- VICReg-style anti-collapse regularizer (optional) ----------
    # Adds variance/covariance penalties on student features (cheap, no extra views required).
    vicreg_weight: float = 0.0
    vicreg_apply_to: str = "pred"  # pred | ctx
    vicreg_gamma: float = 1.0
    vicreg_var_weight: float = 1.0
    vicreg_cov_weight: float = 1.0
    vicreg_max_tokens: int = 2048

    # EMA teacher
    ema_momentum: float = 0.996
    ema_momentum_final: float = 0.9999

    # --------- Runtime ----------
    output_dir: str = "./checkpoints/eeg_jepa"
    log_every: int = 50
    save_every: int = 5000

    mixed_precision: str = "bf16"
    use_wandb: bool = True
    wandb_project: str = "EEG_FM"
    run_name: Optional[str] = None

    # --------- Logging ----------
    log_bucket_key: bool = True
    log_proxies: bool = True
    proxy_max_tokens: int = 2048

    # --------- Resume / init ----------
    resume_from: Optional[str] = None
    init_from: Optional[str] = None

    # Debug / small runs
    limit_num_samples: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_json(path: str) -> "TrainConfig":
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        allowed = {fd.name for fd in fields(TrainConfig)}
        d_f = {k: v for k, v in d.items() if k in allowed}
        return TrainConfig(**d_f)

    def save_json(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
