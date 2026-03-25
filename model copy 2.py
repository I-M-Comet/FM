# eeg_fm/model.py
from __future__ import annotations

import math
import os
import json
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import EEGModelConfig

from contextlib import nullcontext
import warnings

try:
    from torch.nn.attention import sdpa_kernel, SDPBackend
except Exception:
    sdpa_kernel = None
    SDPBackend = None

try:
    from torch.backends.cuda import (
        SDPAParams,
        can_use_flash_attention,
        can_use_efficient_attention,
        can_use_cudnn_attention,
    )
except Exception:
    SDPAParams = None
    can_use_flash_attention = None
    can_use_efficient_attention = None
    can_use_cudnn_attention = None

def enable_sdpa_debug(model, force_backend="flash", debug_once=False):
    for i, blk in enumerate(model.blocks):
        if isinstance(blk, FullAttentionBlock):
            blk.attn.sdpa_label = f"blocks.{i}.full"
        elif isinstance(blk, DividedSpatiotemporalBlock):
            blk.attn_t.sdpa_label = f"blocks.{i}.divided.temporal"
            blk.attn_s.sdpa_label = f"blocks.{i}.divided.spatial"
    for m in model.modules():
        if isinstance(m, MultiheadSelfAttentionRoPE):
            m.sdpa_debug = True
            m.sdpa_debug_once = debug_once
            m.sdpa_force_backend = force_backend  # "", "flash", "efficient", "math"

def _gather_channel_features(x_ch: torch.Tensor, c_idx: torch.Tensor) -> torch.Tensor:
    """
    x_ch: (B,C,F)
    c_idx: (B,L)
    returns: (B,L,F)
    """
    B, C, F = x_ch.shape
    B2, L = c_idx.shape
    assert B == B2
    idx = c_idx[..., None].expand(B, L, F)
    return x_ch.gather(dim=1, index=idx)


class LegendreAnchorFeatures(nn.Module):
    """
    Practical low-rank spherical feature map:
      phi_r(u) = sum_{l=0}^L w_{r,l} P_l(u · a_r)
    where a_r are learnable anchor directions on the unit sphere.

    Output shape:
      (B, C, F) where F = spatial_qk_feat_dim
    """
    def __init__(self, cfg: EEGModelConfig):
        super().__init__()
        self.num_anchors = int(getattr(cfg, "spatial_qk_num_anchors", 32))
        self.degree = int(getattr(cfg, "spatial_qk_degree", getattr(cfg, "spatial_bias_degree", 8)))
        self.out_dim = int(getattr(cfg, "spatial_qk_feat_dim", 64))
        self.use_unit = bool(getattr(cfg, "spatial_bias_use_unit_sphere", True))
        self.eps = float(getattr(cfg, "spatial_bias_eps", 1e-6))

        # learnable anchor directions on sphere
        anchors = torch.randn(self.num_anchors, 3)
        anchors = F.normalize(anchors, p=2, dim=-1)
        self.anchors = nn.Parameter(anchors)

        # per-anchor Legendre coefficients
        self.coeff = nn.Parameter(torch.zeros(self.num_anchors, self.degree + 1))

        self.proj = nn.Linear(self.num_anchors, self.out_dim, bias=False)
        self.norm = nn.LayerNorm(self.out_dim)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        coords: (B, C, 3)
        returns: (B, C, F)
        """
        if self.use_unit:
            u = coords / coords.norm(dim=-1, keepdim=True).clamp_min(self.eps)
        else:
            u = coords

        a = F.normalize(self.anchors, p=2, dim=-1)  # (R, 3)
        x = torch.einsum("bcd,rd->bcr", u.float(), a.float()).clamp(-1.0, 1.0)  # (B,C,R)

        # Legendre basis along each anchor
        basis = []
        Pm2 = torch.ones_like(x)
        basis.append(Pm2)

        if self.degree >= 1:
            Pm1 = x
            basis.append(Pm1)

            for l in range(2, self.degree + 1):
                Pl = ((2 * l - 1) * x * Pm1 - (l - 1) * Pm2) / float(l)
                basis.append(Pl)
                Pm2, Pm1 = Pm1, Pl

        basis = torch.stack(basis, dim=-1)  # (B,C,R,L+1)
        coeff = self.coeff[None, None, :, :].to(dtype=basis.dtype, device=basis.device)
        feat = (basis * coeff).sum(dim=-1)  # (B,C,R)

        feat = self.proj(feat)
        feat = self.norm(feat)
        return feat.to(coords.dtype)

class LayerSpecAlignHeads(nn.Module):
    """
    layer-wise hidden -> spectrum-view alignment head
    target view = fixed raw log-spectrum bins
    """
    def __init__(self, n_layers: int, d_model: int, spec_dim: int):
        super().__init__()
        self.heads = nn.ModuleList([
            nn.Sequential(
                make_norm("rms", d_model, eps=1e-6),
                nn.Linear(d_model, spec_dim),
            )
            for _ in range(n_layers)
        ])
        for h in self.heads:
            nn.init.zeros_(h[-1].weight)
            nn.init.zeros_(h[-1].bias)

    def forward_one(self, x: torch.Tensor, layer_idx: int) -> torch.Tensor:
        return self.heads[layer_idx](x)

class Z_Projector(nn.Module):
    def __init__(self, d_model: int, spec_dim: int):
        super().__init__()
        self.heads = nn.Sequential(
            make_norm("rms", d_model, eps=1e-6),
            nn.Linear(d_model, spec_dim),
        )

        nn.init.zeros_(self.heads[-1].weight)
        nn.init.zeros_(self.heads[-1].bias)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.heads(x)


# ============================================================
# RoPE utilities
# ============================================================
def build_rope_cache(
    max_pos: int,
    rotary_dim: int,
    theta: float,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns cos, sin: (max_pos, rotary_dim/2)
    """
    assert rotary_dim % 2 == 0
    half = rotary_dim // 2
    inv_freq = 1.0 / (theta ** (torch.arange(0, half, device=device, dtype=torch.float32) / half))
    t = torch.arange(max_pos, device=device, dtype=torch.float32)
    freqs = torch.einsum("i,j->ij", t, inv_freq)  # (max_pos, half)
    cos = torch.cos(freqs).to(dtype)
    sin = torch.sin(freqs).to(dtype)
    return cos, sin


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    x: (B, H, N, rotary_dim)
    cos/sin:
      - (N, half) OR
      - (B, N, half)
    """
    B, H, N, D = x.shape
    assert D % 2 == 0
    half = D // 2

    x1 = x[..., 0::2]
    x2 = x[..., 1::2]

    if cos.dim() == 2:
        # (N,half) -> (1,1,N,half)
        cos = cos[None, None, :, :]
        sin = sin[None, None, :, :]
    elif cos.dim() == 3:
        # (B,N,half) -> (B,1,N,half)
        cos = cos[:, None, :, :]
        sin = sin[:, None, :, :]
    else:
        raise ValueError(f"cos dim must be 2 or 3, got {cos.dim()}")

    y1 = x1 * cos - x2 * sin
    y2 = x1 * sin + x2 * cos
    y = torch.stack([y1, y2], dim=-1).flatten(-2)
    return y


# ============================================================
# Spatial embedding: coords -> Fourier features
# ============================================================
class CoordFourierEmbedding(nn.Module):
    def __init__(self, 
                 d_model: int, 
                 num_freqs: int, 
                 max_freq: float, 
                 include_raw: bool = True,
                 coord_jitter_std: float = 0.05,
                 coord_jitter_prob: float = 0.5,
                 renormalize: bool = False):
        super().__init__()
        self.num_freqs = int(num_freqs)
        self.include_raw = bool(include_raw)
        self.coord_jitter_std = float(coord_jitter_std)
        self.coord_jitter_prob = float(coord_jitter_prob)
        self.renormalize = bool(renormalize)

        freqs = 2.0 ** torch.arange(self.num_freqs, dtype=torch.float32)
        freqs = freqs / freqs.max() * float(max_freq)
        self.register_buffer("freqs", freqs, persistent=False)

        in_dim = 0
        if self.include_raw:
            in_dim += 3
        in_dim += 3 * 2 * self.num_freqs
        self.proj = nn.Linear(in_dim, d_model)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        coords: (B, C, 3)
        return: (B, C, d_model)
        """
        B, C, _ = coords.shape
        if self.training and (self.coord_jitter_std > 0) and (self.coord_jitter_prob > 0):
            gate = (torch.rand((B,), device=coords.device) < self.coord_jitter_prob).to(coords.dtype)
            coords = coords + torch.randn_like(coords) * self.coord_jitter_std * gate[:, None, None]
            if self.renormalize:
                coords = F.normalize(coords, p=2, dim=-1)
        ang = coords[..., None] * (self.freqs[None, None, None, :] * math.pi)  # (B,C,3,F)
        s = torch.sin(ang)
        c = torch.cos(ang)
        sc = torch.cat([s, c], dim=-1).reshape(B, C, -1)

        if self.include_raw:
            feat = torch.cat([coords, sc], dim=-1)
        else:
            feat = sc
        return self.proj(feat)


class CoordMLPEmbedding(nn.Module):
    def __init__(self, 
                 d_model: int, 
                 coord_jitter_std: float = 0.05,
                 coord_jitter_prob: float = 0.5,
                 w_init: float = 0.0
                 ):
        super().__init__()
        self.coord_jitter_std = float(coord_jitter_std)
        self.coord_jitter_prob = float(coord_jitter_prob)

        in_dim = 3
        self.proj = nn.Linear(in_dim, d_model)
        self.emb_w = nn.Parameter(torch.tensor([w_init], dtype=torch.float32))  

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        coords: (B, C, 3)
        return: (B, C, d_model)
        """
        B, C, _ = coords.shape
        if self.training and (self.coord_jitter_std > 0) and (self.coord_jitter_prob > 0):
            gate = (torch.rand((B,), device=coords.device) < self.coord_jitter_prob).to(coords.dtype)
            coords = coords + torch.randn_like(coords) * self.coord_jitter_std * gate[:, None, None]

        coords = F.normalize(coords, p=2, dim=-1) # unit vector
        return self.proj(coords) * self.emb_w


# ============================================================
# Frequency features: packed rFFT + filterbank
# ============================================================
def make_triangular_filterbank(
    freqs_hz: torch.Tensor,   # (F,)
    n_bins: int,
    f_min: float,
    f_max: float,
    spacing: str = "log",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    returns:
      fb: (K, F) non-negative, each row sums to 1
      centers: (K,) center freqs (Hz)
    """
    device = freqs_hz.device
    f_min = float(f_min)
    f_max = float(f_max)
    assert f_max > f_min

    if spacing == "log":
        edges = torch.logspace(
            math.log10(f_min),
            math.log10(f_max),
            steps=n_bins + 2,
            device=device,
            dtype=torch.float32,
        )
    else:
        edges = torch.linspace(f_min, f_max, steps=n_bins + 2, device=device, dtype=torch.float32)

    fb = torch.zeros((n_bins, freqs_hz.numel()), device=device, dtype=torch.float32)
    centers = edges[1:-1].clone()

    for k in range(n_bins):
        left, center, right = edges[k], edges[k + 1], edges[k + 2]
        up = (freqs_hz - left) / (center - left + 1e-12)
        down = (right - freqs_hz) / (right - center + 1e-12)
        w = torch.clamp(torch.minimum(up, down), min=0.0)
        fb[k] = w

    fb = fb / (fb.sum(dim=-1, keepdim=True) + 1e-12)
    return fb, centers


class RFFTFreqFeatures(nn.Module):
    """
    rFFT on each patch (packed or dense), followed by triangular filterbank pooling + log-power + LN.
    Supports overlap by design (overlap is handled in patch extraction, not here).
    """
    def __init__(self, cfg: EEGModelConfig):
        super().__init__()
        self.cfg = cfg
        self.patch_samples = int(round(cfg.sample_rate * cfg.patch_seconds))
        self.n_fft = self.patch_samples

        window = torch.hann_window(self.patch_samples, periodic=True, dtype=torch.float32)
        self.register_buffer("window", window, persistent=False)

        freqs = torch.fft.rfftfreq(self.n_fft, d=1.0 / cfg.sample_rate).to(torch.float32)
        sel = (freqs >= cfg.freq_min_hz) & (freqs <= cfg.freq_max_hz)
        self.register_buffer("sel_idx", torch.nonzero(sel, as_tuple=False).squeeze(-1), persistent=False)
        freqs_sel = freqs[sel]
        self.register_buffer("freqs_sel", freqs_sel, persistent=False)

        fb, centers = make_triangular_filterbank(
            freqs_hz=freqs_sel,
            n_bins=cfg.freq_bins,
            f_min=cfg.freq_min_hz,
            f_max=cfg.freq_max_hz,
            spacing=cfg.freq_spacing,
        )
        self.register_buffer("fb", fb, persistent=False)
        self.register_buffer("bin_centers_hz", centers, persistent=False)

        self.ln = make_norm(cfg.norm_type, cfg.freq_bins, eps=1e-6)
        self.freq_dim = cfg.freq_bins + (1 if cfg.freq_use_scale else 0)

    def forward_packed(self, patches: torch.Tensor) -> torch.Tensor:
        """
        patches: (B, L, patch_samples)
        returns: f (B, L, freq_dim)
        """
        cfg = self.cfg
        B, L, S = patches.shape
        assert S == self.patch_samples

        # FFT dtype
        if cfg.fft_dtype == "float32":
            x = patches.to(torch.float32)
        elif cfg.fft_dtype == "bfloat16":
            x = patches.to(torch.bfloat16)
        else:
            x = patches

        x = x * self.window[None, None, :]  # Hann

        X = torch.fft.rfft(x, dim=-1)  # (B,L,F_full) complex
        P = (X.real ** 2 + X.imag ** 2)  # (B,L,F_full)
        P = P.index_select(dim=-1, index=self.sel_idx)  # (B,L,F_sel)

        fb = self.fb.to(P.dtype)  # (K,F_sel)
        feats = torch.einsum("blf,kf->blk", P, fb)  # (B,L,K)

        logp = torch.log(feats + cfg.freq_eps)
        if cfg.freq_use_scale:
            scale = logp.mean(dim=-1, keepdim=True)
            shape = self.ln(logp)
            out = torch.cat([shape, scale], dim=-1)
        else:
            out = self.ln(logp)
        return out


# ============================================================
# Time patch embedding (packed)
# ============================================================
class TimePatchEmbed(nn.Module):
    def __init__(self, cfg: EEGModelConfig):
        super().__init__()
        self.patch_samples = int(round(cfg.sample_rate * cfg.patch_seconds))
        self.proj = nn.Linear(self.patch_samples, cfg.d_model)

    def forward_packed(self, patches: torch.Tensor) -> torch.Tensor:
        """
        patches: (B, L, patch_samples)
        returns: (B, L, d_model)
        """
        return self.proj(patches)


# ============================================================
# FiLM fusion
# ============================================================
class FiLMFusion(nn.Module):
    def __init__(self, freq_dim: int, d_model: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(freq_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2 * d_model),
        )
        # freq input 0 -> identity
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
        """
        x: (..., D)
        f: (..., F) same leading dims
        """
        gb = self.net(f)
        gamma, beta = gb.chunk(2, dim=-1)
        return x * (1.0 + gamma) + beta


# ============================================================
# Spatial relative bias (channel-channel)
# ============================================================
class SpatialBias(nn.Module):
    """Compute a (B,C,C) additive attention bias from electrode coordinates.

    Supported types:
      - legendre: truncated Legendre series in cos(gamma) (default)
      - none: returns None

    NOTE:
      - The previous optional MLP spatial bias has been removed to keep the model
        simpler and faster. If you want a learnable non-parametric mapping, you
        can re-introduce it later, but Legendre is the default here.

    Notes:
      - By default we normalize coords to unit vectors so the bias depends only on direction.
      - Bias is shared across heads (broadcasted) to keep memory reasonable for full attention.
    """

    def __init__(self, cfg: EEGModelConfig):
        super().__init__()
        self.bias_type = str(getattr(cfg, "spatial_bias_type", "legendre")).lower()
        self.use_unit = bool(getattr(cfg, "spatial_bias_use_unit_sphere", True))
        self.eps = float(getattr(cfg, "spatial_bias_eps", 1e-6))
        self.scale = float(getattr(cfg, "spatial_bias_scale", 1.0))

        init_std = float(getattr(cfg, "spatial_bias_init_std", 0.0))

        if self.bias_type in ("none", "off", "disable", "disabled"):
            self.enabled = False
            self.degree = 0
            self.coeff = None
            return

        if self.bias_type != "legendre":
            raise ValueError(
                f"SpatialBias only supports 'legendre' or 'none'. Got spatial_bias_type={self.bias_type!r}. "
                "(MLP spatial bias was removed for efficiency.)"
            )

        self.enabled = True
        self.degree = int(getattr(cfg, "spatial_bias_degree", 8))
        # coeffs a_l (l=0..L). Init ~0 to start near unbiased.
        w = torch.zeros(self.degree + 1)
        if init_std > 0:
            w = w.normal_(mean=0.0, std=init_std)
        self.coeff = nn.Parameter(w)

    def _cosine(self, coords: torch.Tensor) -> torch.Tensor:
        # coords: (B,C,3)
        if self.use_unit:
            # direction-only
            u = coords / (coords.norm(dim=-1, keepdim=True).clamp_min(self.eps))
        else:
            u = coords
        # cos(gamma) between channels
        x = torch.matmul(u, u.transpose(1, 2))  # (B,C,C)
        return x.clamp(-1.0, 1.0)

    def forward(self, coords: torch.Tensor) -> Optional[torch.Tensor]:
        """Return bias_cc: (B,C,C) or None."""
        if not getattr(self, "enabled", True):
            return None

        # compute in fp32 for stability, then cast later
        x = self._cosine(coords.float())  # (B,C,C)
        # legendre series
        assert self.coeff is not None
        L = int(self.degree)
        # P0
        Pm2 = torch.ones_like(x)
        bias = self.coeff[0].to(x.dtype) * Pm2
        if L == 0:
            return bias

        # P1
        Pm1 = x
        bias = bias + self.coeff[1].to(x.dtype) * Pm1

        for l in range(2, L + 1):
            Pl = ((2 * l - 1) * x * Pm1 - (l - 1) * Pm2) / float(l)
            bias = bias + self.coeff[l].to(x.dtype) * Pl
            Pm2, Pm1 = Pm1, Pl

        if self.scale != 1.0:
            bias = bias * self.scale
        return bias


# ============================================================
# Attention blocks (PreNorm) with RoPE + Flash SDP
# ============================================================
class MultiheadSelfAttentionRoPE(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        attn_dropout: float,
        rope_theta: float,
        rotary_pct: float,
        qk_norm: str = "off",
        qk_norm_eps: float = 1e-6,
        spatial_qk: str = "none",
        spatial_qk_dim: int = 0, 
        spatial_qk_scale: float = 1.0,
    ):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.attn_dropout = float(attn_dropout)

        self.qk_norm = str(qk_norm).lower()
        self.qk_norm_eps = float(qk_norm_eps)
        if self.qk_norm != "off":
            if self.qk_norm != "l2":
                self.q_norm_weight = nn.Parameter(torch.ones(self.head_dim))
                self.k_norm_weight = nn.Parameter(torch.ones(self.head_dim))
            
            if self.qk_norm in ("layernorm", "ln"):
                self.q_norm_bias = nn.Parameter(torch.zeros(self.head_dim))
                self.k_norm_bias = nn.Parameter(torch.zeros(self.head_dim))

        rotary_dim = int(self.head_dim * rotary_pct)
        rotary_dim = rotary_dim - (rotary_dim % 2)
        self.rotary_dim = max(0, rotary_dim)
        self.rope_theta = float(rope_theta)

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=True)
        self.out = nn.Linear(d_model, d_model, bias=True)

        self._rope_cache = None  # (cos, sin, max_pos, dtype, device)

        self.spatial_qk_dim = int(spatial_qk_dim)
        self.spatial_qk_scale = float(spatial_qk_scale)
        self.spatial_qk = str(spatial_qk).lower()
        if self.spatial_qk != "none":
            self.spatial_q_proj = nn.Linear(self.spatial_qk_dim, d_model, bias=False)
            self.spatial_k_proj = nn.Linear(self.spatial_qk_dim, d_model, bias=False)
        else:
            self.spatial_q_proj = None
            self.spatial_k_proj = None        

        self.qk_l2_scale = self.head_dim ** 0.5

        # runtime debug switches
        self.sdpa_debug = False
        self.sdpa_debug_once = True
        self.sdpa_force_backend = ""   # "", "flash", "efficient", "math", "cudnn"
        self.sdpa_label = ""
        self._sdpa_debug_seen = False

    @staticmethod
    def _drop_trivial_padding_mask(padding_mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if padding_mask is None:
            return None
        if padding_mask.dtype != torch.bool:
            padding_mask = padding_mask.to(dtype=torch.bool)
        if padding_mask.numel() == 0:
            return None
        # all-False mask는 굳이 SDPA에 넘기지 않는다.
        if not bool(padding_mask.any().item()):
            return None
        return padding_mask

    def _build_sdpa_attn_mask(
        self,
        q: torch.Tensor,
        padding_mask: Optional[torch.Tensor],
        attn_bias: Optional[torch.Tensor],
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        padding_mask = self._drop_trivial_padding_mask(padding_mask)

        if attn_bias is None:
            if padding_mask is None:
                return None, None
            # bool keep-mask
            return (~padding_mask)[:, None, None, :], padding_mask

        if attn_bias.dim() == 3:
            attn_mask = attn_bias[:, None, :, :]
        elif attn_bias.dim() == 4:
            attn_mask = attn_bias
        else:
            raise ValueError(f"attn_bias must have dim 3 or 4, got {attn_bias.shape}")

        if attn_mask.dtype != q.dtype:
            attn_mask = attn_mask.to(dtype=q.dtype)

        if padding_mask is not None:
            attn_mask = attn_mask.masked_fill(padding_mask[:, None, None, :], float("-inf"))

        return attn_mask, padding_mask

    def _debug_backend_once(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        dropout_p: float,
    ) -> None:
        if not self.sdpa_debug:
            return
        if self.sdpa_debug_once and self._sdpa_debug_seen:
            return
        q = q.to(torch.bfloat16)
        k = k.to(torch.bfloat16)
        label = self.sdpa_label or self.__class__.__name__
        print(f"\n[SDPA DEBUG] {label}")
        print(f"  q={tuple(q.shape)} k={tuple(k.shape)} v={tuple(v.shape)}")
        print(f"  dtype={q.dtype} device={q.device} head_dim={self.head_dim} dropout_p={float(dropout_p)}")

        if attn_mask is None:
            print("  attn_mask=None")
        else:
            msg = f"shape={tuple(attn_mask.shape)} dtype={attn_mask.dtype}"
            if attn_mask.is_floating_point():
                finite_ratio = float(torch.isfinite(attn_mask).float().mean().item())
                msg += f" finite_ratio={finite_ratio:.6f}"
            print(f"  attn_mask={msg}")

        if (not q.is_cuda) or (SDPAParams is None):
            print("  CUDA SDPA debug helper unavailable on this device/build.")
            self._sdpa_debug_seen = True
            return

        params = SDPAParams(q, k, v, attn_mask, dropout_p, False, False)

        def _run_check(name: str, fn):
            if fn is None:
                print(f"  {name}: unavailable")
                return
            with warnings.catch_warnings(record=True) as ws:
                warnings.simplefilter("always")
                ok = fn(params, debug=True)
            print(f"  {name}: {ok}")
            for w in ws:
                print(f"    - {w.message}")

        _run_check("flash", can_use_flash_attention)
        _run_check("efficient", can_use_efficient_attention)
        _run_check("cudnn", can_use_cudnn_attention)

        self._sdpa_debug_seen = True

    def _sdpa_context(self):
        if (not self.sdpa_force_backend) or (sdpa_kernel is None) or (SDPBackend is None):
            return nullcontext()

        name = self.sdpa_force_backend.lower().strip()
        mapping = {
            "flash": [SDPBackend.FLASH_ATTENTION],
            "efficient": [SDPBackend.EFFICIENT_ATTENTION],
            "math": [SDPBackend.MATH],
        }

        cudnn_backend = getattr(SDPBackend, "CUDNN_ATTENTION", None)
        if cudnn_backend is not None:
            mapping["cudnn"] = [cudnn_backend]

        if name not in mapping:
            raise ValueError(
                f"Unknown sdpa_force_backend={self.sdpa_force_backend!r}. "
                f"Use one of: {list(mapping.keys())}"
            )
        return sdpa_kernel(mapping[name])
    
    def _get_rope(
        self,
        rope_pos: torch.Tensor,   # (N,) or (B,N)
        dtype: torch.dtype,
        device: torch.device,
        max_pos_hint: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        returns cos,sin indexed by positions:
          - if rope_pos is (N,), returns (N,half)
          - if rope_pos is (B,N), returns (B,N,half)
        """
        if self.rotary_dim == 0:
            raise RuntimeError("rotary_dim is 0 but _get_rope called")

        max_pos = int(max_pos_hint) if max_pos_hint is not None else (int(rope_pos.max().item()) + 1)

        need_rebuild = True
        if self._rope_cache is not None:
            cos, sin, cached_max, cached_dtype, cached_device = self._rope_cache
            if cached_max >= max_pos and cached_dtype == dtype and cached_device == device:
                need_rebuild = False
        if need_rebuild:
            cos, sin = build_rope_cache(
                max_pos=max_pos,
                rotary_dim=self.rotary_dim,
                theta=self.rope_theta,
                device=device,
                dtype=dtype,
            )
            self._rope_cache = (cos, sin, max_pos, dtype, device)

        cos, sin, _, _, _ = self._rope_cache
        if rope_pos.dim() == 1:
            return cos.index_select(0, rope_pos), sin.index_select(0, rope_pos)
        elif rope_pos.dim() == 2:
            # rope_pos (B,N) -> gather from (max_pos,half)
            # Use take_along_dim for speed
            half = cos.shape[1]
            cos_g = cos.index_select(0, rope_pos.reshape(-1)).reshape(rope_pos.shape[0], rope_pos.shape[1], half)
            sin_g = sin.index_select(0, rope_pos.reshape(-1)).reshape(rope_pos.shape[0], rope_pos.shape[1], half)
            return cos_g, sin_g
        else:
            raise ValueError(f"rope_pos must be (N,) or (B,N), got {rope_pos.shape}")

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor],
        rope_pos: Optional[torch.Tensor] = None,
        attn_bias: Optional[torch.Tensor] = None,          # SDPA용 dense bias
        spatial_q_add: Optional[torch.Tensor] = None,
        spatial_k_add: Optional[torch.Tensor] = None,
        rope_seq_len: Optional[int] = None,
    ) -> torch.Tensor:
        """
        x: (B, N, D)
        padding_mask: (B, N) bool, True for PAD
        rope_pos: (N,) or (B,N) long, time positions. Required if rotary_dim > 0.
        attn_bias: optional additive bias.
          - (B, N, N) float (broadcasts over heads)
          - or (B, 1, N, N) float
        """
        B, N, D = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, N, self.n_heads, self.head_dim).transpose(1, 2)  # (B,H,N,hd)
        k = k.view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, N, self.n_heads, self.head_dim).transpose(1, 2)

        if self.rotary_dim > 0:
            if rope_pos is None:
                raise ValueError("rope_pos must be provided when rotary_dim > 0")
            cos, sin = self._get_rope(rope_pos, dtype=q.dtype, device=q.device, max_pos_hint=rope_seq_len)  # (N,half) or (B,N,half)
            q_rot, q_pass = q[..., : self.rotary_dim], q[..., self.rotary_dim :]
            k_rot, k_pass = k[..., : self.rotary_dim], k[..., self.rotary_dim :]
            q_rot = apply_rope(q_rot, cos, sin)
            k_rot = apply_rope(k_rot, cos, sin)
            q = torch.cat([q_rot, q_pass], dim=-1)
            k = torch.cat([k_rot, k_pass], dim=-1)

        if (spatial_q_add is not None) or (spatial_k_add is not None):
            if (spatial_q_add is None) or (spatial_k_add is None):
                raise ValueError("spatial_q_add and spatial_k_add must be provided together")
            if spatial_q_add.shape != x.shape or spatial_k_add.shape != x.shape:
                raise ValueError(
                    f"preprojected spatial q/k must have shape {x.shape}, got {q_sp.shape} and {k_sp.shape}"
                )
            q_sp = spatial_q_add.view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
            k_sp = spatial_k_add.view(B, N, self.n_heads, self.head_dim).transpose(1, 2)

            scale = self.spatial_qk_scale
            q = q + scale * q_sp
            k = k + scale * k_sp

        attn_dtype = v.dtype   # 보통 bf16/fp16 유지
        # Optional QK normalization (attention stability)
        if self.qk_norm != "off":
            eps = self.qk_norm_eps
            if self.qk_norm == "l2":
                q_norm = torch.linalg.vector_norm(
                    q, dim=-1, keepdim=True, dtype=torch.float32
                ).clamp_min(eps)
                k_norm = torch.linalg.vector_norm(
                    k, dim=-1, keepdim=True, dtype=torch.float32
                ).clamp_min(eps)
                scale = self.head_dim ** 0.5

                q = (q.float() * (scale / q_norm)).to(attn_dtype)
                k = (k.float() * (scale / k_norm)).to(attn_dtype)
            elif self.qk_norm in ("rms", "rmsnorm"):
                qw = self.q_norm_weight.to(dtype=attn_dtype, device=q.device)
                kw = self.k_norm_weight.to(dtype=attn_dtype, device=k.device)
                q = F.rms_norm(q.to(attn_dtype), (self.head_dim,), weight=qw, eps=eps)
                k = F.rms_norm(k.to(attn_dtype), (self.head_dim,), weight=kw, eps=eps)

            elif self.qk_norm in ("layernorm", "ln"):
                qw = self.q_norm_weight.to(dtype=attn_dtype, device=q.device)
                kw = self.k_norm_weight.to(dtype=attn_dtype, device=k.device)
                qb = self.q_norm_bias.to(dtype=attn_dtype, device=q.device)
                kb = self.k_norm_bias.to(dtype=attn_dtype, device=k.device)

                q = F.layer_norm(q.to(attn_dtype), (self.head_dim,), weight=qw, bias=qb, eps=eps)
                k = F.layer_norm(k.to(attn_dtype), (self.head_dim,), weight=kw, bias=kb, eps=eps)
            else:
                raise ValueError(f"Unknown attn_qk_norm: {self.qk_norm}")
        if q.dtype != v.dtype or k.dtype != v.dtype:
            q = q.to(v.dtype)
            k = k.to(v.dtype)

        attn_mask, padding_mask = self._build_sdpa_attn_mask(
            q=q,
            padding_mask=padding_mask,
            attn_bias=attn_bias,
        )
        dropout_p = self.attn_dropout if self.training else 0.0
        self._debug_backend_once(q, k, v, attn_mask, dropout_p)

        try:
            with self._sdpa_context():
                out = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=attn_mask,
                    dropout_p=dropout_p,
                    is_causal=False,
                )
        except Exception as e:
            label = self.sdpa_label or self.__class__.__name__
            print(f"[SDPA DEBUG] forced backend={self.sdpa_force_backend!r} failed at {label}: {e}")
            raise

        out = out.transpose(1, 2).contiguous().view(B, N, D)
        return self.out(out)


class CrossAttentionRoPE(nn.Module):
    """
    Cross-attention: queries attend to context keys/values.
    RoPE is applied independently to Q and K using their time positions.
    """
    def __init__(self, d_model: int, n_heads: int, attn_dropout: float, rope_theta: float, rotary_pct: float, qk_norm: str = "off", qk_norm_eps: float = 1e-6):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.attn_dropout = float(attn_dropout)

        self.qk_norm = str(qk_norm).lower()
        self.qk_norm_eps = float(qk_norm_eps)
        if self.qk_norm != "off":
            if self.qk_norm != "l2":
                self.q_norm_weight = nn.Parameter(torch.ones(self.head_dim))
                self.k_norm_weight = nn.Parameter(torch.ones(self.head_dim))
            
            if self.qk_norm in ("layernorm", "ln"):
                self.q_norm_bias = nn.Parameter(torch.zeros(self.head_dim))
                self.k_norm_bias = nn.Parameter(torch.zeros(self.head_dim))

        rotary_dim = int(self.head_dim * rotary_pct)
        rotary_dim = rotary_dim - (rotary_dim % 2)
        self.rotary_dim = max(0, rotary_dim)
        self.rope_theta = float(rope_theta)

        self.q = nn.Linear(d_model, d_model, bias=True)
        self.kv = nn.Linear(d_model, 2 * d_model, bias=True)
        self.out = nn.Linear(d_model, d_model, bias=True)

        self._rope_cache = None  # (cos, sin, max_pos, dtype, device)

    def _get_rope(
        self,
        rope_pos: torch.Tensor,
        dtype: torch.dtype,
        device: torch.device,
        max_pos_hint: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # shared logic with self-attn; keep minimal
        if self.rotary_dim == 0:
            raise RuntimeError("rotary_dim is 0 but _get_rope called")

        max_pos = int(max_pos_hint) if max_pos_hint is not None else (int(rope_pos.max().item()) + 1)

        need_rebuild = True
        if self._rope_cache is not None:
            cos, sin, cached_max, cached_dtype, cached_device = self._rope_cache
            if cached_max >= max_pos and cached_dtype == dtype and cached_device == device:
                need_rebuild = False
        if need_rebuild:
            cos, sin = build_rope_cache(
                max_pos=max_pos,
                rotary_dim=self.rotary_dim,
                theta=self.rope_theta,
                device=device,
                dtype=dtype,
            )
            self._rope_cache = (cos, sin, max_pos, dtype, device)

        cos, sin, _, _, _ = self._rope_cache
        if rope_pos.dim() == 1:
            return cos.index_select(0, rope_pos), sin.index_select(0, rope_pos)
        elif rope_pos.dim() == 2:
            half = cos.shape[1]
            cos_g = cos.index_select(0, rope_pos.reshape(-1)).reshape(rope_pos.shape[0], rope_pos.shape[1], half)
            sin_g = sin.index_select(0, rope_pos.reshape(-1)).reshape(rope_pos.shape[0], rope_pos.shape[1], half)
            return cos_g, sin_g
        else:
            raise ValueError(f"rope_pos must be (N,) or (B,N), got {rope_pos.shape}")

    def forward(
        self,
        q_in: torch.Tensor,                 # (B, Lq, D)
        kv_in: torch.Tensor,                # (B, Lk, D)
        kv_padding_mask: Optional[torch.Tensor],  # (B, Lk) bool True=PAD
        rope_pos_q: torch.Tensor,           # (B, Lq) or (Lq,)
        rope_pos_k: torch.Tensor,           # (B, Lk) or (Lk,)
    ) -> torch.Tensor:
        B, Lq, D = q_in.shape
        _, Lk, _ = kv_in.shape

        q = self.q(q_in).view(B, Lq, self.n_heads, self.head_dim).transpose(1, 2)  # (B,H,Lq,hd)
        kv = self.kv(kv_in)
        k, v = kv.chunk(2, dim=-1)
        k = k.view(B, Lk, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, Lk, self.n_heads, self.head_dim).transpose(1, 2)

        if self.rotary_dim > 0:
            cos_q, sin_q = self._get_rope(rope_pos_q, dtype=q.dtype, device=q.device, max_pos_hint=Lq)
            cos_k, sin_k = self._get_rope(rope_pos_k, dtype=q.dtype, device=q.device, max_pos_hint=Lk)

            q_rot, q_pass = q[..., : self.rotary_dim], q[..., self.rotary_dim :]
            k_rot, k_pass = k[..., : self.rotary_dim], k[..., self.rotary_dim :]
            q_rot = apply_rope(q_rot, cos_q, sin_q)
            k_rot = apply_rope(k_rot, cos_k, sin_k)
            q = torch.cat([q_rot, q_pass], dim=-1)
            k = torch.cat([k_rot, k_pass], dim=-1)

        attn_dtype = v.dtype   # 보통 bf16/fp16 유지
        # Optional QK normalization (attention stability)
        if self.qk_norm != "off":
            eps = self.qk_norm_eps
            if self.qk_norm == "l2":
                q_norm = torch.linalg.vector_norm(
                    q, dim=-1, keepdim=True, dtype=torch.float32
                ).clamp_min(eps)
                k_norm = torch.linalg.vector_norm(
                    k, dim=-1, keepdim=True, dtype=torch.float32
                ).clamp_min(eps)
                scale = self.head_dim ** 0.5

                q = (q.float() * (scale / q_norm)).to(attn_dtype)
                k = (k.float() * (scale / k_norm)).to(attn_dtype)
            elif self.qk_norm in ("rms", "rmsnorm"):
                qw = self.q_norm_weight.to(dtype=attn_dtype, device=q.device)
                kw = self.k_norm_weight.to(dtype=attn_dtype, device=k.device)
                q = F.rms_norm(q.to(attn_dtype), (self.head_dim,), weight=qw, eps=eps)
                k = F.rms_norm(k.to(attn_dtype), (self.head_dim,), weight=kw, eps=eps)

            elif self.qk_norm in ("layernorm", "ln"):
                qw = self.q_norm_weight.to(dtype=attn_dtype, device=q.device)
                kw = self.k_norm_weight.to(dtype=attn_dtype, device=k.device)
                qb = self.q_norm_bias.to(dtype=attn_dtype, device=q.device)
                kb = self.k_norm_bias.to(dtype=attn_dtype, device=k.device)

                q = F.layer_norm(q.to(attn_dtype), (self.head_dim,), weight=qw, bias=qb, eps=eps)
                k = F.layer_norm(k.to(attn_dtype), (self.head_dim,), weight=kw, bias=kb, eps=eps)
            else:
                raise ValueError(f"Unknown attn_qk_norm: {self.qk_norm}")
        if q.dtype != v.dtype or k.dtype != v.dtype:
            q = q.to(v.dtype)
            k = k.to(v.dtype)
            
        attn_mask = None
        if kv_padding_mask is not None:
            attn_mask = (~kv_padding_mask)[:, None, None, :]  # (B,1,1,Lk)

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.attn_dropout if self.training else 0.0,
            is_causal=False,
        )  # (B,H,Lq,hd)

        out = out.transpose(1, 2).contiguous().view(B, Lq, D)
        return self.out(out)

class LayerNorm(nn.LayerNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_dtype = x.dtype
        y = F.layer_norm(
            x.float(),
            self.normalized_shape,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        )
        return y.to(x_dtype)
    
class RMSNorm(nn.LayerNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_dtype = x.dtype
        y = F.rms_norm(
            x.float(),
            self.normalized_shape,
            self.weight.float() if self.weight is not None else None,
            self.eps,
        )
        return y.to(x_dtype)

def make_norm(norm_type: str, dim: int, eps: float = 1e-6) -> nn.Module:
    norm_type = norm_type.lower()
    if norm_type in ("layernorm", "ln"):
        return LayerNorm(dim, eps=eps)
    if norm_type in ("rmsnorm", "rms"):
        return RMSNorm(dim, eps=eps)
    raise ValueError(f"Unknown norm_type: {norm_type}")

class LayerScale(nn.Module):
    def __init__(self, dim: int, init_value: float = 1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim) * init_value)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,N,dim)
        return x * self.gamma.to(dtype=x.dtype, device=x.device)

class MLP_GELU(nn.Module):
    def __init__(self, d_model: int, mlp_ratio: float, dropout: float):
        super().__init__()
        hidden = int(d_model * mlp_ratio)
        self.fc1 = nn.Linear(d_model, hidden)
        self.fc2 = nn.Linear(hidden, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class MLP_Gated(nn.Module):
    """
    GEGLU: GELU(a) * b
    SwiGLU: SiLU(a) * b
    set hidden scale to 2/3 to compare with GELU_MLP (similar #params)
    """
    def __init__(self, d_model: int, mlp_ratio: float, dropout: float, act: str = "swiglu", gate_scale: float = 2/3):
        super().__init__()
        act = act.lower()
        assert act in ("geglu", "swiglu")
        hidden = int(d_model * mlp_ratio * gate_scale)
        self.fc = nn.Linear(d_model, 2 * hidden)
        self.proj = nn.Linear(hidden, d_model)
        self.drop = nn.Dropout(dropout)
        self.act = act

    def forward(self, x):
        a, b = self.fc(x).chunk(2, dim=-1)
        if self.act == "geglu":
            x = F.gelu(a) * b
        else:
            x = F.silu(a) * b
        x = self.drop(x)
        x = self.proj(x)
        x = self.drop(x)
        return x

def make_mlp(mlp_type: str, d_model: int, mlp_ratio: float, dropout: float) -> nn.Module:
    mlp_type = mlp_type.lower()
    if mlp_type in ("gelu", "mlp"):
        return MLP_GELU(d_model, mlp_ratio, dropout)
    if mlp_type in ("geglu",):
        return MLP_Gated(d_model, mlp_ratio, dropout, act="geglu")
    if mlp_type in ("swiglu",):
        return MLP_Gated(d_model, mlp_ratio, dropout, act="swiglu")
    raise ValueError(f"Unknown mlp_type: {mlp_type}")



# ============================================================
# Hybrid encoder blocks: divided spatiotemporal attention + occasional full attention
# ============================================================
class FullAttentionBlock(nn.Module):
    """Full self-attention over packed (channel,time) tokens.

    - RoPE on time indices
    - Optional spatial relative bias (channel-channel) added to attention logits
    """

    def __init__(self, cfg: EEGModelConfig):
        super().__init__()
        self.use_spatial_bias = bool(getattr(cfg, "full_attn_use_spatial_bias", True))

        self.norm1 = make_norm(cfg.norm_type, cfg.d_model, eps=1e-6)
        self.attn = MultiheadSelfAttentionRoPE(
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            attn_dropout=cfg.attn_dropout,
            rope_theta=cfg.rope_theta,
            rotary_pct=cfg.rotary_pct,
            qk_norm=getattr(cfg, "attn_qk_norm", "off"),
            qk_norm_eps=getattr(cfg, "attn_qk_norm_eps", 1e-6),
            spatial_qk=getattr(cfg, "spatial_qk_type", "none"),
            spatial_qk_dim=getattr(cfg, "spatial_qk_feat_dim", 64),
            spatial_qk_scale=getattr(cfg, "spatial_qk_scale", 1.0)
        )

        self.norm2 = make_norm(cfg.norm_type, cfg.d_model, eps=1e-6)
        self.mlp = make_mlp(cfg.mlp_type, cfg.d_model, cfg.mlp_ratio, cfg.dropout)
        self.dropout = nn.Dropout(cfg.dropout)

        if getattr(cfg, "layerscale_init", 0.0) and cfg.layerscale_init > 0:
            self.ls1 = LayerScale(cfg.d_model, init_value=cfg.layerscale_init)
            self.ls2 = LayerScale(cfg.d_model, init_value=cfg.layerscale_init)
        else:
            self.ls1 = None
            self.ls2 = None

        self.use_bucket_full = bool(getattr(cfg, "use_bucket_full", True))
        self.bucket_full_exact_bias = bool(getattr(cfg, "bucket_full_exact_bias", False))

    @staticmethod
    def _build_sample_length_buckets(padding_mask: torch.Tensor):
        """
        padding_mask: (B,L) True=PAD
        returns:
        buckets[seqlen] = {
            "flat_idx": [LongTensor(S), ...],   # one per sample
            "sample_ids": [int, ...],
        }
        """
        B, L = padding_mask.shape
        lengths = (~padding_mask).sum(dim=1)   # (B,)
        base = torch.arange(L, device=padding_mask.device, dtype=torch.long)

        buckets = {}
        for b in range(B):
            s = int(lengths[b].item())
            if s <= 0:
                continue
            payload = buckets.setdefault(s, {"flat_idx": [], "sample_ids": []})
            payload["flat_idx"].append(base[:s] + b * L)   # packed valid prefix
            payload["sample_ids"].append(b)
        return buckets
    
    def _run_bucketed_full_attn(
        self,
        x: torch.Tensor,                     # (B,L,D)
        padding_mask: torch.Tensor,          # (B,L)
        rope_pos: torch.Tensor,              # (B,L)
        chan_idx: torch.Tensor,              # (B,L)
        spatial_bias_cc: Optional[torch.Tensor],   # (B,C,C) or None
        spatial_qk_ch: Optional[torch.Tensor],     # (B,C,Dq) or None
        rope_seq_len: Optional[int] = None,
    ) -> torch.Tensor:
        B, L, D = x.shape
        x_flat = x.reshape(B * L, D)
        out_flat = None

        buckets = self._build_sample_length_buckets(padding_mask)
        if len(buckets) == 0:
            return x.new_zeros((B, L, D))

        rope_flat = rope_pos.reshape(B * L)
        chan_flat = chan_idx.reshape(B * L)

        q_add_flat = None
        k_add_flat = None
        if (spatial_qk_ch is not None) and (self.attn.spatial_q_proj is not None):
            q_sp_ch = self.attn.spatial_q_proj(spatial_qk_ch)   # (B,C,D)
            k_sp_ch = self.attn.spatial_k_proj(spatial_qk_ch)   # (B,C,D)
            q_tok = _gather_channel_features(q_sp_ch, chan_idx).masked_fill(padding_mask[..., None], 0.0)
            k_tok = _gather_channel_features(k_sp_ch, chan_idx).masked_fill(padding_mask[..., None], 0.0)
            q_add_flat = q_tok.reshape(B * L, D)
            k_add_flat = k_tok.reshape(B * L, D)

        for seqlen, payload in buckets.items():
            idx = torch.stack(payload["flat_idx"], dim=0)   # (G,S)
            G = idx.shape[0]

            xb = x_flat.index_select(0, idx.reshape(-1)).view(G, seqlen, D)
            ropeb = rope_flat.index_select(0, idx.reshape(-1)).view(G, seqlen)

            q_add = None
            k_add = None
            if q_add_flat is not None:
                q_add = q_add_flat.index_select(0, idx.reshape(-1)).view(G, seqlen, D)
                k_add = k_add_flat.index_select(0, idx.reshape(-1)).view(G, seqlen, D)

            biasb = None
            if spatial_bias_cc is not None:
                # exact semantics, but no longer flash-friendly
                sample_ids = torch.tensor(payload["sample_ids"], device=x.device, dtype=torch.long)
                c_sel = chan_flat.index_select(0, idx.reshape(-1)).view(G, seqlen)
                bias_src = spatial_bias_cc.index_select(0, sample_ids)   # (G,C,C)
                g = torch.arange(G, device=x.device)[:, None, None]
                biasb = bias_src[g, c_sel[:, :, None], c_sel[:, None, :]]   # (G,S,S)

            yb = self.attn(
                xb,
                padding_mask=None,
                rope_pos=ropeb,
                attn_bias=biasb,
                spatial_q_add=q_add,
                spatial_k_add=k_add,
                rope_seq_len=rope_seq_len,
            )

            if out_flat is None:
                out_flat = yb.new_zeros((B * L, D))

            out_flat.index_copy_(0, idx.reshape(-1), yb.reshape(-1, D))

        if out_flat is None:
            out_flat = x.new_zeros((B * L, D))

        return out_flat.view(B, L, D)

    def forward(
        self,
        x,
        padding_mask,
        rope_pos,
        chan_idx,
        spatial_bias_cc,
        spatial_qk_ch,
        grid_channels: Optional[int] = None,
        grid_patches: Optional[int] = None,
    ):
        if padding_mask is None:
            padding_mask = torch.zeros(x.shape[:2], dtype=torch.bool, device=x.device)

        # bucketed full path를 쓸지 결정
        has_real_pad = bool(padding_mask.any().item())
        can_bucket = self.use_bucket_full and has_real_pad and (chan_idx is not None)

        # flash-friendly path: bias 없음
        if can_bucket and ((not self.use_spatial_bias) or (spatial_bias_cc is None)):
            y = self._run_bucketed_full_attn(
                self.norm1(x),
                padding_mask=padding_mask,
                rope_pos=rope_pos,
                chan_idx=chan_idx,
                spatial_bias_cc=None,
                spatial_qk_ch=spatial_qk_ch,
                rope_seq_len=grid_patches,
            )
        # exact semantics with bias, but likely not flash
        elif can_bucket and self.bucket_full_exact_bias and self.use_spatial_bias and (spatial_bias_cc is not None):
            y = self._run_bucketed_full_attn(
                self.norm1(x),
                padding_mask=padding_mask,
                rope_pos=rope_pos,
                chan_idx=chan_idx,
                spatial_bias_cc=spatial_bias_cc,
                spatial_qk_ch=spatial_qk_ch,
                rope_seq_len=grid_patches,
            )
        else:
            attn_bias = None
            spatial_q_add = None
            spatial_k_add = None

            if (spatial_qk_ch is not None) and (chan_idx is not None) and (self.attn.spatial_q_proj is not None):
                q_sp_ch = self.attn.spatial_q_proj(spatial_qk_ch)
                k_sp_ch = self.attn.spatial_k_proj(spatial_qk_ch)
                spatial_q_add = _gather_channel_features(q_sp_ch, chan_idx)
                spatial_k_add = _gather_channel_features(k_sp_ch, chan_idx)
                if padding_mask is not None:
                    spatial_q_add = spatial_q_add.masked_fill(padding_mask[..., None], 0.0)
                    spatial_k_add = spatial_k_add.masked_fill(padding_mask[..., None], 0.0)

            if self.use_spatial_bias and (spatial_bias_cc is not None) and (chan_idx is not None):
                c = chan_idx
                B, N = c.shape
                b = torch.arange(B, device=c.device)[:, None, None]
                attn_bias = spatial_bias_cc[b, c[:, :, None], c[:, None, :]]

            y = self.attn(
                self.norm1(x),
                padding_mask=padding_mask,
                rope_pos=rope_pos,
                attn_bias=attn_bias,
                spatial_q_add=spatial_q_add,
                spatial_k_add=spatial_k_add,
                rope_seq_len=grid_patches,
            )

        if self.ls1 is not None:
            y = self.ls1(y)
        x = x + self.dropout(y)

        y = self.mlp(self.norm2(x))
        if self.ls2 is not None:
            y = self.ls2(y)
        x = x + self.dropout(y)
        return x



class DividedSpatiotemporalBlock(nn.Module):
    """Divided attention block (Time-pass then Space-pass) on packed tokens.

    Temporal pass (per-channel sequences):
      - RoPE(time)

    Spatial pass (per-time sequences):
      - spatial relative bias (Legendre)

    This is designed for EEG tokens where (C,P_t) is sparse/packed due to JEPA masking.
    """

    def __init__(self, cfg: EEGModelConfig):
        super().__init__()
        self.cfg = cfg
        D = cfg.d_model

        # temporal pass
        self.norm_t = make_norm(cfg.norm_type, D, eps=1e-6)
        self.attn_t = MultiheadSelfAttentionRoPE(
            d_model=D,
            n_heads=cfg.n_heads,
            attn_dropout=cfg.attn_dropout,
            rope_theta=cfg.rope_theta,
            rotary_pct=cfg.rotary_pct,
            qk_norm=getattr(cfg, "attn_qk_norm", "off"),
            qk_norm_eps=getattr(cfg, "attn_qk_norm_eps", 1e-6),
        )

        # spatial pass
        self.norm_s = make_norm(cfg.norm_type, D, eps=1e-6)
        self.attn_s = MultiheadSelfAttentionRoPE(
            d_model=D,
            n_heads=cfg.n_heads,
            attn_dropout=cfg.attn_dropout,
            rope_theta=cfg.rope_theta,
            rotary_pct=0.0,  # no RoPE in spatial pass
            qk_norm=getattr(cfg, "attn_qk_norm", "off"),
            qk_norm_eps=getattr(cfg, "attn_qk_norm_eps", 1e-6),
            spatial_qk=getattr(cfg, "spatial_qk_type", "none"),
            spatial_qk_dim=getattr(cfg, "spatial_qk_feat_dim", 64),
            spatial_qk_scale=getattr(cfg, "spatial_qk_scale", 1.0)
        )

        # MLP
        self.norm_m = make_norm(cfg.norm_type, D, eps=1e-6)
        self.mlp = make_mlp(cfg.mlp_type, D, cfg.mlp_ratio, cfg.dropout)

        self.dropout = nn.Dropout(cfg.dropout)

        if getattr(cfg, "layerscale_init", 0.0) and cfg.layerscale_init > 0:
            self.ls_t = LayerScale(D, init_value=cfg.layerscale_init)
            self.ls_s = LayerScale(D, init_value=cfg.layerscale_init)
            self.ls_m = LayerScale(D, init_value=cfg.layerscale_init)
        else:
            self.ls_t = None
            self.ls_s = None
            self.ls_m = None

        self._batch_rows_cache = {}
        self._temporal_rope_cache = {}        
        self.use_bucket_divided = bool(getattr(cfg, "use_bucket_divided", True))

    @staticmethod
    def _device_key(device: torch.device):
        return (device.type, -1 if device.index is None else int(device.index))

    def _get_batch_rows(self, batch_size: int, device: torch.device) -> torch.Tensor:
        key = (self._device_key(device), int(batch_size))
        rows = self._batch_rows_cache.get(key)
        if rows is None:
            rows = torch.arange(batch_size, device=device, dtype=torch.long)[:, None]
            self._batch_rows_cache[key] = rows
        return rows

    def _get_temporal_rope(self, P: int, device: torch.device) -> torch.Tensor:
        key = (self._device_key(device), int(P))
        rope = self._temporal_rope_cache.get(key)
        if rope is None:
            rope = torch.arange(P, device=device, dtype=torch.long)
            self._temporal_rope_cache[key] = rope
        return rope

    def _scatter_to_grid(
        self,
        x: torch.Tensor,            # (B,L,D)
        pad: torch.Tensor,          # (B,L) True=PAD
        c_idx: torch.Tensor,            # (B,L) channel indices (safe)
        t_idx: torch.Tensor,            # (B,L) time indices (safe)
        C: int,
        P: int,
        b_rows: torch.Tensor,       # (B,1)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Scatter packed tokens to a dense (B,C,P,D) grid.

        Returns:
          grid: (B,C,P,D)
          grid_pad: (B,C,P) True=missing
        """
        B, L, D = x.shape
        device = x.device
        grid = x.new_zeros((B, C, P + 1, D))
        grid_pad = torch.ones((B, C, P + 1), dtype=torch.bool, device=device)

        b = b_rows.expand(B, L)
        trash = torch.full_like(t_idx, P)
        t_safe = torch.where(pad, trash, t_idx)

        grid[b, c_idx, t_safe] = x
        grid_pad[b, c_idx, t_safe] = pad
        return grid[:, :, :P, :], grid_pad[:, :, :P]

    def _gather_from_grid(
        self,
        grid: torch.Tensor,         # (B,C,P,D)
        pad: torch.Tensor,          # (B,L)
        c: torch.Tensor,            # (B,L)
        t: torch.Tensor,            # (B,L)
        b_rows: torch.Tensor,       # (B,1)
    ) -> torch.Tensor:
        B, L = c.shape
        out = grid[b_rows.expand(B, L), c, t]
        return out.masked_fill(pad[..., None], 0.0)

    @staticmethod
    def _mask_grid_output(x: torch.Tensor, grid_pad: torch.Tensor) -> torch.Tensor:
        return x.masked_fill(grid_pad[..., None], 0.0)

    def _temporal_from_grid(
        self,
        grid: torch.Tensor,         # (B,C,P,D)
        grid_pad: torch.Tensor,     # (B,C,P)
        P: int,
    ) -> torch.Tensor:
        B, C, P2, D = grid.shape
        assert P2 == P
        x_t = grid.reshape(B * C, P, D)
        pad_t = grid_pad.reshape(B * C, P)

        all_pad = pad_t.all(dim=1)
        pad_t_safe = pad_t.clone()
        pad_t_safe[:, 0] = pad_t_safe[:, 0] & (~all_pad)

        rope = self._get_temporal_rope(P, grid.device)
        y_t = self.attn_t(
            x_t,
            padding_mask=pad_t_safe,
            rope_pos=rope,
            attn_bias=None,
            rope_seq_len=P,
        )
        y_t = y_t.masked_fill(all_pad[:, None, None], 0.0)
        return y_t.reshape(B, C, P, D)

    def _spatial_from_grid(
        self,
        grid: torch.Tensor,                     # (B,C,P,D)
        grid_pad: torch.Tensor,                 # (B,C,P)
        spatial_bias_cc: Optional[torch.Tensor],
        spatial_qk_ch: Optional[torch.Tensor],  # (B,C,F)
        P: int,
    ) -> torch.Tensor:
        B, C, P2, D = grid.shape
        assert P2 == P

        grid_tp = grid.permute(0, 2, 1, 3).contiguous()   # (B,P,C,D)
        pad_tp = grid_pad.permute(0, 2, 1).contiguous()   # (B,P,C)
        x_s = grid_tp.reshape(B * P, C, D)
        pad_s = pad_tp.reshape(B * P, C)

        all_pad = pad_s.all(dim=1)
        pad_s_safe = pad_s.clone()
        pad_s_safe[:, 0] = pad_s_safe[:, 0] & (~all_pad)

        bias = None
        spatial_q_add = None
        spatial_k_add = None
        if spatial_bias_cc is not None:
            bias = spatial_bias_cc[:, None, :, :].expand(-1, P, -1, -1).reshape(B * P, C, C)
        else:
            if (spatial_qk_ch is not None) and (self.attn_s.spatial_q_proj is not None):
                q_sp_ch = self.attn_s.spatial_q_proj(spatial_qk_ch)   # (B,C,D)
                k_sp_ch = self.attn_s.spatial_k_proj(spatial_qk_ch)   # (B,C,D)
                spatial_q_add = q_sp_ch[:, None, :, :].expand(-1, P, -1, -1).reshape(B * P, C, D)
                spatial_k_add = k_sp_ch[:, None, :, :].expand(-1, P, -1, -1).reshape(B * P, C, D)

        y_s = self.attn_s(
            x_s,
            padding_mask=pad_s_safe,
            rope_pos=None,
            attn_bias=bias,
            spatial_q_add=spatial_q_add,
            spatial_k_add=spatial_k_add,
        )
        y_s = y_s.masked_fill(all_pad[:, None, None], 0.0)

        grid_tp_out = y_s.reshape(B, P, C, D)
        return grid_tp_out.permute(0, 2, 1, 3).contiguous()

    def _valid_token_meta(
        self,
        padding_mask: torch.Tensor,
        chan_idx: torch.Tensor,
        rope_pos: torch.Tensor,
    ):
        B, L = padding_mask.shape
        b_ids, l_ids = torch.where(~padding_mask)    # valid only
        flat_ids = b_ids * L + l_ids
        c_ids = chan_idx[b_ids, l_ids]
        t_ids = rope_pos[b_ids, l_ids]
        return b_ids, l_ids, flat_ids, c_ids, t_ids

    @staticmethod
    def _build_length_buckets(
        group_keys: torch.Tensor,
        order_keys: torch.Tensor,
        flat_ids: torch.Tensor,
        sort_base: int,
        aux_ids: Optional[torch.Tensor] = None,
    ):
        """
        group_keys: token이 속한 group id
        order_keys: group 내부 정렬 기준
        flat_ids: x.view(B*L, D)에서의 flat index
        aux_ids: temporal에서는 rope_pos(t_idx), spatial에서는 None
        """
        if flat_ids.numel() == 0:
            return {}

        sort_key = group_keys.to(torch.long) * int(sort_base) + order_keys.to(torch.long)
        perm = torch.argsort(sort_key)

        group_sorted = group_keys[perm]
        flat_sorted = flat_ids[perm]
        aux_sorted = None if aux_ids is None else aux_ids[perm]

        _, counts = torch.unique_consecutive(group_sorted, return_counts=True)
        starts = torch.cat([counts.new_zeros(1), counts.cumsum(0)[:-1]], dim=0)

        buckets = {}
        for start, seqlen in zip(starts.tolist(), counts.tolist()):
            sl = slice(start, start + seqlen)
            payload = buckets.setdefault(
                seqlen,
                {"flat_idx": [], "aux": [] if aux_sorted is not None else None},
            )
            payload["flat_idx"].append(flat_sorted[sl])
            if aux_sorted is not None:
                payload["aux"].append(aux_sorted[sl])

        return buckets

    def _run_bucketed_self_attn(
        self,
        x: torch.Tensor,      # (B,L,D)
        buckets,
        attn: MultiheadSelfAttentionRoPE,
        rope_seq_len: Optional[int] = None,
        spatial_q_add_flat: Optional[torch.Tensor] = None,   # (B*L,D)
        spatial_k_add_flat: Optional[torch.Tensor] = None,   # (B*L,D)
    ) -> torch.Tensor:
        B, L, D = x.shape
        x_flat = x.reshape(B * L, D)
        out_flat = None

        for seqlen, payload in buckets.items():
            idx = torch.stack(payload["flat_idx"], dim=0)   # (G,S)
            G = idx.shape[0]

            xb = x_flat.index_select(0, idx.reshape(-1)).view(G, seqlen, D)

            rope = None
            if payload["aux"] is not None:
                rope = torch.stack(payload["aux"], dim=0)   # (G,S)

            q_add = None
            k_add = None
            if spatial_q_add_flat is not None:
                q_add = spatial_q_add_flat.index_select(0, idx.reshape(-1)).view(G, seqlen, D)
                k_add = spatial_k_add_flat.index_select(0, idx.reshape(-1)).view(G, seqlen, D)

            yb = attn(
                xb,
                padding_mask=None,
                rope_pos=rope,
                attn_bias=None,
                spatial_q_add=q_add,
                spatial_k_add=k_add,
                rope_seq_len=rope_seq_len,
            )
            if out_flat is None:
                out_flat = yb.new_zeros((B * L, D))

            out_flat.index_copy_(0, idx.reshape(-1), yb.reshape(-1, D))

        if out_flat is None:
            out_flat = x.new_zeros((B * L, D))
        return out_flat.view(B, L, D)

    def _temporal_bucketed(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor,
        rope_pos: torch.Tensor,
        chan_idx: torch.Tensor,
        C: int,
        P: int,
    ) -> torch.Tensor:
        B, L, D = x.shape
        b_ids, _, flat_ids, c_ids, t_ids = self._valid_token_meta(
            padding_mask=padding_mask,
            chan_idx=chan_idx,
            rope_pos=rope_pos,
        )

        if flat_ids.numel() == 0:
            return x.new_zeros((B, L, D))

        # group = (batch, channel), order = time index
        group_keys = b_ids.to(torch.long) * int(C) + c_ids.to(torch.long)

        buckets = self._build_length_buckets(
            group_keys=group_keys,
            order_keys=t_ids,
            flat_ids=flat_ids,
            sort_base=P,
            aux_ids=t_ids,   # actual rope positions
        )

        return self._run_bucketed_self_attn(
            x,
            buckets=buckets,
            attn=self.attn_t,
            rope_seq_len=P,
        )

    def _spatial_bucketed(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor,
        rope_pos: torch.Tensor,
        chan_idx: torch.Tensor,
        spatial_bias_cc: Optional[torch.Tensor],
        spatial_qk_ch: Optional[torch.Tensor],
        C: int,
        P: int,
    ) -> torch.Tensor:
        if spatial_bias_cc is not None:
            raise RuntimeError(
                "_spatial_bucketed only supports spatial_bias_cc=None. "
                "If you want flash-friendly spatial attention, use "
                "spatial_bias_type='none' with spatial_qk_type='legendre_anchor'."
            )

        B, L, D = x.shape
        b_ids, _, flat_ids, c_ids, t_ids = self._valid_token_meta(
            padding_mask=padding_mask,
            chan_idx=chan_idx,
            rope_pos=rope_pos,
        )

        if flat_ids.numel() == 0:
            return x.new_zeros((B, L, D))

        # group = (batch, time), order = channel index
        group_keys = b_ids.to(torch.long) * int(P) + t_ids.to(torch.long)

        buckets = self._build_length_buckets(
            group_keys=group_keys,
            order_keys=c_ids,
            flat_ids=flat_ids,
            sort_base=C,
            aux_ids=None,
        )

        spatial_q_add_flat = None
        spatial_k_add_flat = None
        if (spatial_qk_ch is not None) and (self.attn_s.spatial_q_proj is not None):
            q_sp_ch = self.attn_s.spatial_q_proj(spatial_qk_ch)  # (B,C,D)
            k_sp_ch = self.attn_s.spatial_k_proj(spatial_qk_ch)  # (B,C,D)

            q_sp_tok = _gather_channel_features(q_sp_ch, chan_idx).masked_fill(
                padding_mask[..., None], 0.0
            )
            k_sp_tok = _gather_channel_features(k_sp_ch, chan_idx).masked_fill(
                padding_mask[..., None], 0.0
            )

            spatial_q_add_flat = q_sp_tok.reshape(B * L, D)
            spatial_k_add_flat = k_sp_tok.reshape(B * L, D)

        return self._run_bucketed_self_attn(
            x,
            buckets=buckets,
            attn=self.attn_s,
            rope_seq_len=None,
            spatial_q_add_flat=spatial_q_add_flat,
            spatial_k_add_flat=spatial_k_add_flat,
        )
    
    def _forward_dense_grid(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor],
        rope_pos: torch.Tensor,
        chan_idx: Optional[torch.Tensor],
        spatial_bias_cc: Optional[torch.Tensor],
        spatial_qk_ch: Optional[torch.Tensor],
        grid_channels: Optional[int] = None,
        grid_patches: Optional[int] = None,
    ) -> torch.Tensor:
        if padding_mask is None:
            padding_mask = torch.zeros(x.shape[:2], dtype=torch.bool, device=x.device)
        if chan_idx is None:
            raise ValueError("DividedSpatiotemporalBlock requires chan_idx")

        b_rows = self._get_batch_rows(int(x.shape[0]), x.device)

        if grid_channels is not None:
            C = int(grid_channels)
        elif spatial_bias_cc is not None:
            C = int(spatial_bias_cc.shape[1])
        elif spatial_qk_ch is not None:
            C = int(spatial_qk_ch.shape[1])
        else:
            C = int(chan_idx.max().item()) + 1

        if grid_patches is not None:
            P = int(grid_patches)
        else:
            P = int(rope_pos.max().item()) + 1

        grid, grid_pad = self._scatter_to_grid(x, padding_mask, chan_idx, rope_pos, C=C, P=P, b_rows=b_rows)

        # temporal
        grid_t_in = self._mask_grid_output(self.norm_t(grid), grid_pad)
        y_t = self._temporal_from_grid(grid_t_in, grid_pad, P=P)
        if self.ls_t is not None:
            y_t = self.ls_t(y_t)
        y_t = self._mask_grid_output(self.dropout(y_t), grid_pad)
        grid = grid + y_t

        # spatial
        grid_s_in = self._mask_grid_output(self.norm_s(grid), grid_pad)
        y_s = self._spatial_from_grid(
            grid_s_in,
            grid_pad,
            spatial_bias_cc=spatial_bias_cc,
            spatial_qk_ch=spatial_qk_ch,
            P=P,
        )
        if self.ls_s is not None:
            y_s = self.ls_s(y_s)
        y_s = self._mask_grid_output(self.dropout(y_s), grid_pad)
        grid = grid + y_s

        # mlp
        grid_m_in = self._mask_grid_output(self.norm_m(grid), grid_pad)
        y_m = self.mlp(grid_m_in)
        if self.ls_m is not None:
            y_m = self.ls_m(y_m)
        y_m = self._mask_grid_output(self.dropout(y_m), grid_pad)
        grid = grid + y_m

        return self._gather_from_grid(grid, padding_mask, chan_idx, rope_pos, b_rows=b_rows)
    
    def forward(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor],
        rope_pos: torch.Tensor,
        chan_idx: Optional[torch.Tensor],
        spatial_bias_cc: Optional[torch.Tensor],
        spatial_qk_ch: Optional[torch.Tensor],
        grid_channels: Optional[int] = None,
        grid_patches: Optional[int] = None,
    ) -> torch.Tensor:
        if padding_mask is None:
            padding_mask = torch.zeros(x.shape[:2], dtype=torch.bool, device=x.device)
        if chan_idx is None:
            raise ValueError("DividedSpatiotemporalBlock requires chan_idx")

        if grid_channels is not None:
            C = int(grid_channels)
        elif spatial_bias_cc is not None:
            C = int(spatial_bias_cc.shape[1])
        elif spatial_qk_ch is not None:
            C = int(spatial_qk_ch.shape[1])
        else:
            C = int(chan_idx.max().item()) + 1

        if grid_patches is not None:
            P = int(grid_patches)
        else:
            P = int(rope_pos.max().item()) + 1

        # spatial_bias가 있으면 결국 additive attn_bias 경로라서 flash-friendly하지 않다.
        # 그 경우는 기존 dense-grid path로 fallback.
        use_bucketed = self.use_bucket_divided and (spatial_bias_cc is None)

        if not use_bucketed:
            return self._forward_dense_grid(
                x,
                padding_mask=padding_mask,
                rope_pos=rope_pos,
                chan_idx=chan_idx,
                spatial_bias_cc=spatial_bias_cc,
                spatial_qk_ch=spatial_qk_ch,
                grid_channels=C,
                grid_patches=P,
            )

        # temporal pass: (batch, channel)별 packed sequence
        y_t = self._temporal_bucketed(
            self.norm_t(x),
            padding_mask=padding_mask,
            rope_pos=rope_pos,
            chan_idx=chan_idx,
            C=C,
            P=P,
        )
        if self.ls_t is not None:
            y_t = self.ls_t(y_t)
        y_t = self.dropout(y_t).masked_fill(padding_mask[..., None], 0.0)
        x = x + y_t

        # spatial pass: (batch, time)별 packed sequence
        y_s = self._spatial_bucketed(
            self.norm_s(x),
            padding_mask=padding_mask,
            rope_pos=rope_pos,
            chan_idx=chan_idx,
            spatial_bias_cc=None,
            spatial_qk_ch=spatial_qk_ch,
            C=C,
            P=P,
        )
        if self.ls_s is not None:
            y_s = self.ls_s(y_s)
        y_s = self.dropout(y_s).masked_fill(padding_mask[..., None], 0.0)
        x = x + y_s

        # mlp
        y_m = self.mlp(self.norm_m(x))
        if self.ls_m is not None:
            y_m = self.ls_m(y_m)
        y_m = self.dropout(y_m).masked_fill(padding_mask[..., None], 0.0)
        x = x + y_m

        return x

class EEGEncoder(nn.Module):
    """
    EEG encoder (ViT-style) that can embed PACKED tokens:
      - time patch embedding (linear proj on window)
      - rFFT filterbank features + FiLM fusion
      - coord Fourier embedding
      - RoPE over time patch index
    """
    def __init__(self, cfg: EEGModelConfig):
        super().__init__()
        self.cfg = cfg
        self.patch_samples = int(round(cfg.sample_rate * cfg.patch_seconds))
        hop_sec = float(getattr(cfg, "patch_hop_seconds", cfg.patch_seconds))
        if hop_sec <= 0:
            hop_sec = float(cfg.patch_seconds)
        self.hop_samples = int(round(cfg.sample_rate * hop_sec))
        self.hop_samples = max(1, self.hop_samples)

        self.time_embed = TimePatchEmbed(cfg)
        self.freq_feat = RFFTFreqFeatures(cfg)
        if cfg.isFourier:
            self.coord_embed = CoordFourierEmbedding(
                d_model=cfg.d_model,
                num_freqs=cfg.coord_num_freqs,
                max_freq=cfg.coord_max_freq,
                include_raw=cfg.coord_include_raw,
                coord_jitter_std=cfg.coord_jitter_std,
                coord_jitter_prob=cfg.coord_jitter_prob,
                renormalize=cfg.coord_renormalize,
            )
        else:
            self.coord_embed = CoordMLPEmbedding(
                d_model=cfg.d_model,
                coord_jitter_std=cfg.coord_jitter_std,
                coord_jitter_prob=cfg.coord_jitter_prob,
                w_init=cfg.coord_w_init,
            )
        if cfg.film_hidden > 0:
            self.film = FiLMFusion(
                freq_dim=self.freq_feat.freq_dim,
                d_model=cfg.d_model,
                hidden=cfg.film_hidden,
            )
        else:
            self.film = None
        
        self.spatial_qk_feat = None
        if str(getattr(cfg, "spatial_qk_type", "none")).lower() == "legendre_anchor":
            self.spatial_qk_feat = LegendreAnchorFeatures(cfg)

        # shared spatial bias module (computed once per forward and reused across blocks)
        self.spatial_bias = SpatialBias(cfg)

        # encoder blocks
        arch = str(getattr(cfg, "encoder_arch", "hybrid")).lower()
        full_every = int(getattr(cfg, "full_attn_every", 4))
        blocks = []
        if arch == "full":
            blocks = [FullAttentionBlock(cfg) for _ in range(cfg.n_layers)]
        elif arch == "divided":
            blocks = [DividedSpatiotemporalBlock(cfg) for _ in range(cfg.n_layers)]
        elif arch == "hybrid":
            for i in range(cfg.n_layers):
                is_full = (full_every > 0) and ((i + 1) % full_every == 0)
                blocks.append(FullAttentionBlock(cfg) if is_full else DividedSpatiotemporalBlock(cfg))
        else:
            raise ValueError(f"Unknown encoder_arch: {arch}")

        self.blocks = nn.ModuleList(blocks)
        self.norm = make_norm(cfg.norm_type, cfg.d_model, eps=1e-6)

    def extract_patches_view(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B,C,T)
        returns view: (B,C,P,patch_samples) using unfold with hop_samples
        """
        return x.unfold(dimension=-1, size=self.patch_samples, step=self.hop_samples)

    @staticmethod
    def _safe_gather_channel(x: torch.Tensor, c_idx: torch.Tensor) -> torch.Tensor:
        """
        x: (B,C,D), c_idx: (B,L) long (>=0)
        returns: (B,L,D)
        """
        B, C, D = x.shape
        B2, L = c_idx.shape
        assert B == B2
        idx = c_idx[..., None].expand(B, L, D)
        return x.gather(dim=1, index=idx)


    def embed_from_indices(
        self,
        x: torch.Tensor,             # (B,C,T)
        coords: torch.Tensor,        # (B,C,3)
        c_idx: torch.Tensor,         # (B,L) long, channel index (>=0 for valid)
        t_idx: torch.Tensor,         # (B,L) long, patch-time index (>=0 for valid)
        pad: torch.Tensor,           # (B,L) bool True=PAD
        # optional precomputed channel coord embeddings
        coord_ch: Optional[torch.Tensor] = None,         # (B,C,D)
        # freq corruption (student context only)
        freq_mask_bins: Optional[torch.Tensor] = None,   # (B,K) bool True=mask
        freq_domain_drop: Optional[torch.Tensor] = None, # (B,) bool
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute packed token embeddings for (c_idx, t_idx).

        Returns:
          tok: (B,L,D)
          pad: (B,L) True=PAD
          rope_pos: (B,L) long (time patch indices)
          chan_idx: (B,L) long (channel indices; PAD already clamped)
        """
        device = x.device
        B, C, T = x.shape
        B2, L = c_idx.shape
        assert B == B2
        assert t_idx.shape == (B, L)
        assert pad.shape == (B, L)

        # safe indices for gather (PAD -> 0)
        c_safe = c_idx.clamp(min=0)
        t_safe = t_idx.clamp(min=0)

        # coord embeddings per channel -> gather by c_idx
        if coord_ch is None:
            coord_ch = self.coord_embed(coords)  # (B,C,D)
        coord_tok = self._safe_gather_channel(coord_ch, c_safe)  # (B,L,D)
        coord_tok = coord_tok.masked_fill(pad[..., None], 0.0)

        # patch signals: unfold view then advanced index gather
        patches_view = self.extract_patches_view(x)  # (B,C,P,S) view
        # build batch index for advanced indexing
        b_idx = torch.arange(B, device=device)[:, None].expand(B, L)
        patches = patches_view[b_idx, c_safe, t_safe]  # (B,L,S)
        patches = patches.masked_fill(pad[..., None], 0.0)

        # time embedding + coord
        e_time = self.time_embed.forward_packed(patches)  # (B,L,D)
        e = e_time + coord_tok

        # freq features
        if self.film is not None:
            f = self.freq_feat.forward_packed(patches)  # (B,L,F)

            # apply freq corruption (only intended for student)
            if freq_domain_drop is not None:
                # drop all freq dims (including scale) for dropped samples
                if freq_domain_drop.device != device or freq_domain_drop.dtype != torch.bool:
                    freq_domain_drop = freq_domain_drop.to(device=device, dtype=torch.bool)
                drop = freq_domain_drop[:, None]  # (B,1)
                f = f.masked_fill(drop[..., None] & (~pad)[..., None], 0.0)

            if freq_mask_bins is not None:
                # only mask the first K bins; keep optional scale dim intact
                K = self.cfg.freq_bins
                if freq_mask_bins.device != device or freq_mask_bins.dtype != torch.bool:
                    freq_mask_bins = freq_mask_bins.to(device=device, dtype=torch.bool)
                band = freq_mask_bins[:, None, :]  # (B,1,K)
                f_shape = f[..., :K].masked_fill(band & (~pad[..., None]), 0.0)
                if f.shape[-1] > K:
                    f = torch.cat([f_shape, f[..., K:]], dim=-1)
                else:
                    f = f_shape

            # FiLM fuse
            tok = self.film(e, f)
        else:
            tok = e
        tok = tok.masked_fill(pad[..., None], 0.0)

        rope_pos = t_safe  # (B,L) patch indices
        chan_idx = c_safe  # (B,L)
        return tok, pad, rope_pos, chan_idx

    def forward(
        self,
        tokens: torch.Tensor,
        padding_mask: torch.Tensor,
        rope_pos: torch.Tensor,
        chan_idx: torch.Tensor,
        coords: Optional[torch.Tensor] = None,
        grid_patches: Optional[int] = None,
    ) -> torch.Tensor:
        # Precompute spatial bias / spatial QK features once per forward.
        spatial_bias_cc = None
        spatial_qk_ch = None
        grid_channels = None

        if coords is not None:
            grid_channels = int(coords.shape[1])

            spatial_bias_cc = self.spatial_bias(coords)
            if spatial_bias_cc is not None and (
                spatial_bias_cc.device != tokens.device or spatial_bias_cc.dtype != tokens.dtype
            ):
                spatial_bias_cc = spatial_bias_cc.to(dtype=tokens.dtype, device=tokens.device)

            if self.spatial_qk_feat is not None:
                spatial_qk_ch = self.spatial_qk_feat(coords)
                if spatial_qk_ch.device != tokens.device or spatial_qk_ch.dtype != tokens.dtype:
                    spatial_qk_ch = spatial_qk_ch.to(dtype=tokens.dtype, device=tokens.device)

        if grid_patches is None and rope_pos.numel() > 0:
            grid_patches = int(rope_pos.max().item()) + 1

        x = tokens
        for blk in self.blocks:
            x = blk(
                x,
                padding_mask=padding_mask,
                rope_pos=rope_pos,
                chan_idx=chan_idx,
                spatial_bias_cc=spatial_bias_cc,
                spatial_qk_ch=spatial_qk_ch,
                grid_channels=grid_channels,
                grid_patches=grid_patches,
            )
        x = self.norm(x)
        return x

    # HF-like save/load
    def save_pretrained(self, out_dir: str) -> None:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(self.cfg.to_dict(), f, indent=2, ensure_ascii=False)
        torch.save(self.state_dict(), os.path.join(out_dir, "pytorch_model.bin"))

    @staticmethod
    def from_pretrained(path: str, map_location: str = "cpu") -> "EEGEncoder":
        cfg = EEGModelConfig.from_json(os.path.join(path, "config.json"))
        model = EEGEncoder(cfg)
        sd = torch.load(os.path.join(path, "pytorch_model.bin"), map_location=map_location, weights_only=True)
        model.load_state_dict(sd, strict=True)
        return model


class PredictorMLP(nn.Module):
    """
    (Kept for ablations / debugging.)
    """
    def __init__(self, d_model: int, hidden: int, depth: int, dropout: float, norm_type: str):
        super().__init__()
        layers = []
        in_dim = d_model
        for _ in range(max(1, depth - 1)):
            layers += [make_norm(norm_type, in_dim, eps=1e-6), nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout)]
            in_dim = hidden
        layers += [make_norm(norm_type, in_dim, eps=1e-6), nn.Linear(in_dim, d_model)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CrossAttnPredictorBlock(nn.Module):
    def __init__(self, cfg: EEGModelConfig):
        super().__init__()
        d_model = cfg.d_model
        n_heads = getattr(cfg, "predictor_n_heads", cfg.n_heads)
        mlp_ratio = getattr(cfg, "predictor_mlp_ratio", cfg.mlp_ratio)
        dropout = cfg.dropout
        attn_dropout = cfg.attn_dropout
        rope_theta = cfg.rope_theta
        rotary_pct = cfg.rotary_pct

        self.norm1 = make_norm(cfg.norm_type, cfg.d_model, eps=1e-6)
        self.xattn = CrossAttentionRoPE(
            d_model=d_model,
            n_heads=n_heads,
            attn_dropout=attn_dropout,
            rope_theta=rope_theta,
            rotary_pct=rotary_pct,
        )
        self.drop = nn.Dropout(dropout)
        self.norm2 = make_norm(cfg.norm_type, cfg.d_model, eps=1e-6)
        self.mlp = make_mlp(cfg.mlp_type, d_model, mlp_ratio=mlp_ratio, dropout=dropout)

    def forward(self, q: torch.Tensor, ctx: torch.Tensor, ctx_pad: torch.Tensor, rope_q: torch.Tensor, rope_ctx: torch.Tensor) -> torch.Tensor:
        q = q + self.drop(self.xattn(self.norm1(q), ctx, ctx_pad, rope_pos_q=rope_q, rope_pos_k=rope_ctx))
        q = q + self.drop(self.mlp(self.norm2(q)))
        return q


class CrossAttentionPredictor(nn.Module):
    """
    NEW(1/B): "정석 I-JEPA" 스타일 predictor:
      - student context encoder 출력(ctx)을 키/밸류로 사용
      - target queries (learned token + coord emb)가 ctx에 cross-attend해서 target latent 예측
    """
    def __init__(self, cfg: EEGModelConfig):
        super().__init__()
        d_model = cfg.d_model
        depth = getattr(cfg, "predictor_layers", 2)

        self.query_token = nn.Parameter(torch.zeros(d_model))
        nn.init.normal_(self.query_token, std=0.02)

        self.blocks = nn.ModuleList([CrossAttnPredictorBlock(cfg) for _ in range(depth)])
        self.norm = make_norm(cfg.norm_type, cfg.d_model, eps=1e-6)

    def forward(
        self,
        ctx: torch.Tensor,            # (B,Lc,D)
        ctx_pad: torch.Tensor,        # (B,Lc) True=PAD
        rope_ctx: torch.Tensor,       # (B,Lc) or (Lc,)
        tgt_coord_emb: torch.Tensor,  # (B,Lt,D)
        tgt_pad: torch.Tensor,        # (B,Lt) True=PAD
        rope_tgt: torch.Tensor,       # (B,Lt) or (Lt,)
        return_hidden: bool = False,   # NEW
    ) -> torch.Tensor:
        # build target queries
        q = self.query_token[None, None, :].to(tgt_coord_emb.dtype) + tgt_coord_emb
        q = q.masked_fill(tgt_pad[..., None], 0.0)

        hidden_states = []
        for blk in self.blocks:
            q = blk(q, ctx, ctx_pad, rope_q=rope_tgt, rope_ctx=rope_ctx)
            if return_hidden:
                hidden_states.append(q)

        q = self.norm(q)
        q = q.masked_fill(tgt_pad[..., None], 0.0)
        
        if return_hidden:
            return q, hidden_states
        return q
