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
    def __init__(self, d_model: int, n_heads: int, attn_dropout: float, rope_theta: float, rotary_pct: float, qk_norm: str = "off", qk_norm_eps: float = 1e-6):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.attn_dropout = float(attn_dropout)

        self.qk_norm = str(qk_norm).lower()
        self.qk_norm_eps = float(qk_norm_eps)

        rotary_dim = int(self.head_dim * rotary_pct)
        rotary_dim = rotary_dim - (rotary_dim % 2)
        self.rotary_dim = max(0, rotary_dim)
        self.rope_theta = float(rope_theta)

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=True)
        self.out = nn.Linear(d_model, d_model, bias=True)

        self._rope_cache = None  # (cos, sin, max_pos, dtype, device)

    def _get_rope(
        self,
        rope_pos: torch.Tensor,   # (N,) or (B,N)
        dtype: torch.dtype,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        returns cos,sin indexed by positions:
          - if rope_pos is (N,), returns (N,half)
          - if rope_pos is (B,N), returns (B,N,half)
        """
        if self.rotary_dim == 0:
            raise RuntimeError("rotary_dim is 0 but _get_rope called")

        max_pos = int(rope_pos.max().item()) + 1

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
        attn_bias: Optional[torch.Tensor] = None,
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
            cos, sin = self._get_rope(rope_pos, dtype=q.dtype, device=q.device)  # (N,half) or (B,N,half)
            q_rot, q_pass = q[..., : self.rotary_dim], q[..., self.rotary_dim :]
            k_rot, k_pass = k[..., : self.rotary_dim], k[..., self.rotary_dim :]
            q_rot = apply_rope(q_rot, cos, sin)
            k_rot = apply_rope(k_rot, cos, sin)
            q = torch.cat([q_rot, q_pass], dim=-1)
            k = torch.cat([k_rot, k_pass], dim=-1)

        # Optional QK normalization (attention stability)
        if self.qk_norm != "off":
            eps = self.qk_norm_eps
            if self.qk_norm == "l2":
                q = q / (q.norm(dim=-1, keepdim=True) + eps)
                k = k / (k.norm(dim=-1, keepdim=True) + eps)
                # compensate SDP's internal 1/sqrt(d) scaling (so logits stay O(1))
                scale = math.sqrt(self.head_dim)
                q = q * scale
                k = k * scale
            elif self.qk_norm in ("rms", "rmsnorm"):
                q = q * torch.rsqrt(q.pow(2).mean(dim=-1, keepdim=True) + eps)
                k = k * torch.rsqrt(k.pow(2).mean(dim=-1, keepdim=True) + eps)
            elif self.qk_norm in ("layernorm", "ln"):
                q = F.layer_norm(q, (self.head_dim,), eps=eps)
                k = F.layer_norm(k, (self.head_dim,), eps=eps)
            else:
                raise ValueError(f"Unknown attn_qk_norm: {self.qk_norm}")
            
        attn_mask = None
        if attn_bias is None:
            # key padding only (efficient)
            if padding_mask is not None:
                attn_mask = (~padding_mask)[:, None, None, :]  # (B,1,1,N)
        else:
            # additive bias mask (float)
            if attn_bias.dim() == 3:
                attn_mask = attn_bias[:, None, :, :]
            elif attn_bias.dim() == 4:
                attn_mask = attn_bias
            else:
                raise ValueError(f"attn_bias must have dim 3 or 4, got {attn_bias.shape}")
            attn_mask = attn_mask.to(dtype=q.dtype)
            if padding_mask is not None:
                attn_mask = attn_mask.masked_fill(padding_mask[:, None, None, :], float("-inf"))

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.attn_dropout if self.training else 0.0,
            is_causal=False,
        )

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

        rotary_dim = int(self.head_dim * rotary_pct)
        rotary_dim = rotary_dim - (rotary_dim % 2)
        self.rotary_dim = max(0, rotary_dim)
        self.rope_theta = float(rope_theta)

        self.q = nn.Linear(d_model, d_model, bias=True)
        self.kv = nn.Linear(d_model, 2 * d_model, bias=True)
        self.out = nn.Linear(d_model, d_model, bias=True)

        self._rope_cache = None  # (cos, sin, max_pos, dtype, device)

    def _get_rope(self, rope_pos: torch.Tensor, dtype: torch.dtype, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        # shared logic with self-attn; keep minimal
        if self.rotary_dim == 0:
            raise RuntimeError("rotary_dim is 0 but _get_rope called")

        max_pos = int(rope_pos.max().item()) + 1

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
            cos_q, sin_q = self._get_rope(rope_pos_q, dtype=q.dtype, device=q.device)
            cos_k, sin_k = self._get_rope(rope_pos_k, dtype=q.dtype, device=q.device)

            q_rot, q_pass = q[..., : self.rotary_dim], q[..., self.rotary_dim :]
            k_rot, k_pass = k[..., : self.rotary_dim], k[..., self.rotary_dim :]
            q_rot = apply_rope(q_rot, cos_q, sin_q)
            k_rot = apply_rope(k_rot, cos_k, sin_k)
            q = torch.cat([q_rot, q_pass], dim=-1)
            k = torch.cat([k_rot, k_pass], dim=-1)

        # Optional QK normalization (attention stability)
        if self.qk_norm != "off":
            eps = self.qk_norm_eps
            if self.qk_norm == "l2":
                q = q / (q.norm(dim=-1, keepdim=True) + eps)
                k = k / (k.norm(dim=-1, keepdim=True) + eps)
                scale = math.sqrt(self.head_dim)
                q = q * scale
                k = k * scale
            elif self.qk_norm in ("rms", "rmsnorm"):
                q = q * torch.rsqrt(q.pow(2).mean(dim=-1, keepdim=True) + eps)
                k = k * torch.rsqrt(k.pow(2).mean(dim=-1, keepdim=True) + eps)
            elif self.qk_norm in ("layernorm", "ln"):
                q = F.layer_norm(q, (self.head_dim,), eps=eps)
                k = F.layer_norm(k, (self.head_dim,), eps=eps)
            else:
                raise ValueError(f"Unknown attn_qk_norm: {self.qk_norm}")
            
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

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, affine: bool = True):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim)) if affine else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., dim)
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        y = x * rms
        if self.weight is not None:
            y = y * self.weight
        return y

def make_norm(norm_type: str, dim: int, eps: float = 1e-6) -> nn.Module:
    norm_type = norm_type.lower()
    if norm_type in ("layernorm", "ln"):
        return nn.LayerNorm(dim, eps=eps)
    if norm_type in ("rmsnorm", "rms"):
        return RMSNorm(dim, eps=eps, affine=True)
    raise ValueError(f"Unknown norm_type: {norm_type}")

class LayerScale(nn.Module):
    def __init__(self, dim: int, init_value: float = 1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim) * init_value)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,N,dim)
        return x * self.gamma

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


class TransformerBlock(nn.Module):
    def __init__(self, cfg: EEGModelConfig):
        super().__init__()
        self.norm1 = make_norm(cfg.norm_type, cfg.d_model, eps=1e-6)
        self.attn = MultiheadSelfAttentionRoPE(
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            attn_dropout=cfg.attn_dropout,
            rope_theta=cfg.rope_theta,
            rotary_pct=cfg.rotary_pct,
            qk_norm=getattr(cfg, "attn_qk_norm", "off"),
            qk_norm_eps=getattr(cfg, "attn_qk_norm_eps", 1e-6),
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

    def forward(self, x: torch.Tensor, padding_mask: Optional[torch.Tensor], rope_pos: torch.Tensor) -> torch.Tensor:
        y = self.attn(self.norm1(x), padding_mask=padding_mask, rope_pos=rope_pos)
        if self.ls1 is not None:
            y = self.ls1(y)
        x = x + self.dropout(y)
        y = self.mlp(self.norm2(x))
        if self.ls2 is not None:
            y = self.ls2(y)
        x = x + self.dropout(y)
        return x


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

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor],
        rope_pos: torch.Tensor,
        chan_idx: Optional[torch.Tensor],
        spatial_bias_cc: Optional[torch.Tensor],
    ) -> torch.Tensor:
        # spatial bias on channels -> gather to token-token
        attn_bias = None
        if self.use_spatial_bias and (spatial_bias_cc is not None) and (chan_idx is not None):
            # token channel indices (PAD already clamped in embed_from_indices)
            c = chan_idx
            B, N = c.shape
            b = torch.arange(B, device=c.device)[:, None, None]
            bias_cc = spatial_bias_cc.to(dtype=x.dtype, device=x.device)
            attn_bias = bias_cc[b, c[:, :, None], c[:, None, :]]  # (B,N,N)

        y = self.attn(self.norm1(x), padding_mask=padding_mask, rope_pos=rope_pos, attn_bias=attn_bias)
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

    @staticmethod
    def _scatter_to_grid(
        x: torch.Tensor,            # (B,L,D)
        pad: torch.Tensor,          # (B,L) True=PAD
        c: torch.Tensor,            # (B,L) channel indices (safe)
        t: torch.Tensor,            # (B,L) time indices (safe)
        C: int,
        P: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Scatter packed tokens to a dense (B,C,P,D) grid.

        Returns:
          grid: (B,C,P,D)
          grid_pad: (B,C,P) True=missing
        """
        B, L, D = x.shape
        device = x.device

        grid = x.new_zeros((B, C, P, D))
        grid_pad = torch.ones((B, C, P), dtype=torch.bool, device=device)

        valid = ~pad
        if valid.any():
            b_idx = torch.arange(B, device=device)[:, None].expand(B, L)
            b = b_idx[valid]
            cc = c[valid]
            tt = t[valid]
            grid[b, cc, tt] = x[valid]
            grid_pad[b, cc, tt] = False

        return grid, grid_pad

    @staticmethod
    def _gather_from_grid(
        grid: torch.Tensor,         # (B,C,P,D)
        pad: torch.Tensor,          # (B,L)
        c: torch.Tensor,            # (B,L)
        t: torch.Tensor,            # (B,L)
    ) -> torch.Tensor:
        B, L = c.shape
        device = c.device
        b_idx = torch.arange(B, device=device)[:, None].expand(B, L)
        out = grid[b_idx, c, t]
        return out.masked_fill(pad[..., None], 0.0)

    def _temporal_pass(
        self,
        x: torch.Tensor,            # (B,L,D)
        pad: torch.Tensor,          # (B,L)
        t_idx: torch.Tensor,        # (B,L)
        c_idx: torch.Tensor,        # (B,L)
        C: int,
        P: int,
    ) -> torch.Tensor:
        # dense grid
        grid, grid_pad = self._scatter_to_grid(x, pad, c_idx, t_idx, C=C, P=P)

        # temporal: (B*C, P, D)
        x_t = grid.reshape(-1, P, grid.shape[-1])
        pad_t = grid_pad.reshape(-1, P)
        nonempty = (~pad_t).any(dim=1)
        if nonempty.any():
            x_sel = x_t[nonempty]
            pad_sel = pad_t[nonempty]
            rope = torch.arange(P, device=x.device, dtype=torch.long)
            y_sel = self.attn_t(x_sel, padding_mask=pad_sel, rope_pos=rope, attn_bias=None)
            y_t = x_t.new_zeros(x_t.shape, dtype=y_sel.dtype)
            y_t[nonempty] = y_sel
        else:
            y_t = x_t.new_zeros(x_t.shape, dtype=x_t.dtype)

        grid_out = y_t.reshape(grid.shape)
        return self._gather_from_grid(grid_out, pad, c_idx, t_idx)

    def _spatial_pass(
        self,
        x: torch.Tensor,            # (B,L,D)
        pad: torch.Tensor,          # (B,L)
        t_idx: torch.Tensor,        # (B,L)
        c_idx: torch.Tensor,        # (B,L)
        spatial_bias_cc: Optional[torch.Tensor],  # (B,C,C)
        C: int,
        P: int,
    ) -> torch.Tensor:
        # dense grid
        grid, grid_pad = self._scatter_to_grid(x, pad, c_idx, t_idx, C=C, P=P)

        # spatial: group by time -> (B*P, C, D)
        grid_tp = grid.permute(0, 2, 1, 3).contiguous()  # (B,P,C,D)
        pad_tp = grid_pad.permute(0, 2, 1).contiguous()  # (B,P,C)
        x_s = grid_tp.reshape(-1, C, grid.shape[-1])
        pad_s = pad_tp.reshape(-1, C)
        nonempty = (~pad_s).any(dim=1)

        if nonempty.any():
            x_sel = x_s[nonempty]
            pad_sel = pad_s[nonempty]
            bias_sel = None
            if spatial_bias_cc is not None:
                # We avoid materializing (B*P,C,C). Map each non-empty (b,p) group back to b.
                # x_s is flattened from (B,P,C,D) -> (B*P,C,D), so group_index // P == b.
                group_idx = torch.nonzero(nonempty, as_tuple=False).squeeze(-1)  # (n_sel,)
                b_idx = group_idx // P
                bias_sel = spatial_bias_cc.to(dtype=x.dtype, device=x.device)[b_idx]  # (n_sel,C,C)
            y_sel = self.attn_s(x_sel, padding_mask=pad_sel, rope_pos=None, attn_bias=bias_sel)
            y_s = x_s.new_zeros(x_s.shape, dtype=y_sel.dtype)
            y_s[nonempty] = y_sel
        else:
            y_s = x_s.new_zeros(x_s.shape, dtype=x_s.dtype)

        grid_tp_out = y_s.reshape(grid_tp.shape)  # (B,P,C,D)
        grid_out = grid_tp_out.permute(0, 2, 1, 3).contiguous()  # (B,C,P,D)
        return self._gather_from_grid(grid_out, pad, c_idx, t_idx)

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor],
        rope_pos: torch.Tensor,
        chan_idx: Optional[torch.Tensor],
        spatial_bias_cc: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if padding_mask is None:
            # assume all valid
            padding_mask = torch.zeros(x.shape[:2], dtype=torch.bool, device=x.device)
        if chan_idx is None:
            raise ValueError("DividedSpatiotemporalBlock requires chan_idx")

        # infer grid sizes
        if spatial_bias_cc is not None:
            C = int(spatial_bias_cc.shape[1])
        else:
            C = int(chan_idx.max().item()) + 1
        P = int(rope_pos.max().item()) + 1

        # temporal pass
        y = self._temporal_pass(self.norm_t(x), padding_mask, rope_pos, chan_idx, C=C, P=P)
        if self.ls_t is not None:
            y = self.ls_t(y)
        x = x + self.dropout(y)

        # spatial pass
        y = self._spatial_pass(self.norm_s(x), padding_mask, rope_pos, chan_idx, spatial_bias_cc=spatial_bias_cc, C=C, P=P)
        if self.ls_s is not None:
            y = self.ls_s(y)
        x = x + self.dropout(y)

        # MLP
        y = self.mlp(self.norm_m(x))
        if self.ls_m is not None:
            y = self.ls_m(y)
        x = x + self.dropout(y)
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
        self.film = FiLMFusion(
            freq_dim=self.freq_feat.freq_dim,
            d_model=cfg.d_model,
            hidden=cfg.film_hidden,
        )

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
        coords: torch.Tensor,         # (B,C,3)
        c_idx: torch.Tensor,          # (B,L) long, channel index (>=0 for valid)
        t_idx: torch.Tensor,          # (B,L) long, patch-time index (>=0 for valid)
        pad: torch.Tensor,            # (B,L) bool True=PAD
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
        f = self.freq_feat.forward_packed(patches)  # (B,L,F)

        # apply freq corruption (only intended for student)
        if freq_domain_drop is not None:
            # drop all freq dims (including scale) for dropped samples
            drop = freq_domain_drop.to(device=device, dtype=torch.bool)[:, None]  # (B,1)
            f = f.masked_fill(drop[..., None] & (~pad)[..., None], 0.0)

        if freq_mask_bins is not None:
            # only mask the first K bins; keep optional scale dim intact
            K = self.cfg.freq_bins
            band = freq_mask_bins.to(device=device, dtype=torch.bool)  # (B,K)
            band = band[:, None, :]  # (B,1,K)
            f_shape = f[..., :K].masked_fill(band & (~pad[..., None]), 0.0)
            if f.shape[-1] > K:
                f = torch.cat([f_shape, f[..., K:]], dim=-1)
            else:
                f = f_shape

        # FiLM fuse
        tok = self.film(e, f)
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
    ) -> torch.Tensor:
        # Precompute spatial bias once per forward (B,C,C). This avoids recomputing
        # the Legendre series in every encoder block.
        spatial_bias_cc = None
        if coords is not None:
            spatial_bias_cc = self.spatial_bias(coords)
            if spatial_bias_cc is not None:
                spatial_bias_cc = spatial_bias_cc.to(dtype=tokens.dtype, device=tokens.device)

        x = tokens
        for blk in self.blocks:
            x = blk(
                x,
                padding_mask=padding_mask,
                rope_pos=rope_pos,
                chan_idx=chan_idx,
                spatial_bias_cc=spatial_bias_cc,
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
    ) -> torch.Tensor:
        # build target queries
        q = self.query_token[None, None, :].to(tgt_coord_emb.dtype) + tgt_coord_emb
        q = q.masked_fill(tgt_pad[..., None], 0.0)

        for blk in self.blocks:
            q = blk(q, ctx, ctx_pad, rope_q=rope_tgt, rope_ctx=rope_ctx)

        q = self.norm(q)
        q = q.masked_fill(tgt_pad[..., None], 0.0)
        return q
