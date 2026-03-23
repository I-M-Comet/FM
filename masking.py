
# eeg_fm/masking.py
from __future__ import annotations

from typing import Dict, Tuple

import torch


def _randint(low: int, high: int, device=None) -> int:
    return int(torch.randint(low, high, (1,), device=device).item())


@torch.no_grad()
def _random_nonneg_composition(total: int, parts: int, device: torch.device) -> torch.Tensor:
    """
    Return nonnegative integer vector g of length=parts with sum(g)=total.
    Simple multinomial-by-scatter (not uniform over compositions, but 충분히 랜덤).
    """
    g = torch.zeros((parts,), dtype=torch.long, device=device)
    if total <= 0 or parts <= 0:
        return g
    idx = torch.randint(0, parts, (total,), device=device)
    g.scatter_add_(0, idx, torch.ones((total,), dtype=torch.long, device=device))
    return g


@torch.no_grad()
def _random_partition_with_min(total: int, parts: int, min_each: int, device: torch.device) -> torch.Tensor:
    """
    Return positive-ish partition of 'total' into 'parts' integers each >= min_each.
    Caller must ensure total >= parts*min_each.
    """
    out = torch.full((parts,), int(min_each), dtype=torch.long, device=device)
    rem = total - parts * min_each
    if rem > 0:
        idx = torch.randint(0, parts, (rem,), device=device)
        out.scatter_add_(0, idx, torch.ones((rem,), dtype=torch.long, device=device))
    return out


@torch.no_grad()
def sample_time_mask(
    n_patches: torch.Tensor,      # (B,) 실제 P_t (각 샘플별)
    C: int,
    P_t_max: int,                 # batch 내 최대 P_t (텐서 shape 용)
    ratio_min: float,
    ratio_max: float,
    device: torch.device,
    style: int = 0,               # 0: "single", 1: "multi", 2: "ssp"
    num_blocks: int = 4,          # multi일 때 "masked blocks" 수
    min_block_patches: int = 2,   # multi일 때 각 block 최소 길이(초 단위 패치면 2초)
    ssp_keep_blocks: int = 3,     # ssp일 때 "keep blocks" 수
    ssp_min_keep_patches: int = 2,# ssp keep 블록 최소 길이
    short_P_threshold: int = 16,   # P_t가 이보다 작으면 multi에서 single로 폴백 (너무 짧은 시퀀스에서 multi가 trivial해지는 걸 방지)
) -> torch.Tensor:
    """
    Returns mask (B,C,P_t_max), True=masked(target), False=visible(context candidate).

    - single: 한 개 연속 time block을 마스크
    - multi : 총 마스크 길이(frac*P)를 여러 개 block으로 쪼개서(서로 disjoint) 마스크
    - ssp   : 전체를 마스크(True)로 두고, 여러 개 keep block만 False로 뚫음(=SSP)
    """
    B = int(n_patches.shape[0])

    if style == 2:
        mask = torch.ones((B, C, P_t_max), dtype=torch.bool, device=device)
    else:
        mask = torch.zeros((B, C, P_t_max), dtype=torch.bool, device=device)

    for b in range(B):
        P = int(n_patches[b].item())
        if P <= 0:
            continue

        frac = float(torch.empty((), device=device).uniform_(ratio_min, ratio_max).item())

        if style == 0:
            L = max(1, int(round(frac * P)))
            L = min(L, P)
            s = _randint(0, P - L + 1, device=device)
            mask[b, :, s : s + L] = True

        elif style == 1:
            if P <= short_P_threshold:
                # fallback to single-block time mask
                L = max(1, int(round(frac * P)))
                if P > 1:
                    L = min(L, P - 1)
                L = min(L, P)
                s = _randint(0, P - L + 1, device=device)
                mask[b, :, s:s+L] = True
                continue
            # 총 masked 길이
            L_total = max(1, int(round(frac * P)))
            # context가 완전히 비면 학습이 깨질 수 있으니 1 patch는 남기기
            if P > 1:
                L_total = min(L_total, P - 1)
            L_total = min(L_total, P)

            # 너무 짧은 시퀀스에서 block이 쪼개져서 trivial해지는 걸 방지
            min_len = int(min_block_patches)
            min_len = max(1, min_len)
            min_len = min(min_len, L_total)  # L_total보다 클 수 없음

            # 가능한 block 수(각 block >= min_len)
            max_blocks = max(1, L_total // min_len)
            n = min(int(num_blocks), max_blocks)
            n = max(1, min(n, L_total))

            # L_total을 n개 block으로 partition (각 block >= min_len)
            # total >= n*min_len은 위에서 n을 잡을 때 보장됨
            lengths = _random_partition_with_min(L_total, n, min_len, device=device)  # (n,)

            # gaps는 (n+1)개: gap + block + gap + block + ... + gap
            gap_total = P - int(lengths.sum().item())  # = P - L_total
            gaps = _random_nonneg_composition(gap_total, n + 1, device=device)

            pos = int(gaps[0].item())
            for i in range(n):
                li = int(lengths[i].item())
                if li > 0:
                    mask[b, :, pos : pos + li] = True
                pos = pos + li + int(gaps[i + 1].item())

        elif style == 2:
            # SSP: mask ratio=frac 만큼 마스크(True), keep blocks만 False로 뚫음
            num_masked = int(round(frac * P))
            num_masked = min(max(0, num_masked), P)
            num_kept = P - num_masked

            if num_kept <= 0:
                # 전부 마스크(True) 상태 유지
                continue
            if num_masked <= 0:
                # 전부 keep
                mask[b, :, :P] = False
                continue

            min_keep = int(ssp_min_keep_patches)
            min_keep = max(1, min_keep)
            min_keep = min(min_keep, num_kept)  # num_kept보다 클 수 없음

            # ✅ 핵심: min_keep를 만족할 수 있는 keep 블록 수로 제한
            feasible_blocks = max(1, num_kept // min_keep)
            n = min(int(ssp_keep_blocks), feasible_blocks)
            n = max(1, min(n, num_kept))  # 안전장치
            if P <= short_P_threshold:
                n = 1

            kept_lengths = _random_partition_with_min(num_kept, n, min_keep, device=device)  # (n,)
            gaps = _random_nonneg_composition(num_masked, n + 1, device=device)              # (n+1,)

            pos = int(gaps[0].item())
            for i in range(n):
                li = int(kept_lengths[i].item())
                if li > 0:
                    mask[b, :, pos : pos + li] = False
                pos = pos + li + int(gaps[i + 1].item())
        
        elif style == 3:
            #SSP origin?
            num_blocks = max(1, P // 10) # 10 sec
            num_masked = int(round(frac * P))
            num_masked = min(max(0, num_masked), P)
            num_kept = P - num_masked
            
            # 예외 처리: 전부 마스킹되거나, 하나도 마스킹 안 되는 경우
            if num_kept <= 0:
                mask[b, :, :] = True
                continue
            if num_masked <= 0:
                continue
            
            # 보존할 블록 개수 조정 (보존할 패치 수보다 블록 수가 많을 순 없음)
            actual_blocks = min(num_blocks, num_kept)
            block_len = num_kept // actual_blocks

            kept_lengths = torch.full((actual_blocks,), block_len, dtype=torch.long, device=device)
            kept_lengths[-1] += num_kept - (block_len * actual_blocks)  # 나머지 패치는 첫 번째 블록에 몰아서 추가
            gaps = _random_nonneg_composition(num_masked, actual_blocks + 1, device=device) 
            mask[b, :, :] = True
            pos = int(gaps[0].item())
            for i in range(actual_blocks):
                li = int(kept_lengths[i].item())
                if li > 0:
                    mask[b, :, pos : pos + li] = False
                pos = pos + li + int(gaps[i + 1].item())

        else:
            raise ValueError(f"unknown time mask style: {style}")

    return mask


@torch.no_grad()
def sample_spatial_block_mask(
    coords: torch.Tensor,      # (B, C, 3)
    valid_chan: torch.Tensor,  # (B, C) bool
    P_t: int,
    ratio_min: float,
    ratio_max: float,
    device: torch.device,
) -> torch.Tensor:
    B, C, _ = coords.shape
    mask = torch.zeros((B, C, P_t), dtype=torch.bool, device=device)

    for b in range(B):
        valid_idx = torch.nonzero(valid_chan[b], as_tuple=False).squeeze(-1)
        if valid_idx.numel() == 0:
            continue
        Cb = int(valid_idx.numel())
        frac = float(torch.empty((), device=device).uniform_(ratio_min, ratio_max).item())
        k = max(1, int(round(frac * Cb)))
        k = min(k, Cb)

        center_i = valid_idx[_randint(0, Cb, device=device)]
        center = coords[b, center_i]

        d = torch.sum((coords[b] - center[None, :]) ** 2, dim=-1)
        d = d.masked_fill(~valid_chan[b], float("inf"))

        nn = torch.topk(d, k, largest=False).indices
        mask[b, nn, :] = True

    return mask


@torch.no_grad()
def sample_jepa_target_mask(
    coords: torch.Tensor,         # (B, C, 3)
    n_channels: torch.Tensor,     # (B,) int
    n_patches: torch.Tensor,      # (B,) int (P_t)
    mask_time_prob: float,
    mask_spatial_prob: float,
    time_ratio_range: Tuple[float, float],
    spatial_ratio_range: Tuple[float, float],
    time_mask_style: int = 0,     # 0: "single", 1: "multi", 2: "ssp"
    time_mask_num_blocks: int = 4,       # multi일 때
    time_mask_min_block_patches: int = 2,
    time_ssp_keep_blocks: int = 3,       # ssp일 때
    time_ssp_min_keep_patches: int = 2,
) -> torch.Tensor:
    device = coords.device
    B, C_max, _ = coords.shape
    P_t_max = int(n_patches.max().item())

    chan_ids = torch.arange(C_max, device=device)[None, :]
    valid_chan = chan_ids < n_channels[:, None]
    time_ids = torch.arange(P_t_max, device=device)[None, :]
    valid_time = time_ids < n_patches[:, None]
    valid_tok = valid_chan[:, :, None] & valid_time[:, None, :]

    target = torch.zeros((B, C_max, P_t_max), dtype=torch.bool, device=device)

    use_time = torch.rand((B,), device=device) < mask_time_prob
    use_spat = torch.rand((B,), device=device) < mask_spatial_prob
    none = ~(use_time | use_spat)
    if none.any():
        use_time = use_time | none

    if use_time.any():
        tmask = sample_time_mask(
            n_patches=n_patches,
            C=C_max,
            P_t_max=P_t_max,
            ratio_min=time_ratio_range[0],
            ratio_max=time_ratio_range[1],
            device=device,
            style=time_mask_style,
            num_blocks=time_mask_num_blocks,
            min_block_patches=time_mask_min_block_patches,
            ssp_keep_blocks=time_ssp_keep_blocks,
            ssp_min_keep_patches=time_ssp_min_keep_patches,
        )
        target = target | (tmask & use_time[:, None, None])

    if use_spat.any():
        smask = sample_spatial_block_mask(
            coords=coords,
            valid_chan=valid_chan,
            P_t=P_t_max,
            ratio_min=spatial_ratio_range[0],
            ratio_max=spatial_ratio_range[1],
            device=device,
        )
        target = target | (smask & use_spat[:, None, None])

    target = target & valid_tok
    return target


@torch.no_grad()
def _random_nonneg_composition_cpu(total: int, parts: int) -> torch.Tensor:
    g = torch.zeros((parts,), dtype=torch.long)
    if total <= 0 or parts <= 0:
        return g
    idx = torch.randint(0, parts, (total,), dtype=torch.long)
    g.scatter_add_(0, idx, torch.ones((total,), dtype=torch.long))
    return g


@torch.no_grad()
def sample_time_mask_style3_same_shape_cpu(
    B: int,
    C: int,
    P: int,
    ratio_min: float,
    ratio_max: float,
) -> torch.Tensor:
    """
    Train fast path for ShapeBatcher output:
      - all samples share the same (C, P)
      - style == 3 only
      - CPU only (worker-side)

    Returns:
      mask: (B, C, P), True=target(masked)
    """
    mask = torch.zeros((B, C, P), dtype=torch.bool)
    if B <= 0 or C <= 0 or P <= 0:
        return mask

    frac = torch.empty((B,), dtype=torch.float32).uniform_(ratio_min, ratio_max)
    num_masked = torch.round(frac * P).to(torch.long).clamp_(0, P)
    num_kept = P - num_masked
    num_blocks = max(1, P // 10)  # keep same policy as your current style-3

    for b in range(B):
        nk = int(num_kept[b])
        nm = int(num_masked[b])

        if nk <= 0:
            mask[b].fill_(True)
            continue
        if nm <= 0:
            continue

        actual_blocks = min(num_blocks, nk)
        base = nk // actual_blocks
        rem = nk - base * actual_blocks

        kept_lengths = torch.full((actual_blocks,), base, dtype=torch.long)
        kept_lengths[-1] += rem
        gaps = _random_nonneg_composition_cpu(nm, actual_blocks + 1)

        row = torch.ones((P,), dtype=torch.bool)
        pos = int(gaps[0])
        for i in range(actual_blocks):
            li = int(kept_lengths[i])
            row[pos:pos + li] = False
            pos += li + int(gaps[i + 1])

        mask[b] = row.unsqueeze(0).expand(C, P)

    return mask


@torch.no_grad()
def sample_spatial_block_mask_same_shape_cpu(
    coords: torch.Tensor,   # (B, C, 3), CPU
    P_t: int,
    ratio_min: float,
    ratio_max: float,
) -> torch.Tensor:
    """
    Same-shape CPU fast path.
    All channels are valid because ShapeBatcher batches exact (C, P).
    """
    B, C, _ = coords.shape
    base = torch.zeros((B, C), dtype=torch.bool)
    if B <= 0 or C <= 0 or P_t <= 0:
        return base.unsqueeze(-1).expand(B, C, max(P_t, 1))

    frac = torch.empty((B,), dtype=torch.float32).uniform_(ratio_min, ratio_max)
    k = torch.round(frac * C).to(torch.long).clamp_(1, C)
    centers = torch.randint(0, C, (B,), dtype=torch.long)

    for b in range(B):
        center = coords[b, centers[b]]
        d = torch.sum((coords[b] - center[None, :]) ** 2, dim=-1)
        nn = torch.topk(d, int(k[b]), largest=False).indices
        base[b, nn] = True

    return base.unsqueeze(-1).expand(B, C, P_t)


@torch.no_grad()
def sample_jepa_target_mask_same_shape_style3_cpu(
    coords: torch.Tensor,   # (B, C, 3), CPU
    P_t: int,
    mask_time_prob: float,
    mask_spatial_prob: float,
    time_ratio_range: Tuple[float, float],
    spatial_ratio_range: Tuple[float, float],
    dilate_time: int = 0,
    **kwargs
) -> torch.Tensor:
    """
    Fast worker-side path for current train setup:
      - ShapeBatcher guarantees same (C, P)
      - style-3 only
      - no valid_chan / valid_time construction needed
      - no device roundtrip
    """
    assert coords.device.type == "cpu", "worker fast path is intended for CPU tensors"

    B, C, _ = coords.shape
    target = torch.zeros((B, C, P_t), dtype=torch.bool)
    if B <= 0 or C <= 0 or P_t <= 0:
        return target

    use_time = torch.rand((B,), dtype=torch.float32) < float(mask_time_prob)
    use_spat = torch.rand((B,), dtype=torch.float32) < float(mask_spatial_prob)
    none = ~(use_time | use_spat)
    use_time[none] = True

    if bool(use_time.any()):
        tmask = sample_time_mask_style3_same_shape_cpu(
            B=B,
            C=C,
            P=P_t,
            ratio_min=float(time_ratio_range[0]),
            ratio_max=float(time_ratio_range[1]),
        )
        target |= tmask & use_time[:, None, None]

    if bool(use_spat.any()):
        smask = sample_spatial_block_mask_same_shape_cpu(
            coords=coords,
            P_t=P_t,
            ratio_min=float(spatial_ratio_range[0]),
            ratio_max=float(spatial_ratio_range[1]),
        )
        target |= smask & use_spat[:, None, None]

    if int(dilate_time) > 0:
        target = dilate_time_mask(target, dilation=int(dilate_time))

    return target

@torch.no_grad()
def physio_band_bin_masks(bin_centers_hz: torch.Tensor) -> Dict[str, torch.Tensor]:
    def m(lo, hi):
        return (bin_centers_hz >= lo) & (bin_centers_hz < hi)

    return {
        "delta": m(0.5, 4.0),
        "theta": m(4.0, 8.0),
        "alpha": m(8.0, 12.0),
        "beta":  m(13.0, 30.0),
        "gamma": m(30.0, 45.0),
    }


@torch.no_grad()
def sample_freq_bin_mask(
    B: int,
    K: int,
    bin_centers_hz: torch.Tensor,  # (K,)
    physio_prob: float,
    num_bands_min: int,
    num_bands_max: int,
    random_width_min: float,
    random_width_max: float,
    device: torch.device,
) -> torch.Tensor:
    masks = torch.zeros((B, K), dtype=torch.bool, device=device)
    band_masks = physio_band_bin_masks(bin_centers_hz.to(device))
    band_names = list(band_masks.keys())

    for b in range(B):
        if float(torch.rand((), device=device).item()) < physio_prob:
            nb = _randint(num_bands_min, num_bands_max + 1, device=device)
            chosen = torch.randperm(len(band_names), device=device)[:nb].tolist()
            mb = torch.zeros((K,), dtype=torch.bool, device=device)
            for idx in chosen:
                mb = mb | band_masks[band_names[idx]]
            masks[b] = mb
        else:
            wmin = max(1, int(round(random_width_min * K)))
            wmax = max(wmin, int(round(random_width_max * K)))
            w = _randint(wmin, wmax + 1, device=device)
            s = _randint(0, K - w + 1, device=device)
            masks[b, s : s + w] = True

    return masks


@torch.no_grad()
def dilate_time_mask(mask: torch.Tensor, dilation: int) -> torch.Tensor:
    """Dilate a (B,C,P) mask along the time-patch axis by +/- dilation."""
    if dilation <= 0:
        return mask
    B, C, P = mask.shape
    out = mask.clone()
    for d in range(1, dilation + 1):
        out[..., d:] |= mask[..., :-d]
        out[..., :-d] |= mask[..., d:]
    return out
