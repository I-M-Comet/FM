
# eeg_fm/masking.py
from __future__ import annotations

from typing import Dict, Tuple

import torch


def _randint(low: int, high: int, device=None) -> int:
    return int(torch.randint(low, high, (1,), device=device).item())


@torch.no_grad()
def sample_time_block_mask(
    B: int,
    C: int,
    P_t: int,
    ratio_min: float,
    ratio_max: float,
    device: torch.device,
    is_SSP: bool = False,
) -> torch.Tensor:
    mask = torch.zeros((B, C, P_t), dtype=torch.bool, device=device)
    for b in range(B):
        if not is_SSP:
            frac = float(torch.empty((), device=device).uniform_(ratio_min, ratio_max).item())
            L = max(1, int(round(frac * P_t)))
            L = min(L, P_t)
            s = _randint(0, P_t - L + 1, device=device)
            mask[b, :, s : s + L] = True
        else:
# ---------------- [신규] SSP (데이터 길이에 비례한 다중 서브시퀀스 보존) ----------------
            # P_t(총 패치 수 = 총 초 수)를 sec_per_block으로 나누어 알맞은 블록 개수를 자동 계산합니다.
            # 예: 10초 -> 1개, 30초 -> 3개, 60초 -> 6개
            num_blocks = max(1, P_t // 10) # 10 sec
            
            num_masked = int(round(frac * P_t))
            num_masked = min(max(0, num_masked), P_t)
            num_kept = P_t - num_masked
            
            # 예외 처리: 전부 마스킹되거나, 하나도 마스킹 안 되는 경우
            if num_kept <= 0:
                mask[b, :, :] = True
                continue
            if num_masked <= 0:
                continue
            
            # 보존할 블록 개수 조정 (보존할 패치 수보다 블록 수가 많을 순 없음)
            actual_blocks = min(num_blocks, num_kept)
            block_len = num_kept // actual_blocks
            
            # 베이스를 모두 마스킹(True)으로 덮음
            mask[b, :, :] = True
            
            # 마스킹될 패치들을 보존할 블록 사이사이 간격에 무작위 분배
            rand_spaces = torch.rand(actual_blocks + 1, device=device)
            spaces = (rand_spaces / rand_spaces.sum() * num_masked).int()
            
            # 캐스팅 오차로 남는 자투리 패치는 랜덤한 공간에 1씩 추가 분배
            remainder = num_masked - spaces.sum().item()
            if remainder > 0:
                indices = torch.randperm(actual_blocks + 1, device=device)[:remainder]
                spaces[indices] += 1
            
            curr_idx = 0
            for i in range(actual_blocks):
                curr_idx += spaces[i].item()
                
                if i == actual_blocks - 1:
                    curr_block_len = num_kept - (block_len * (actual_blocks - 1))
                else:
                    curr_block_len = block_len
                    
                # 보존할 영역 마스킹 해제 (False)
                mask[b, :, curr_idx : curr_idx + curr_block_len] = False
                curr_idx += curr_block_len
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
    is_SSP: bool = False,
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
        tmask = sample_time_block_mask(
            B=B, C=C_max, P_t=P_t_max,
            ratio_min=time_ratio_range[0], ratio_max=time_ratio_range[1],
            device=device,
            is_SSP=is_SSP,
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
