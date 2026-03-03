# eeg_fm/data.py
from __future__ import annotations

import glob
import hashlib
import io
import json
import numpy as np
import os
import random
import shutil
import torch

from contextlib import contextmanager
from torch.utils.data import IterableDataset
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

try:
    import webdataset as wds
except Exception:
    wds = None


EEG_KEY = "eeg.npy"
COORD_KEY = "coords.npy"
META_KEY = "meta.json"

@contextmanager
def _file_lock(lock_path: str):
    # Linux 전제. 없으면 락 없이 동작(최악에도 try/except로 안전하게)
    try:
        import fcntl
        os.makedirs(os.path.dirname(lock_path), exist_ok=True)
        with open(lock_path, "w") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            yield
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    except Exception:
        yield


def find_shards(data_root: str, shard_glob: str) -> List[str]:
    pat = os.path.join(data_root, shard_glob)
    shards = sorted(glob.glob(pat, recursive=True))
    return shards


def _as_torch(x: Any) -> torch.Tensor:
    """Convert decoded WebDataset field into a torch.Tensor.

    WebDataset decoding behavior differs by version. In some versions, `.npy` fields
    may remain as raw bytes. We handle:
      - torch.Tensor
      - np.ndarray
      - bytes/bytearray containing .npy (or .npz) payload
    """
    if isinstance(x, torch.Tensor):
        return x
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    if isinstance(x, (bytes, bytearray)):
        bio = io.BytesIO(x)
        arr = np.load(bio, allow_pickle=False)
        # Support .npz just in case (shouldn't happen in our pipeline).
        if isinstance(arr, np.lib.npyio.NpzFile):
            if len(arr.files) == 0:
                raise ValueError("Empty NPZ payload")
            arr = arr[arr.files[0]]
        return torch.from_numpy(arr)
    raise TypeError(type(x))


def decode_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    eeg = sample[EEG_KEY]
    coord = sample[COORD_KEY]
    meta = sample.get(META_KEY, {})
    if isinstance(meta, (bytes, bytearray)):
        meta = json.loads(meta.decode("utf-8"))

    # Preserve webdataset identifiers for traceability / multi-view pairing.
    # These keys are provided by webdataset tarfile_to_samples.
    if not isinstance(meta, dict):
        meta = {"meta": meta}
    else:
        # copy to avoid accidental shared-mutation across pipeline stages
        meta = dict(meta)
    if "__key__" in sample:
        meta["__key__"] = sample.get("__key__")
    if "__url__" in sample:
        meta["__url__"] = sample.get("__url__")

    eeg = _as_torch(eeg).to(torch.float16)     # stored float16
    coord = _as_torch(coord).to(torch.float32)
    coord = coord * 10 # (0.1m -> 1.0)
    # coord = coord - coord.mean(dim=0, keepdim=True)
    # coord = coord / (coord.norm(dim=-1).mean().clamp_min(1e-6))

    return {"eeg": eeg, "coord": coord, "meta": meta}

def _flatmap_stage(mapper):
    """Compatibility replacement for `webdataset.flatmap` (not present in some versions).

    `mapper` should return either:
      - a single sample dict
      - an iterable (e.g., list) of sample dicts
      - None (to drop the sample)
    The returned object is an iterator->iterator pipeline stage compatible with `wds.DataPipeline`.
    """
    def _iter(src):
        for ex in src:
            out = mapper(ex)
            if out is None:
                continue
            if isinstance(out, dict):
                yield out
            else:
                for y in out:
                    if y is None:
                        continue
                    yield y
    return _iter


def compute_num_patches(T: int, patch_samples: int, hop_samples: int) -> int:
    """Number of sliding-window patches.
    P = floor((T - patch)/hop) + 1. Requires T >= patch.
    """
    if T < patch_samples:
        return 0
    return int((T - patch_samples) // hop_samples + 1)


def maybe_random_window_crop_60s(
    ex: Dict[str, Any],
    crop_prob: float,
    crop_30_prob: float,
    align_to_seconds: bool = True,   # 추가: 1초 경계로 자를지
) -> Dict[str, Any]:
    if crop_prob <= 0:
        return ex

    eeg: torch.Tensor = ex["eeg"]
    meta: Dict[str, Any] = ex.get("meta", {})

    fs = int(meta.get("fs", 200)) or 200
    duration_sec = int(eeg.shape[-1] // fs)

    if duration_sec != 60:
        return ex
    if random.random() >= float(crop_prob):
        return ex

    crop_sec = 30 if (random.random() < float(crop_30_prob)) else 10
    crop_samples = int(crop_sec * fs)
    C, T = eeg.shape
    if T < crop_samples:
        return ex

    if align_to_seconds:
        start_sec = random.randint(0, 60 - crop_sec)
        start = start_sec * fs
    else:
        start = random.randint(0, T - crop_samples)
        start_sec = start / fs

    eeg2 = eeg[:, start : start + crop_samples].contiguous()

    # ✅ 기존 필드 유지
    out = dict(ex)
    out["eeg"] = eeg2

    meta2 = dict(meta)
    meta2["duration_sec"] = int(crop_sec)
    meta2["parent_duration_sec"] = int(duration_sec)
    meta2["crop_start"] = int(start)
    meta2["crop_start_sec"] = float(start_sec)
    meta2["cropped_from_long_window"] = True
    out["meta"] = meta2

    # (선택) key를 바꿔서 디버깅/분석에 도움
    # if "__key__" in out and isinstance(out["__key__"], str):
    #     out["__key__"] = f"{out['__key__']}_t{int(start_sec):02d}_d{crop_sec:02d}"

    return out


def maybe_group_channels(
    ex: Dict[str, Any],
    max_tokens: int,
    patch_samples: int,
    hop_samples: int,
    prefer_durations_sec: tuple = (10, 30, 60),   # 너의 정책 반영
) -> Dict[str, Any]:
    """
    기존: max_tokens 초과 시 채널을 K로 줄임
    변경: 채널은 유지하고, 필요 시 시간 길이를 crop 해서 P_t를 줄임 (예: 60s -> 30s)
    """
    eeg: torch.Tensor = ex["eeg"]      # (C,T)
    coord: torch.Tensor = ex["coord"]  # (C,3)
    meta: Dict[str, Any] = ex.get("meta", {})

    C, T = eeg.shape
    P_t = compute_num_patches(T, patch_samples=patch_samples, hop_samples=hop_samples)
    if P_t <= 0:
        ex["n_tokens"] = 0
        ex["n_patches"] = 0
        ex["n_channels"] = int(C)
        return ex

    N = C * P_t
    if N <= max_tokens:
        ex["n_tokens"] = int(N)
        ex["n_patches"] = int(P_t)
        ex["n_channels"] = int(C)
        return ex

    # ---- 토큰 초과: 채널 유지, P를 줄이기 ----
    P_max = max_tokens // C  # 허용 가능한 최대 패치 수
    if P_max < 1:
        # 이 케이스는 C가 너무 커서 1패치도 못 넣는 상황.
        # 너의 데이터(C<=134)에서는 사실상 안 생김.
        ex["n_tokens"] = 0
        ex["n_patches"] = 0
        ex["n_channels"] = int(C)
        return ex

    # prefer durations 중 "가능한 가장 긴 길이" 선택 (예: 60->30)
    fs = int(meta.get("fs", 200))  # 전처리에서 200Hz 고정이라면 200으로 fallback OK
    best_P = None
    best_Tneed = None

    for sec in prefer_durations_sec:
        T_sec = sec * fs
        P_sec = compute_num_patches(T_sec, patch_samples=patch_samples, hop_samples=hop_samples)
        if P_sec <= P_max and P_sec <= P_t:
            if best_P is None or P_sec > best_P:
                best_P = P_sec
                best_Tneed = (P_sec - 1) * hop_samples + patch_samples

    if best_P is None:
        # prefer durations로 안 되면, P_max에 맞춰 crop (일반적 fallback)
        P_new = min(P_t, P_max)
        T_need = (P_new - 1) * hop_samples + patch_samples
    else:
        P_new = best_P
        T_need = best_Tneed

    if T_need <= 0:
        ex["n_tokens"] = 0
        ex["n_patches"] = 0
        ex["n_channels"] = int(C)
        return ex

    # 랜덤 crop (다양성 확보)
    if T > T_need:
        start = random.randint(0, T - T_need)
    else:
        start = 0

    eeg = eeg[:, start:start + T_need].contiguous()

    ex["eeg"] = eeg
    ex["coord"] = coord
    # meta 업데이트(디버깅/추적용)
    ex["meta"] = dict(meta)
    ex["meta"]["cropped_to_fit_tokens"] = True
    ex["meta"]["crop_start"] = int(start)

    # 재계산
    P_t2 = compute_num_patches(eeg.shape[1], patch_samples=patch_samples, hop_samples=hop_samples)
    ex["n_tokens"] = int(C * P_t2)
    ex["n_patches"] = int(P_t2)
    ex["n_channels"] = int(C)
    return ex


def collate_stack(batch: List[Dict[str, Any]], patch_samples: int, hop_samples: int) -> Dict[str, Any]:
    """
    (C,P) 동일 shape 배치 전용 collate. padding 없이 stack.
    - batch 내 모든 샘플의 n_channels, n_patches가 동일해야 함.
    """
    assert len(batch) > 0
    C = int(batch[0]["n_channels"])
    P = int(batch[0]["n_patches"])
    for b in batch:
        assert int(b["n_channels"]) == C and int(b["n_patches"]) == P, "ShapeBatcher produced mixed (C,P) batch"

    T_need = (P - 1) * hop_samples + patch_samples if P > 0 else 0

    eeg = torch.stack([b["eeg"][:, :T_need].contiguous() for b in batch], dim=0)    # (B,C,T)
    coord = torch.stack([b["coord"].contiguous() for b in batch], dim=0)           # (B,C,3)

    B = len(batch)
    n_channels = torch.full((B,), C, dtype=torch.long)
    n_patches = torch.full((B,), P, dtype=torch.long)

    meta = [b.get("meta", {}) for b in batch]

    return {
        "eeg": eeg,
        "coord": coord,
        "n_channels": n_channels,
        "n_patches": n_patches,
        "meta": meta,
    }


class LRUShardCache:
    """
    On-demand shard cache with LRU eviction by mtime.
    - shard path -> cached path
    - copy on miss (atomic rename)
    - touch(mtime) on access
    - evict oldest until total_bytes <= max_bytes

    NOTE:
      - basename 충돌 방지 위해 full path hash prefix 사용
      - 멀티프로세스/멀티워커 레이스는 lock + atomic rename + try/except로 완화
    """
    def __init__(
        self,
        cache_dir: str,
        max_bytes: int,                 # e.g. 500 * 1024**3
        eviction_interval: int = 64,     # 매 N번 access마다 eviction check
        enable_eviction: bool = True,
    ):
        self.cache_dir = cache_dir
        self.max_bytes = int(max_bytes)
        self.eviction_interval = int(eviction_interval)
        self.enable_eviction = bool(enable_eviction)

        os.makedirs(self.cache_dir, exist_ok=True)
        self.lock_path = os.path.join(self.cache_dir, ".lru.lock")
        self._counter = 0

    def __call__(self, shard_path: str) -> str:
        return self.get(shard_path)

    def _cache_name(self, shard_path: str) -> str:
        h = hashlib.sha1(shard_path.encode("utf-8")).hexdigest()[:16]
        base = os.path.basename(shard_path)
        return f"{h}-{base}"

    def get(self, shard_path: str) -> str:
        cached = os.path.join(self.cache_dir, self._cache_name(shard_path))

        if not os.path.exists(cached):
            tmp = cached + f".tmp.{os.getpid()}"
            copied = False
            try:
                try:
                    shutil.copyfile(shard_path, tmp)
                    os.replace(tmp, cached)
                    copied = True
                except OSError as e:
                    # No space left on device -> eviction retry
                    if self.enable_eviction and getattr(e, "errno", None) == 28:
                        self.evict_if_needed()
                        shutil.copyfile(shard_path, tmp)
                        os.replace(tmp, cached)
                        copied = True
            except Exception:
                copied = False
            finally:
                try:
                    if os.path.exists(tmp):
                        os.remove(tmp)
                except Exception:
                    pass

            # if copy failed, return original path (fallback)
            if not copied or not os.path.exists(cached):
                return shard_path

        # touch: LRU 위해 mtime 갱신
        try:
            os.utime(cached, None)
        except Exception:
            pass

        self._counter += 1
        if self.enable_eviction and (self._counter % self.eviction_interval == 0):
            self.evict_if_needed()

        return cached

    def evict_if_needed(self) -> None:
        if self.max_bytes <= 0:
            return

        with _file_lock(self.lock_path):
            # 캐시 파일 목록(임시파일 제외)
            files = []
            total = 0
            for name in os.listdir(self.cache_dir):
                if name.endswith(".tmp") or ".tmp." in name:
                    continue
                path = os.path.join(self.cache_dir, name)
                if not os.path.isfile(path):
                    continue
                try:
                    st = os.stat(path)
                except FileNotFoundError:
                    continue
                files.append((st.st_mtime, st.st_size, path))
                total += st.st_size

            if total <= self.max_bytes:
                return

            # 오래된 것부터 삭제
            files.sort(key=lambda x: x[0])  # oldest mtime first
            for _, sz, path in files:
                try:
                    os.remove(path)
                    total -= sz
                except FileNotFoundError:
                    pass
                except Exception:
                    pass
                if total <= self.max_bytes:
                    break


class ShapeBatcher(IterableDataset):
    """
    (C,P) 완전 동일 shape 기준으로 배치 생성:
      key = (n_channels, n_patches)

    - tokens_per_batch 목표에 맞춰 shape별 batch size를 자동 결정:
        bs_target = floor(tokens_per_batch / (C*P))
        (최소 1, 최대 max_samples_per_batch)

    - 희귀 shape가 계속 쌓이지 않도록 flush 정책:
      1) max_wait_samples:
         어떤 key에 첫 샘플이 들어온 뒤, global seen count 기준으로 max_wait_samples 이상 지나도
         batch가 안 채워지면 "현재까지 모인 것"을 작은 배치로 방출(yield)하고 비움.
      2) max_pending_samples / max_pending_tokens:
         전체 버퍼에 쌓인 샘플(또는 토큰)이 너무 많아지면 가장 오래된 key부터 강제 flush.

    - flush_check_every:
         매 샘플마다 모든 key를 검사하면 비효율이므로, N샘플마다 한 번만 expired 검사를 수행.

    NOTE:
      flush로 작은 배치가 나올 수 있음 -> token-budget 누적 step(train.py)과 같이 쓰는 게 정석.
    """
    def __init__(
        self,
        dataset: Iterable[Dict[str, Any]],
        tokens_per_batch: int,
        max_samples_per_batch: int,
        patch_samples: int,
        hop_samples: int,
        # flush 정책
        max_wait_samples: int = 5000,
        flush_check_every: int = 256,
        max_pending_samples: int = 512,
        max_pending_tokens: int = 0,      # 0이면 비활성
        # 기타
        shuffle_within_bucket: bool = True,
        yield_incomplete: bool = True,    # False면 flush 시 버림(drop)
    ):
        self.dataset = dataset
        self.tokens_per_batch = int(tokens_per_batch)
        self.max_samples_per_batch = int(max_samples_per_batch)
        self.patch_samples = int(patch_samples)
        self.hop_samples = int(hop_samples)

        self.max_wait_samples = int(max_wait_samples)
        self.flush_check_every = int(flush_check_every)
        self.max_pending_samples = int(max_pending_samples)
        self.max_pending_tokens = int(max_pending_tokens)

        self.shuffle_within_bucket = bool(shuffle_within_bucket)
        self.yield_incomplete = bool(yield_incomplete)

        assert self.tokens_per_batch > 0
        assert self.max_samples_per_batch > 0
        assert self.flush_check_every > 0

    def _target_bs(self, C: int, P: int) -> int:
        tokens_per_sample = C * P
        if tokens_per_sample <= 0:
            return 1
        bs = self.tokens_per_batch // tokens_per_sample
        bs = max(1, bs)
        bs = min(bs, self.max_samples_per_batch)
        return bs

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        buckets: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
        first_seen: Dict[Tuple[int, int], int] = {}

        seen = 0
        pending_samples = 0
        pending_tokens = 0

        def _flush_key(key: Tuple[int, int]):
            nonlocal pending_samples, pending_tokens
            buf = buckets.pop(key, [])
            first_seen.pop(key, None)
            if not buf:
                return

            pending_samples -= len(buf)
            pending_tokens -= sum(int(x.get("n_tokens", 0)) for x in buf)

            if self.yield_incomplete:
                yield collate_stack(buf, patch_samples=self.patch_samples, hop_samples=self.hop_samples)
            # else: drop

        def _flush_oldest_until_under_limits():
            nonlocal pending_samples, pending_tokens
            # pending_samples / pending_tokens가 상한을 넘으면 oldest key부터 flush
            while True:
                over_samples = (self.max_pending_samples > 0) and (pending_samples > self.max_pending_samples)
                over_tokens = (self.max_pending_tokens > 0) and (pending_tokens > self.max_pending_tokens)
                if not (over_samples or over_tokens):
                    break
                if not first_seen:
                    break
                oldest = min(first_seen, key=first_seen.get)
                for out in _flush_key(oldest):
                    yield out

        def _flush_expired():
            if self.max_wait_samples <= 0:
                return
            # seen 기준으로 오래된 key flush
            expired = []
            for k, t0 in first_seen.items():
                if (seen - t0) >= self.max_wait_samples:
                    expired.append(k)
            for k in expired:
                for out in _flush_key(k):
                    yield out

        for ex in self.dataset:
            seen += 1

            C = int(ex.get("n_channels", 0))
            P = int(ex.get("n_patches", 0))
            if C <= 0 or P <= 0:
                continue

            key = (C, P)
            if key not in buckets:
                buckets[key] = []
                first_seen[key] = seen

            buckets[key].append(ex)
            pending_samples += 1
            pending_tokens += int(ex.get("n_tokens", C * P))

            # shape별 목표 batch size
            bs_target = self._target_bs(C, P)

            # 충분히 모이면 즉시 방출(가능하면 여러 배치)
            buf = buckets[key]
            if self.shuffle_within_bucket and len(buf) == bs_target:
                random.shuffle(buf)

            while len(buf) >= bs_target:
                batch = buf[:bs_target]
                del buf[:bs_target]
                pending_samples -= bs_target
                pending_tokens -= sum(int(x.get("n_tokens", C * P)) for x in batch)

                yield collate_stack(batch, patch_samples=self.patch_samples, hop_samples=self.hop_samples)

                # 남은 게 있으면 wait 타이머 리셋(남은 샘플들이 바로 flush되지 않게)
                if len(buf) > 0:
                    first_seen[key] = seen
                else:
                    buckets.pop(key, None)
                    first_seen.pop(key, None)
                    break

            # (1) 버퍼가 너무 커지면 oldest flush
            for out in _flush_oldest_until_under_limits():
                yield out

            # (2) 주기적으로 expired flush
            if (seen % self.flush_check_every) == 0:
                for out in _flush_expired():
                    yield out

        # dataset이 finite일 경우 마지막에 남은 것 flush
        if self.yield_incomplete:
            for k in list(buckets.keys()):
                for out in _flush_key(k):
                    yield out


def build_webdataset(
    shards: List[str],
    cache_dir: Optional[str],
    shard_shuffle: int,
    sample_shuffle: int,
    long_crop_prob: float,
    long_crop_30_prob: float,
    max_tokens: int,
    patch_samples: int,
    hop_samples: int,
    enable_channel_grouping: bool,
    limit_num_samples: int = 0,
    cache_max_bytes: int = 0, # ex: 500GB -> 500*1024**3
    post_split_shuffle: int = 256,
    eviction_interval: int = 8,
) -> Iterable[Dict[str, Any]]:
    if wds is None:
        raise RuntimeError("webdataset is not installed. pip install webdataset")

    # on-demand LRU cache
    cache = None
    if cache_dir and cache_max_bytes > 0:
        # enable_evict = (int(os.environ.get("LOCAL_RANK", "0")) == 0)
        cache = LRUShardCache(
            cache_dir=cache_dir,
            max_bytes=cache_max_bytes,
            eviction_interval=eviction_interval,
            enable_eviction=True,
        )

    def _prep(ex):
        if enable_channel_grouping:
            ex = maybe_group_channels(ex, max_tokens=max_tokens, patch_samples=patch_samples, hop_samples=hop_samples)
        else:
            eeg = ex["eeg"]
            C, T = eeg.shape
            P_t = compute_num_patches(T, patch_samples=patch_samples, hop_samples=hop_samples)
            ex["n_tokens"] = int(C * P_t)
            ex["n_patches"] = int(P_t)
            ex["n_channels"] = int(C)
        return ex

    # ---- Recommended pipeline: shard-level shuffle -> split_by_node/worker -> (optional) cache -> tar -> sample shuffle(before decode) ----
    src = wds.ResampledShards(shards)  # infinite pretrain
    pipeline = [
        src,
        # shard-level shuffle (decode 이전, 가장 가벼움)
        # webdataset에 detshuffle이 있으면 그걸 추천 (재현/분산에서 안정)
        wds.shuffle(shard_shuffle),
        wds.split_by_node,
        wds.split_by_worker,
    ]

    if cache is not None:
        # shard url(str) -> cached path(str)
        pipeline.append(wds.map(cache))

    pipeline += [
        wds.tarfile_to_samples(handler=wds.ignore_and_continue),
        # sample shuffle BEFORE decode: bytes 수준에서 섞기 (torch 텐서 단계보다 RAM 부담 적음)
        wds.shuffle(sample_shuffle),
        wds.decode(),
        wds.map(decode_sample),  # 여기서 torch로 변환
    ]
    if long_crop_prob and long_crop_prob > 0:
        pipeline.append(wds.map(lambda ex: maybe_random_window_crop_60s(ex, crop_prob=long_crop_prob, crop_30_prob=long_crop_30_prob)))
    # small shuffle after crop/split
    pipeline += [
        wds.shuffle(post_split_shuffle),
        wds.map(_prep),
    ]

    ds = wds.DataPipeline(*pipeline)

    if limit_num_samples and limit_num_samples > 0:
        ds = ds.take(limit_num_samples)

    return ds