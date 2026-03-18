#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import random
import re
import sys
import tarfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np


# -----------------------------
# Path helpers
# -----------------------------
WINDOWS_DRIVE_RE = re.compile(r"^([A-Za-z]):[\\/](.*)$")
WSL_MNT_RE = re.compile(r"^/mnt/([a-zA-Z])(?:/(.*))?$")


def windows_to_wsl(path_str: str) -> str:
    m = WINDOWS_DRIVE_RE.match(path_str)
    if not m:
        return path_str
    drive = m.group(1).lower()
    rest = m.group(2).replace("\\", "/")
    return f"/mnt/{drive}/{rest}"


def wsl_to_windows(path_str: str) -> str:
    m = WSL_MNT_RE.match(path_str)
    if not m:
        return path_str
    drive = m.group(1).upper()
    rest = (m.group(2) or "").replace("/", "\\")
    return f"{drive}:\\{rest}" if rest else f"{drive}:\\"



def resolve_scan_root(path_str: str) -> Path:
    # Accept Windows-style input when running under WSL/Linux.
    if os.name != "nt":
        path_str = windows_to_wsl(path_str)
    return Path(path_str).expanduser().resolve()



def format_path_for_manifest(path: Path, style: str) -> str:
    p = str(path)
    style = style.lower().strip()
    if style == "scan":
        return p
    if style == "wsl":
        return windows_to_wsl(p) if WINDOWS_DRIVE_RE.match(p) else p
    if style == "windows":
        return wsl_to_windows(p)
    if style == "relative":
        return path.as_posix()
    raise ValueError(f"unknown path style: {style}")


# -----------------------------
# WebDataset member parsing
# -----------------------------
EEG_PATTERNS = (".eeg.npy", "/eeg.npy", "eeg.npy")
COORD_PATTERNS = (".coords.npy", "/coords.npy", ".coord.npy", "/coord.npy", "coords.npy", "coord.npy")
META_PATTERNS = (".meta.json", "/meta.json", "meta.json")
JSON_PATTERNS = (".json", "/json")


def _norm_member_name(name: str) -> str:
    return name.lstrip("./")



def split_member_prefix_and_kind(name: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Accept both common WebDataset naming conventions:
      - samplekey.eeg.npy
      - samplekey.coords.npy
      - samplekey.meta.json
    and directory-like layouts:
      - samplekey/eeg.npy
      - samplekey/coords.npy
      - samplekey/meta.json
    Returns (sample_prefix, kind) or (None, None).
    """
    name = _norm_member_name(name)

    for pat in (".eeg.npy", "/eeg.npy"):
        if name.endswith(pat):
            return name[: -len(pat)], "eeg"
    for pat in (".coords.npy", "/coords.npy", ".coord.npy", "/coord.npy"):
        if name.endswith(pat):
            return name[: -len(pat)], "coords"
    for pat in (".meta.json", "/meta.json"):
        if name.endswith(pat):
            return name[: -len(pat)], "meta"

    # Very defensive fallback for flat names like eeg.npy / coords.npy / meta.json
    base = os.path.basename(name)
    parent = os.path.dirname(name)
    if base == "eeg.npy":
        return parent or "__root__", "eeg"
    if base in ("coords.npy", "coord.npy"):
        return parent or "__root__", "coords"
    if base == "meta.json":
        return parent or "__root__", "meta"
    return None, None


# -----------------------------
# Shape / token estimation
# -----------------------------

def compute_num_patches(T: int, patch_samples: int, hop_samples: int) -> int:
    if T < patch_samples:
        return 0
    return int((T - patch_samples) // hop_samples + 1)



def infer_fs_from_meta(meta: Dict[str, Any], default_fs: int) -> int:
    keys = [
        "fs",
        "sfreq",
        "sampling_rate",
        "sample_rate",
        "sr",
        "freq",
    ]
    for k in keys:
        if k in meta:
            try:
                v = float(meta[k])
                if v > 0:
                    return int(round(v))
            except Exception:
                pass
    return int(default_fs)



def infer_duration_sec(meta: Dict[str, Any], T: int, fs: int) -> Optional[float]:
    for k in ("duration_sec", "segment_sec", "duration_seconds", "window_sec", "seconds"):
        if k in meta:
            try:
                v = float(meta[k])
                if v > 0:
                    return v
            except Exception:
                pass
    if fs > 0:
        return float(T) / float(fs)
    return None



def snap_duration(duration_sec: Optional[float]) -> Optional[int]:
    if duration_sec is None:
        return None
    candidates = [10, 30, 60]
    for c in candidates:
        if abs(float(duration_sec) - c) <= 0.25:
            return c
    return int(round(float(duration_sec)))



def estimate_fit_shape(
    *,
    C: int,
    T: int,
    fs: int,
    patch_seconds: float,
    hop_seconds: float,
    max_tokens: int,
    prefer_durations_sec: List[int],
) -> Dict[str, Any]:
    patch_samples = max(1, int(round(float(patch_seconds) * fs)))
    hop_samples = max(1, int(round(float(hop_seconds) * fs)))

    raw_patches = compute_num_patches(T, patch_samples=patch_samples, hop_samples=hop_samples)
    raw_tokens = int(C * raw_patches)
    raw_duration = float(T) / float(fs) if fs > 0 else None

    out = {
        "patch_samples": patch_samples,
        "hop_samples": hop_samples,
        "raw_patches": int(raw_patches),
        "raw_tokens": int(raw_tokens),
        "raw_duration_sec": raw_duration,
        "fit_patches": int(raw_patches),
        "fit_tokens": int(raw_tokens),
        "fit_duration_sec": raw_duration,
        "cropped_to_fit_tokens": False,
    }

    if max_tokens <= 0 or raw_tokens <= max_tokens or C <= 0:
        return out

    p_max = max_tokens // C
    if p_max < 1:
        out.update(
            {
                "fit_patches": 0,
                "fit_tokens": 0,
                "fit_duration_sec": 0.0,
                "cropped_to_fit_tokens": True,
            }
        )
        return out

    best_p = None
    best_duration = None
    for sec in prefer_durations_sec:
        t_sec = int(round(sec * fs))
        p_sec = compute_num_patches(t_sec, patch_samples=patch_samples, hop_samples=hop_samples)
        if p_sec <= p_max and p_sec <= raw_patches:
            if best_p is None or p_sec > best_p:
                best_p = p_sec
                best_duration = float(sec)

    if best_p is None:
        fit_patches = min(raw_patches, p_max)
        fit_T = (fit_patches - 1) * hop_samples + patch_samples if fit_patches > 0 else 0
        fit_duration = float(fit_T) / float(fs) if fs > 0 else 0.0
    else:
        fit_patches = int(best_p)
        fit_duration = float(best_duration)

    out.update(
        {
            "fit_patches": int(fit_patches),
            "fit_tokens": int(C * fit_patches),
            "fit_duration_sec": fit_duration,
            "cropped_to_fit_tokens": True,
        }
    )
    return out


# -----------------------------
# Tar inspection
# -----------------------------
@dataclass
class InspectConfig:
    root: str
    output_root: str
    path_style: str
    default_fs: int
    patch_seconds: float
    hop_seconds: float
    max_tokens: int
    prefer_durations_sec: List[int]
    dataset_depth: int
    nominal_shard_size_bytes: int
    strict: bool



def rel_dataset_id(path: Path, root: Path, depth: int = 1) -> str:
    rel = path.relative_to(root)
    parts = rel.parts
    if len(parts) <= 1:
        return "__root__"
    depth = max(1, int(depth))
    take = min(depth, len(parts) - 1)
    return "/".join(parts[:take])



def _read_numpy_member(tf: tarfile.TarFile, member: tarfile.TarInfo) -> np.ndarray:
    f = tf.extractfile(member)
    if f is None:
        raise RuntimeError(f"cannot extract {member.name}")
    data = f.read()
    arr = np.load(io.BytesIO(data), allow_pickle=False)
    if isinstance(arr, np.lib.npyio.NpzFile):
        if len(arr.files) == 0:
            raise ValueError(f"empty NPZ payload in {member.name}")
        arr = arr[arr.files[0]]
    return arr



def _read_json_member(tf: tarfile.TarFile, member: tarfile.TarInfo) -> Dict[str, Any]:
    f = tf.extractfile(member)
    if f is None:
        return {}
    try:
        return json.loads(f.read().decode("utf-8"))
    except Exception:
        return {}



def inspect_one_tar(tar_path_str: str, cfg: InspectConfig) -> Dict[str, Any]:
    tar_path = Path(tar_path_str)
    root = Path(cfg.root)
    output_root = Path(cfg.output_root)
    stat = tar_path.stat()

    dataset_id = rel_dataset_id(tar_path, root=root, depth=cfg.dataset_depth)
    rel_path = tar_path.relative_to(root)

    record: Dict[str, Any] = {
        "dataset_id": dataset_id,
        "scan_path": str(tar_path),
        "shard_path": format_path_for_manifest(output_root / rel_path, cfg.path_style),
        "relative_path": rel_path.as_posix(),
        "basename": tar_path.name,
        "size_bytes": int(stat.st_size),
        "size_gb": round(stat.st_size / (1024 ** 3), 6),
        "is_tail_shard": bool(stat.st_size < int(0.80 * cfg.nominal_shard_size_bytes)),
        "segment_count": 0,
        "channels": None,
        "timepoints": None,
        "coords_channels": None,
        "coords_dims": None,
        "fs": None,
        "duration_sec": None,
        "duration_bucket_sec": None,
        "raw_patches": None,
        "raw_tokens": None,
        "fit_patches": None,
        "fit_tokens": None,
        "fit_duration_sec": None,
        "cropped_to_fit_tokens": None,
        "shape_key": None,
        "fit_shape_key": None,
        "dataset_shape_key": None,
        "dataset_fit_shape_key": None,
        "first_sample_key": None,
        "warnings": [],
        "error": None,
    }

    try:
        with tarfile.open(tar_path, mode="r") as tf:
            first_eeg_member: Optional[tarfile.TarInfo] = None
            first_prefix: Optional[str] = None
            meta_members: Dict[str, tarfile.TarInfo] = {}
            coords_members: Dict[str, tarfile.TarInfo] = {}

            for member in tf:
                if not member.isfile():
                    continue
                prefix, kind = split_member_prefix_and_kind(member.name)
                if kind is None:
                    continue
                if kind == "eeg":
                    record["segment_count"] += 1
                    if first_eeg_member is None:
                        first_eeg_member = member
                        first_prefix = prefix
                elif kind == "meta":
                    meta_members.setdefault(prefix or "", member)
                elif kind == "coords":
                    coords_members.setdefault(prefix or "", member)

            if first_eeg_member is None:
                raise RuntimeError("no eeg.npy member found")

            record["first_sample_key"] = first_prefix
            eeg = _read_numpy_member(tf, first_eeg_member)
            if eeg.ndim < 2:
                raise ValueError(f"unexpected EEG shape: {tuple(eeg.shape)}")
            C = int(eeg.shape[0])
            T = int(eeg.shape[1])
            record["channels"] = C
            record["timepoints"] = T

            meta: Dict[str, Any] = {}
            meta_member = meta_members.get(first_prefix or "")
            if meta_member is not None:
                meta = _read_json_member(tf, meta_member)

            coords_member = coords_members.get(first_prefix or "")
            if coords_member is not None:
                try:
                    coords = _read_numpy_member(tf, coords_member)
                    if coords.ndim >= 1:
                        record["coords_channels"] = int(coords.shape[0])
                    if coords.ndim >= 2:
                        record["coords_dims"] = int(coords.shape[-1])
                    if record["coords_channels"] is not None and record["coords_channels"] != C:
                        record["warnings"].append(
                            f"coords/eeg channel mismatch: eeg={C}, coords={record['coords_channels']}"
                        )
                except Exception as e:
                    record["warnings"].append(f"coords read failed: {type(e).__name__}: {e}")

            fs = infer_fs_from_meta(meta, default_fs=cfg.default_fs)
            duration_sec = infer_duration_sec(meta, T=T, fs=fs)
            duration_bucket = snap_duration(duration_sec)
            fit = estimate_fit_shape(
                C=C,
                T=T,
                fs=fs,
                patch_seconds=cfg.patch_seconds,
                hop_seconds=cfg.hop_seconds,
                max_tokens=cfg.max_tokens,
                prefer_durations_sec=cfg.prefer_durations_sec,
            )

            record.update(
                {
                    "fs": int(fs),
                    "duration_sec": None if duration_sec is None else round(float(duration_sec), 6),
                    "duration_bucket_sec": duration_bucket,
                    "raw_patches": int(fit["raw_patches"]),
                    "raw_tokens": int(fit["raw_tokens"]),
                    "fit_patches": int(fit["fit_patches"]),
                    "fit_tokens": int(fit["fit_tokens"]),
                    "fit_duration_sec": round(float(fit["fit_duration_sec"]), 6),
                    "cropped_to_fit_tokens": bool(fit["cropped_to_fit_tokens"]),
                    "shape_key": f"C{C}_D{duration_bucket if duration_bucket is not None else 'unk'}_P{fit['raw_patches']}",
                    "fit_shape_key": f"C{C}_FD{snap_duration(fit['fit_duration_sec']) if fit['fit_duration_sec'] else 0}_FP{fit['fit_patches']}",
                    "dataset_shape_key": f"{dataset_id}|C{C}_D{duration_bucket if duration_bucket is not None else 'unk'}_P{fit['raw_patches']}",
                    "dataset_fit_shape_key": f"{dataset_id}|C{C}_FD{snap_duration(fit['fit_duration_sec']) if fit['fit_duration_sec'] else 0}_FP{fit['fit_patches']}",
                }
            )

    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        record["error"] = msg
        if cfg.strict:
            raise

    return record


# -----------------------------
# Aggregation / window building
# -----------------------------

def counter_to_sorted_dict(counter: Counter) -> Dict[str, int]:
    return {str(k): int(counter[k]) for k in sorted(counter, key=lambda x: (str(x)))}



def weighted_mode(counter: Counter) -> Optional[Any]:
    if not counter:
        return None
    return max(counter.items(), key=lambda kv: (kv[1], str(kv[0])))[0]



def summarize_datasets(shards: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for s in shards:
        groups[s["dataset_id"]].append(s)

    out: Dict[str, Dict[str, Any]] = {}
    for dataset_id, xs in sorted(groups.items(), key=lambda kv: kv[0]):
        ch = Counter()
        dur = Counter()
        fs = Counter()
        raw_tok = Counter()
        fit_tok = Counter()
        fit_shape = Counter()
        err = 0
        warnings_count = 0
        total_bytes = 0
        total_segments = 0
        tail_shards = 0

        for x in xs:
            segs = int(x.get("segment_count") or 0)
            w = max(1, segs)
            total_bytes += int(x["size_bytes"])
            total_segments += segs
            tail_shards += int(bool(x.get("is_tail_shard")))
            warnings_count += len(x.get("warnings") or [])
            if x.get("error"):
                err += 1
            if x.get("channels") is not None:
                ch[int(x["channels"])] += w
            if x.get("duration_bucket_sec") is not None:
                dur[int(x["duration_bucket_sec"])] += w
            if x.get("fs") is not None:
                fs[int(x["fs"])] += w
            if x.get("raw_tokens") is not None:
                raw_tok[int(x["raw_tokens"])] += w
            if x.get("fit_tokens") is not None:
                fit_tok[int(x["fit_tokens"])] += w
            if x.get("fit_shape_key") is not None:
                fit_shape[str(x["fit_shape_key"])] += w

        out[dataset_id] = {
            "dataset_id": dataset_id,
            "num_shards": len(xs),
            "tail_shards": int(tail_shards),
            "total_bytes": int(total_bytes),
            "total_gb": round(total_bytes / (1024 ** 3), 6),
            "total_segments": int(total_segments),
            "segments_per_gb": round(total_segments / max(total_bytes / (1024 ** 3), 1e-9), 3),
            "channels_mode": weighted_mode(ch),
            "duration_mode_sec": weighted_mode(dur),
            "fs_mode": weighted_mode(fs),
            "fit_shape_mode": weighted_mode(fit_shape),
            "channel_hist_by_segments": counter_to_sorted_dict(ch),
            "duration_hist_by_segments": counter_to_sorted_dict(dur),
            "fs_hist_by_segments": counter_to_sorted_dict(fs),
            "raw_tokens_hist_by_segments": counter_to_sorted_dict(raw_tok),
            "fit_tokens_hist_by_segments": counter_to_sorted_dict(fit_tok),
            "fit_shape_hist_by_segments": counter_to_sorted_dict(fit_shape),
            "num_error_shards": int(err),
            "num_warnings": int(warnings_count),
        }
    return out



def choose_resident_shards(
    shards: List[Dict[str, Any]],
    *,
    resident_budget_bytes: int,
    resident_group_max_bytes: int,
    resident_group_max_shards: int,
    rare_key: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    if resident_budget_bytes <= 0:
        return [], list(shards), []

    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for s in shards:
        groups[str(s.get(rare_key) or "UNK")].append(s)

    candidates: List[Tuple[str, List[Dict[str, Any]], int]] = []
    for k, xs in groups.items():
        total_bytes = sum(int(x["size_bytes"]) for x in xs)
        if (resident_group_max_bytes > 0 and total_bytes <= resident_group_max_bytes) or (
            resident_group_max_shards > 0 and len(xs) <= resident_group_max_shards
        ):
            candidates.append((k, xs, total_bytes))

    # Smallest rare groups first; if tie, fewer shards first.
    candidates.sort(key=lambda t: (t[2], len(t[1]), t[0]))

    resident: List[Dict[str, Any]] = []
    resident_groups: List[Dict[str, Any]] = []
    used = 0
    resident_keys = set()
    for key, xs, total_bytes in candidates:
        if used + total_bytes > resident_budget_bytes:
            continue
        resident.extend(xs)
        resident_keys.add(key)
        resident_groups.append(
            {
                "group_key": key,
                "num_shards": len(xs),
                "total_bytes": int(total_bytes),
                "total_gb": round(total_bytes / (1024 ** 3), 6),
                "datasets": sorted({x["dataset_id"] for x in xs}),
            }
        )
        used += total_bytes

    rotating = [s for s in shards if str(s.get(rare_key) or "UNK") not in resident_keys]
    return resident, rotating, resident_groups



def assign_windows(
    shards: List[Dict[str, Any]],
    *,
    window_target_bytes: int,
    seed: int,
    group_key: str,
) -> List[Dict[str, Any]]:
    if not shards:
        return []
    total_bytes = sum(int(s["size_bytes"]) for s in shards)
    n_windows = max(1, int(math.ceil(total_bytes / max(1, window_target_bytes))))

    rng = random.Random(seed)
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for s in shards:
        groups[str(s.get(group_key) or "UNK")].append(s)

    windows: List[Dict[str, Any]] = []
    for i in range(n_windows):
        windows.append(
            {
                "window_id": i,
                "shards": [],
                "total_bytes": 0,
                "dataset_bytes": defaultdict(int),
                "group_bytes": defaultdict(int),
                "segments": 0,
            }
        )

    ordered_groups = sorted(groups.items(), key=lambda kv: sum(int(x["size_bytes"]) for x in kv[1]), reverse=True)
    for gkey, xs in ordered_groups:
        xs = list(xs)
        rng.shuffle(xs)
        xs.sort(key=lambda s: int(s["size_bytes"]), reverse=True)
        for s in xs:
            dataset_id = str(s["dataset_id"])
            size = int(s["size_bytes"])
            # Spread same group first, then same dataset, then overall size.
            best = min(
                windows,
                key=lambda w: (
                    w["group_bytes"][gkey],
                    w["dataset_bytes"][dataset_id],
                    w["total_bytes"],
                    w["window_id"],
                ),
            )
            best["shards"].append(s)
            best["total_bytes"] += size
            best["segments"] += int(s.get("segment_count") or 0)
            best["dataset_bytes"][dataset_id] += size
            best["group_bytes"][gkey] += size

    manifest_windows: List[Dict[str, Any]] = []
    for w in windows:
        dataset_hist = Counter()
        fit_shape_hist = Counter()
        channels_hist = Counter()
        duration_hist = Counter()
        fit_tokens_hist = Counter()
        for s in w["shards"]:
            size = int(s["size_bytes"])
            dataset_hist[s["dataset_id"]] += size
            if s.get("fit_shape_key") is not None:
                fit_shape_hist[str(s["fit_shape_key"])] += size
            if s.get("channels") is not None:
                channels_hist[int(s["channels"])] += size
            if s.get("duration_bucket_sec") is not None:
                duration_hist[int(s["duration_bucket_sec"])] += size
            if s.get("fit_tokens") is not None:
                fit_tokens_hist[int(s["fit_tokens"])] += size
        manifest_windows.append(
            {
                "window_id": int(w["window_id"]),
                "num_shards": len(w["shards"]),
                "total_bytes": int(w["total_bytes"]),
                "total_gb": round(w["total_bytes"] / (1024 ** 3), 6),
                "total_segments": int(w["segments"]),
                "dataset_bytes_hist": counter_to_sorted_dict(dataset_hist),
                "fit_shape_bytes_hist": counter_to_sorted_dict(fit_shape_hist),
                "channels_bytes_hist": counter_to_sorted_dict(channels_hist),
                "duration_bytes_hist": counter_to_sorted_dict(duration_hist),
                "fit_tokens_bytes_hist": counter_to_sorted_dict(fit_tokens_hist),
                "shards": [s["shard_path"] for s in sorted(w["shards"], key=lambda x: x["relative_path"])],
                "relative_paths": [s["relative_path"] for s in sorted(w["shards"], key=lambda x: x["relative_path"])],
            }
        )
    return manifest_windows


# -----------------------------
# Writers
# -----------------------------

def write_csv(path: Path, rows: Iterable[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            flat = {}
            for k in fieldnames:
                v = row.get(k)
                if isinstance(v, (dict, list)):
                    flat[k] = json.dumps(v, ensure_ascii=False, sort_keys=True)
                else:
                    flat[k] = v
            w.writerow(flat)



def write_window_lists(out_dir: Path, windows: List[Dict[str, Any]], resident_shards: List[str]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if resident_shards:
        with open(out_dir / "resident_shards.txt", "w", encoding="utf-8") as f:
            for p in resident_shards:
                f.write(f"{p}\n")
    for w in windows:
        wid = int(w["window_id"])
        with open(out_dir / f"window_{wid:03d}.txt", "w", encoding="utf-8") as f:
            for p in w["shards"]:
                f.write(f"{p}\n")


# -----------------------------
# CLI
# -----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build a staging-window manifest for WebDataset EEG shards.")
    p.add_argument("--data-root", type=str, required=True, help="Dataset root, e.g. D:/open_eeg or /mnt/d/open_eeg")
    p.add_argument("--out-json", type=str, required=True, help="Output manifest JSON path")
    p.add_argument("--out-dir", type=str, default="", help="Optional directory for CSV summaries and per-window shard lists")
    p.add_argument("--path-style", type=str, default="scan", choices=["scan", "wsl", "windows"],
                   help="How shard paths are stored in the manifest")
    p.add_argument("--dataset-depth", type=int, default=1,
                   help="How many directory levels under data_root define dataset_id. Default=1 (first folder under root).")
    p.add_argument("--workers", type=int, default=max(1, min(4, os.cpu_count() or 1)),
                   help="Parallel tar inspection workers. On HDD/NTFS start small (e.g. 2).")
    p.add_argument("--strict", action="store_true", help="Fail immediately on unreadable tar instead of recording errors.")

    # Shape / token estimation
    p.add_argument("--default-fs", type=int, default=200, help="Fallback fs if meta.json does not provide one")
    p.add_argument("--patch-seconds", type=float, default=1.0, help="Patch size in seconds for token estimation")
    p.add_argument("--hop-seconds", type=float, default=1.0, help="Hop size in seconds for token estimation")
    p.add_argument("--max-tokens", type=int, default=0,
                   help="If >0, also estimate post-fit tokens using the same 'crop time to fit tokens' rule as pretraining")
    p.add_argument("--prefer-durations", type=str, default="10,30,60",
                   help="Preferred durations used when estimating fit-to-max_tokens behavior")

    # Staging / resident plan
    p.add_argument("--stage-budget-gb", type=float, default=300.0,
                   help="Total SSD budget reserved for one staged working set (resident + rotating)")
    p.add_argument("--resident-budget-gb", type=float, default=0.0,
                   help="SSD budget for always-resident rare groups. 0 disables resident shards.")
    p.add_argument("--resident-group-max-gb", type=float, default=5.0,
                   help="Candidate rare group threshold. Groups <= this size can become resident.")
    p.add_argument("--resident-group-max-shards", type=int, default=8,
                   help="Candidate rare group threshold by shard count.")
    p.add_argument("--rare-key", type=str, default="fit_shape_key",
                   choices=["fit_shape_key", "shape_key", "dataset_id", "dataset_shape_key", "dataset_fit_shape_key"],
                   help="What defines a 'rare group' for resident shards.")
    p.add_argument("--window-group-key", type=str, default="fit_shape_key",
                   choices=["fit_shape_key", "shape_key", "dataset_id", "dataset_shape_key", "dataset_fit_shape_key"],
                   help="What to spread evenly across windows.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--nominal-shard-size-gb", type=float, default=1.0,
                   help="Used only to label tail shards (<80%% of this size).")

    return p.parse_args()



def main() -> int:
    args = parse_args()

    root = resolve_scan_root(args.data_root)
    if not root.exists():
        raise FileNotFoundError(f"data root does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"data root is not a directory: {root}")

    out_json = Path(args.out_json).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else out_json.parent / (out_json.stem + "_artifacts")

    prefer_durations = [int(x.strip()) for x in args.prefer_durations.split(",") if x.strip()]
    if not prefer_durations:
        prefer_durations = [10, 30, 60]

    nominal_shard_size_bytes = int(round(float(args.nominal_shard_size_gb) * (1024 ** 3)))
    stage_budget_bytes = int(round(float(args.stage_budget_gb) * (1024 ** 3)))
    resident_budget_bytes = int(round(float(args.resident_budget_gb) * (1024 ** 3)))
    resident_group_max_bytes = int(round(float(args.resident_group_max_gb) * (1024 ** 3)))

    tar_paths = sorted(root.rglob("*.tar"))
    if not tar_paths:
        raise FileNotFoundError(f"no .tar shards found under: {root}")

    cfg = InspectConfig(
        root=str(root),
        output_root=str(root),
        path_style=args.path_style,
        default_fs=int(args.default_fs),
        patch_seconds=float(args.patch_seconds),
        hop_seconds=float(args.hop_seconds),
        max_tokens=int(args.max_tokens),
        prefer_durations_sec=prefer_durations,
        dataset_depth=int(args.dataset_depth),
        nominal_shard_size_bytes=nominal_shard_size_bytes,
        strict=bool(args.strict),
    )

    print(f"[scan] root={root}")
    print(f"[scan] found {len(tar_paths)} tar shards")
    print(f"[scan] workers={args.workers} patch={args.patch_seconds}s hop={args.hop_seconds}s max_tokens={args.max_tokens}")

    shards: List[Dict[str, Any]] = []
    if int(args.workers) <= 1:
        for i, tp in enumerate(tar_paths, start=1):
            rec = inspect_one_tar(str(tp), cfg)
            shards.append(rec)
            if (i % 50) == 0 or i == len(tar_paths):
                print(f"[scan] {i}/{len(tar_paths)}")
    else:
        with ThreadPoolExecutor(max_workers=int(args.workers)) as ex:
            futs = {ex.submit(inspect_one_tar, str(tp), cfg): tp for tp in tar_paths}
            done = 0
            for fut in as_completed(futs):
                shards.append(fut.result())
                done += 1
                if (done % 50) == 0 or done == len(tar_paths):
                    print(f"[scan] {done}/{len(tar_paths)}")
        shards.sort(key=lambda x: x["relative_path"])

    datasets = summarize_datasets(shards)

    resident_shards, rotating_shards, resident_groups = choose_resident_shards(
        shards,
        resident_budget_bytes=resident_budget_bytes,
        resident_group_max_bytes=resident_group_max_bytes,
        resident_group_max_shards=int(args.resident_group_max_shards),
        rare_key=str(args.rare_key),
    )

    resident_bytes = sum(int(x["size_bytes"]) for x in resident_shards)
    rotating_budget_bytes = stage_budget_bytes - resident_bytes
    if rotating_budget_bytes <= 0:
        raise ValueError(
            f"resident shards already consume {resident_bytes / (1024 ** 3):.3f} GB, exceeding stage budget {args.stage_budget_gb:.3f} GB"
        )

    windows = assign_windows(
        rotating_shards,
        window_target_bytes=rotating_budget_bytes,
        seed=int(args.seed),
        group_key=str(args.window_group_key),
    )

    total_bytes = sum(int(s["size_bytes"]) for s in shards)
    total_segments = sum(int(s.get("segment_count") or 0) for s in shards)
    total_errors = sum(1 for s in shards if s.get("error"))

    manifest = {
        "version": 1,
        "schema": "open_eeg_staging_manifest",
        "scan": {
            "data_root_input": args.data_root,
            "data_root_scan": str(root),
            "path_style": args.path_style,
            "num_shards": len(shards),
            "num_datasets": len(datasets),
            "total_bytes": int(total_bytes),
            "total_gb": round(total_bytes / (1024 ** 3), 6),
            "total_segments": int(total_segments),
            "num_error_shards": int(total_errors),
            "default_fs": int(args.default_fs),
            "patch_seconds": float(args.patch_seconds),
            "hop_seconds": float(args.hop_seconds),
            "max_tokens": int(args.max_tokens),
            "prefer_durations_sec": prefer_durations,
        },
        "staging": {
            "stage_budget_bytes": int(stage_budget_bytes),
            "stage_budget_gb": round(stage_budget_bytes / (1024 ** 3), 6),
            "resident_budget_bytes": int(resident_budget_bytes),
            "resident_budget_gb": round(resident_budget_bytes / (1024 ** 3), 6),
            "resident_actual_bytes": int(resident_bytes),
            "resident_actual_gb": round(resident_bytes / (1024 ** 3), 6),
            "rotating_window_target_bytes": int(rotating_budget_bytes),
            "rotating_window_target_gb": round(rotating_budget_bytes / (1024 ** 3), 6),
            "num_windows": len(windows),
            "rare_key": str(args.rare_key),
            "window_group_key": str(args.window_group_key),
            "resident_groups": resident_groups,
        },
        "datasets": datasets,
        "resident_shards": [s["shard_path"] for s in sorted(resident_shards, key=lambda x: x["relative_path"])],
        "windows": windows,
        "shards": shards,
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_rows = list(datasets.values())
    dataset_fields = [
        "dataset_id",
        "num_shards",
        "tail_shards",
        "total_bytes",
        "total_gb",
        "total_segments",
        "segments_per_gb",
        "channels_mode",
        "duration_mode_sec",
        "fs_mode",
        "fit_shape_mode",
        "channel_hist_by_segments",
        "duration_hist_by_segments",
        "fs_hist_by_segments",
        "raw_tokens_hist_by_segments",
        "fit_tokens_hist_by_segments",
        "fit_shape_hist_by_segments",
        "num_error_shards",
        "num_warnings",
    ]
    write_csv(out_dir / "dataset_summary.csv", dataset_rows, dataset_fields)

    shard_fields = [
        "dataset_id",
        "relative_path",
        "shard_path",
        "basename",
        "size_bytes",
        "size_gb",
        "is_tail_shard",
        "segment_count",
        "channels",
        "timepoints",
        "coords_channels",
        "coords_dims",
        "fs",
        "duration_sec",
        "duration_bucket_sec",
        "raw_patches",
        "raw_tokens",
        "fit_patches",
        "fit_tokens",
        "fit_duration_sec",
        "cropped_to_fit_tokens",
        "shape_key",
        "fit_shape_key",
        "dataset_shape_key",
        "dataset_fit_shape_key",
        "first_sample_key",
        "warnings",
        "error",
    ]
    write_csv(out_dir / "shard_summary.csv", shards, shard_fields)
    write_window_lists(out_dir / "window_lists", windows, manifest["resident_shards"])

    print(f"[done] manifest: {out_json}")
    print(f"[done] artifacts: {out_dir}")
    print(
        f"[done] datasets={len(datasets)} shards={len(shards)} total={total_bytes / (1024 ** 3):.2f} GB "
        f"resident={resident_bytes / (1024 ** 3):.2f} GB windows={len(windows)} rotating_target={rotating_budget_bytes / (1024 ** 3):.2f} GB"
    )
    if total_errors > 0:
        print(f"[warn] {total_errors} shard(s) had read/parse errors. Check shard_summary.csv / manifest['shards'].")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
