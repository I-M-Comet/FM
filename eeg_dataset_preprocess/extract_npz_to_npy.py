#!/usr/bin/env python3
"""
extract_npz_to_npy.py

Memory-safe extraction of large .npz datasets into .npy files for fast memmap-based evaluation.

Why this exists:
  - np.load("train.npz") will decompress whole arrays into RAM (no mmap), which can OOM/Swap-thrash for large files.
  - An .npz is a ZIP that stores each array as a (possibly compressed) .npy member.
  - We can stream-copy the .npy member out of the ZIP without ever materializing the array in RAM.

Output layout (recommended):
  OUT_DIR/
    train/
      eeg.npy
      coords.npy
      label.npy
      meta.json
    val/
      ...
    test/
      ...

Usage:
  python extract_npz_to_npy.py --in_dir /path/TUAB --out_dir /ssd/cache/TUAB
  python extract_npz_to_npy.py --in_dir /path/ISRUC --out_dir /ssd/cache/ISRUC --splits train val test
  python extract_npz_to_npy.py --in_dir D:\open_eeg_eval\PhysioNetMI_npz --out_dir D:\open_eeg_eval\PhysioNetMI_npy --splits train val test

Notes:
  - This script does NOT change dtype/values; it copies exactly the stored .npy bytes.
  - coords key can be either "coords" or "coord"; output is always "coords.npy".
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


def _human_bytes(n: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    x = float(n)
    for u in units:
        if x < 1024.0 or u == units[-1]:
            return f"{x:.2f}{u}"
        x /= 1024.0
    return f"{x:.2f}TB"


def _list_npz_members(npz_path: Path) -> List[str]:
    with zipfile.ZipFile(npz_path, "r") as zf:
        return zf.namelist()


def _find_member(zf: zipfile.ZipFile, candidates: List[str]) -> Optional[str]:
    names = set(zf.namelist())
    for c in candidates:
        if c in names:
            return c
    # Sometimes saved with directories, try basename match
    base_map = {Path(n).name: n for n in zf.namelist()}
    for c in candidates:
        if c in base_map:
            return base_map[c]
    return None


def _atomic_copy_from_zip(zf: zipfile.ZipFile, member: str, out_path: Path, chunk_bytes: int = 16 << 20) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + f".tmp.{os.getpid()}")
    if tmp.exists():
        try:
            tmp.unlink()
        except Exception:
            pass
    with zf.open(member, "r") as src, open(tmp, "wb") as dst:
        shutil.copyfileobj(src, dst, length=chunk_bytes)
    os.replace(tmp, out_path)


def _read_npy_header(npy_path: Path) -> Dict[str, object]:
    # mmap_mode='r' does not load array into RAM, only header
    arr = np.load(npy_path, mmap_mode="r")
    return {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "fortran_order": bool(arr.flags["F_CONTIGUOUS"]) and not bool(arr.flags["C_CONTIGUOUS"]),
    }


def extract_split(
    npz_path: Path,
    out_split_dir: Path,
    force: bool,
    chunk_mb: int,
) -> Dict[str, object]:
    out_split_dir.mkdir(parents=True, exist_ok=True)
    meta: Dict[str, object] = {
        "source_npz": str(npz_path),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "members": {},
        "arrays": {},
        "files": {},
    }

    with zipfile.ZipFile(npz_path, "r") as zf:
        # Keys: eeg, coords/coord, label/labels/y
        eeg_member = _find_member(zf, ["eeg.npy", "eeg", "arr_0.npy"])
        coord_member = _find_member(zf, ["coords.npy", "coord.npy", "coords", "coord", "arr_1.npy"])
        label_member = _find_member(zf, ["label.npy", "labels.npy", "y.npy", "label", "labels", "y", "arr_2.npy"])

        if eeg_member is None or coord_member is None or label_member is None:
            raise KeyError(
                f"Could not find required members in {npz_path}. "
                f"Need eeg/coord/label. Available={zf.namelist()[:20]}..."
            )

        mapping: List[Tuple[str, str, Path]] = [
            ("eeg", eeg_member, out_split_dir / "eeg.npy"),
            ("coords", coord_member, out_split_dir / "coords.npy"),
            ("label", label_member, out_split_dir / "label.npy"),
        ]

        for logical_key, member, out_file in mapping:
            meta["members"][logical_key] = member
            if out_file.exists() and not force:
                pass
            else:
                _atomic_copy_from_zip(zf, member, out_file, chunk_bytes=max(1, chunk_mb) << 20)

            try:
                st = out_file.stat()
                meta["files"][logical_key] = {"path": str(out_file), "size_bytes": int(st.st_size), "size_h": _human_bytes(st.st_size)}
            except Exception:
                pass

            # Read header only (mmap)
            try:
                meta["arrays"][logical_key] = _read_npy_header(out_file)
            except Exception as e:
                meta["arrays"][logical_key] = {"error": repr(e)}

    # write meta.json
    with open(out_split_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    return meta


def auto_splits(in_dir: Path) -> List[str]:
    splits = []
    for s in ["train", "val", "test"]:
        if (in_dir / f"{s}.npz").exists():
            splits.append(s)
    if not splits:
        raise FileNotFoundError(f"No train/val/test .npz found under {in_dir}")
    return splits


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", type=str, required=True, help="Directory that contains train.npz/val.npz/test.npz")
    ap.add_argument("--out_dir", type=str, required=True, help="Output directory for extracted .npy files")
    ap.add_argument("--splits", type=str, nargs="*", default=None, help="Splits to extract: train val test. Default: auto-detect")
    ap.add_argument("--force", action="store_true", help="Overwrite existing .npy files")
    ap.add_argument("--chunk_mb", type=int, default=16, help="Copy chunk size in MB (streaming).")
    args = ap.parse_args()

    in_dir = Path(args.in_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = args.splits if args.splits else auto_splits(in_dir)

    print(f"[extract] in_dir={in_dir}")
    print(f"[extract] out_dir={out_dir}")
    print(f"[extract] splits={splits}")
    print(f"[extract] force={args.force} chunk_mb={args.chunk_mb}")

    for s in splits:
        npz_path = in_dir / f"{s}.npz"
        if not npz_path.exists():
            raise FileNotFoundError(npz_path)
        out_split_dir = out_dir / s
        print(f"\n[extract] split={s} npz={npz_path.name} -> {out_split_dir}")
        meta = extract_split(npz_path=npz_path, out_split_dir=out_split_dir, force=args.force, chunk_mb=args.chunk_mb)

        # quick summary
        eeg_sz = meta.get("files", {}).get("eeg", {}).get("size_h", "?")
        coord_sz = meta.get("files", {}).get("coords", {}).get("size_h", "?")
        lab_sz = meta.get("files", {}).get("label", {}).get("size_h", "?")
        eeg_shape = meta.get("arrays", {}).get("eeg", {}).get("shape", "?")
        print(f"  - eeg   : {eeg_sz} shape={eeg_shape}")
        print(f"  - coords: {coord_sz}")
        print(f"  - label : {lab_sz}")

    print("\n[extract] done.")


if __name__ == "__main__":
    main()
