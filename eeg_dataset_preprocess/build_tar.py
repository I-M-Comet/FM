#!/usr/bin/env python3
"""
build_webdataset_tar.py

*_npy 폴더들의 eeg.npy, coords.npy, label.npy를
WebDataset용 .tar 파일로 변환합니다.

입력 구조:
  ROOT/
    Cho2017_npy/
      train/ eeg.npy, coords.npy, label.npy
      val/   ...
      test/  ...
    BCIC_IV_2a_npy/
      train/ ...
      ...

출력 구조:
  OUT_DIR/
    Cho2017_train.tar
    Cho2017_val.tar
    Cho2017_test.tar
    BCIC_IV_2a_train.tar
    ...

tar 내부 (WebDataset 형식):
  Cho2017_train_seg000000.eeg.npy
  Cho2017_train_seg000000.coords.npy
  Cho2017_train_seg000000.label.npy
  Cho2017_train_seg000001.eeg.npy
  ...

Usage:
  python build_webdataset_tar.py --root_dir D:/open_eeg_eval --out_dir D:/open_eeg_eval/webdataset
  python build_webdataset_tar.py --root_dir D:/open_eeg_eval  # out_dir defaults to root_dir/webdataset
"""

import argparse
import io
import os
import tarfile
import time
from pathlib import Path

import numpy as np

MAX_TAR_BYTES = 1 * 1024 * 1024 * 1024  # 1 GB

def _human_bytes(n: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    x = float(n)
    for u in units:
        if x < 1024.0 or u == units[-1]:
            return f"{x:.2f} {u}"
        x /= 1024.0
    return f"{x:.2f} TB"


def npy_to_bytes(arr: np.ndarray) -> bytes:
    """numpy 배열을 .npy 바이트로 직렬화"""
    buf = io.BytesIO()
    np.save(buf, arr)
    return buf.getvalue()


def add_npy_to_tar(tf: tarfile.TarFile, name: str, data: bytes):
    """tar에 .npy 바이트를 멤버로 추가"""
    info = tarfile.TarInfo(name=name)
    info.size = len(data)
    info.mtime = time.time()
    tf.addfile(info, io.BytesIO(data))


def process_split(dataset_name: str, split_dir: Path, out_dir: Path) -> int:
    """
    하나의 split 폴더 → .tar 파일 변환
    반환: 저장된 세그먼트 수
    """
    split_name = split_dir.name  # train / val / test

    eeg_path = split_dir / "eeg.npy"
    coords_path = split_dir / "coords.npy"
    label_path = split_dir / "label.npy"

    if not eeg_path.exists():
        print(f"  [SKIP] {split_dir}: eeg.npy not found")
        return 0

    # mmap으로 로드 (RAM 절약)
    eeg = np.load(eeg_path, mmap_mode="r")       # (N, C, T)
    coords = np.load(coords_path, mmap_mode="r")  # (N, C, 3)

    has_label = label_path.exists()
    label = np.load(label_path, mmap_mode="r") if has_label else None  # (N,)

    num_segments = eeg.shape[0]
    # 변경
    tar_idx = 0
    current_size = 0
    tar_path = out_dir / f"{dataset_name}_{split_name}_{tar_idx:04d}.tar"
    tf = tarfile.open(tar_path, "w")

    for i in range(num_segments):
        prefix = f"{dataset_name}_{split_name}_seg{i:06d}"

        eeg_bytes = npy_to_bytes(np.array(eeg[i]))
        coords_bytes = npy_to_bytes(np.array(coords[i]))
        label_bytes = npy_to_bytes(np.array(label[i])) if has_label else b""

        sample_size = len(eeg_bytes) + len(coords_bytes) + len(label_bytes)

        # 현재 tar가 한계를 넘으면 닫고 새 tar 생성
        if current_size > 0 and current_size + sample_size > MAX_TAR_BYTES:
            tf.close()
            print(f"  [{split_name}] shard {tar_idx:04d}: {_human_bytes(tar_path.stat().st_size)}")
            tar_idx += 1
            tar_path = out_dir / f"{dataset_name}_{split_name}_{tar_idx:04d}.tar"
            tf = tarfile.open(tar_path, "w")
            current_size = 0

        add_npy_to_tar(tf, f"{prefix}.eeg.npy", eeg_bytes)
        add_npy_to_tar(tf, f"{prefix}.coords.npy", coords_bytes)
        if has_label:
            add_npy_to_tar(tf, f"{prefix}.label.npy", label_bytes)

        current_size += sample_size

    tf.close()
    print(f"  [{split_name}] shard {tar_idx:04d}: {_human_bytes(tar_path.stat().st_size)}")

    return num_segments


def main():
    ap = argparse.ArgumentParser(description="Convert *_npy folders to WebDataset .tar files")
    ap.add_argument("--root_dir", type=str, required=True,
                    help="Top-level directory containing *_npy folders")
    ap.add_argument("--out_dir", type=str, default=None,
                    help="Output directory for .tar files (default: root_dir/webdataset)")
    args = ap.parse_args()

    root_dir = Path(args.root_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else root_dir / "webdataset"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[config] root_dir : {root_dir}")
    print(f"[config] out_dir  : {out_dir}\n")

    # *_npy 폴더 탐색
    npy_dirs = sorted([d for d in root_dir.iterdir() if d.is_dir() and d.name.endswith("_npy")])

    if not npy_dirs:
        print(f"[ERROR] No *_npy directories found under {root_dir}")
        return

    print(f"[init] Found {len(npy_dirs)} dataset(s): {[d.name for d in npy_dirs]}\n")

    grand_total = 0

    for npy_dir in npy_dirs:
        # 데이터셋 이름: "Cho2017_npy" → "Cho2017"
        dataset_name = npy_dir.name.rsplit("_npy", 1)[0]
        print(f"[{dataset_name}]")

        dataset_total = 0
        for split in ["train", "val", "test"]:
            split_dir = npy_dir / split
            if split_dir.is_dir():
                dataset_total += process_split(dataset_name, split_dir, out_dir)

        print(f"  → {dataset_name} total: {dataset_total} segments\n")
        grand_total += dataset_total

    print("=" * 60)
    print(f"[Done] {len(npy_dirs)} dataset(s), {grand_total} total segments")
    print(f"[Done] tar files saved to: {out_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()