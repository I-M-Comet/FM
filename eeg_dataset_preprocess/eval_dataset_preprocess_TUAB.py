#!/usr/bin/env python3
"""
preprocess_tuab_to_npy_bipolar.py

Direct TUAB preprocessing to memory-mappable .npy files using the common
16-channel bipolar montage protocol used by BIOT/CBraMod-style downstream
experiments.

Expected source layout
----------------------
ROOT_DIR/
  train/
    abnormal/**/*.edf
    normal/**/*.edf
  eval/
    abnormal/**/*.edf
    normal/**/*.edf

Output layout
-------------
OUT_DIR/
  train/
    eeg.npy      (N, 16, 2000) for 10-second windows at 200 Hz
    coords.npy   (N, 16, 3)
    label.npy    (N,)   normal=0, abnormal=1
    meta.json
  val/
    ...
  test/
    ...
  dataset_meta.json
  scan_report.json

Notes
-----
- TUAB EDF channels are referential/unipolar labels such as EEG FP1-REF.
  This script reconstructs the standard 16 bipolar derivations by subtraction.
- Segmentation is non-overlapping 10-second windows. Any trailing remainder
  shorter than 10 seconds is discarded.
- Each segment inherits the recording-level label from its folder:
    normal -> 0
    abnormal -> 1
- By default, per-window z-score normalization is enabled.
# python ./eeg_dataset_preprocess/eval_dataset_preprocess_TUAB.py --root_dir "D:\One_한양대학교\private object minsu\coding\data\TUAB\v3.0.1\edf" --out_dir "D:/open_eeg_eval/TUAB_npy"
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import warnings
from collections import Counter
from fractions import Fraction
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import mne
import numpy as np
import scipy.signal as signal
from numpy.lib.format import open_memmap
from tqdm import tqdm

# -----------------------------------------------------------------------------
# Label definitions
# -----------------------------------------------------------------------------
LABEL_NAME_TO_INDEX = {
    "normal": 0,
    "abnormal": 1,
}
INDEX_TO_LABEL_NAME = {v: k.upper() for k, v in LABEL_NAME_TO_INDEX.items()}

# -----------------------------------------------------------------------------
# 16 bipolar derivations (legacy TUH naming with T3/T4/T5/T6)
# -----------------------------------------------------------------------------
REFERENTIAL_ORDER = [
    "FP1", "F7", "T3", "T5", "O1",
    "FP2", "F8", "T4", "T6", "O2",
    "F3", "C3", "P3", "F4", "C4", "P4",
]
REFERENTIAL_SET = set(REFERENTIAL_ORDER)

BIPOLAR_PAIRS: List[Tuple[str, str]] = [
    ("FP1", "F7"),
    ("F7", "T3"),
    ("T3", "T5"),
    ("T5", "O1"),
    ("FP2", "F8"),
    ("F8", "T4"),
    ("T4", "T6"),
    ("T6", "O2"),
    ("FP1", "F3"),
    ("F3", "C3"),
    ("C3", "P3"),
    ("P3", "O1"),
    ("FP2", "F4"),
    ("F4", "C4"),
    ("C4", "P4"),
    ("P4", "O2"),
]
BIPOLAR_NAMES = [f"{a}-{b}" for a, b in BIPOLAR_PAIRS]

# Internal referential name -> MNE template montage channel name.
MONTAGE_NAME_MAP = {
    "FP1": "Fp1",
    "FP2": "Fp2",
    "F3": "F3",
    "F4": "F4",
    "C3": "C3",
    "C4": "C4",
    "P3": "P3",
    "P4": "P4",
    "O1": "O1",
    "O2": "O2",
    "F7": "F7",
    "F8": "F8",
    "T3": "T7",
    "T4": "T8",
    "T5": "P7",
    "T6": "P8",
}

# Accept either modern or legacy spellings from EDF.
REFERENTIAL_ALIAS_MAP = {
    "FP1": "FP1",
    "FP2": "FP2",
    "F3": "F3",
    "F4": "F4",
    "C3": "C3",
    "C4": "C4",
    "P3": "P3",
    "P4": "P4",
    "O1": "O1",
    "O2": "O2",
    "F7": "F7",
    "F8": "F8",
    "T3": "T3",
    "T4": "T4",
    "T5": "T5",
    "T6": "T6",
    "T7": "T3",
    "T8": "T4",
    "P7": "T5",
    "P8": "T6",
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="TUAB -> direct NPY preprocessing with 16 bipolar channels")
    ap.add_argument("--root_dir", type=str, required=True, help="TUAB edf root that contains train/ and eval/")
    ap.add_argument("--out_dir", type=str, required=True, help="Output directory")
    ap.add_argument("--target_sr", type=int, default=200, help="Target sampling rate")
    ap.add_argument("--bandpass_low", type=float, default=0.3, help="Band-pass low cutoff (Hz)")
    ap.add_argument("--bandpass_high", type=float, default=75.0, help="Band-pass high cutoff (Hz)")
    ap.add_argument("--notch_freq", type=float, default=60.0, help="Notch frequency (Hz). Use <=0 to disable")
    ap.add_argument("--notch_q", type=float, default=30.0, help="Notch Q factor")
    ap.add_argument("--window_seconds", type=float, default=10.0, help="Segment length in seconds")
    ap.add_argument("--val_ratio", type=float, default=0.2, help="Validation split ratio from train, using CBraMod-style sorted subject IDs")
    ap.add_argument("--seed_text", type=str, default="ignored_for_cbramod_split", help="Ignored; kept only for backward CLI compatibility")
    ap.add_argument(
        "--eeg_dtype",
        type=str,
        default="float32",
        choices=["float32", "float16"],
        help="Output dtype for eeg.npy",
    )
    ap.add_argument(
        "--coords_dtype",
        type=str,
        default="float32",
        choices=["float32", "float16"],
        help="Output dtype for coords.npy",
    )
    ap.set_defaults(normalize_per_window=True)
    ap.add_argument(
        "--normalize_per_window",
        dest="normalize_per_window",
        action="store_true",
        help="Enable per-window z-score after extraction (default: enabled)",
    )
    ap.add_argument(
        "--no_normalize_per_window",
        dest="normalize_per_window",
        action="store_false",
        help="Disable per-window z-score",
    )
    ap.add_argument(
        "--disable_val_split",
        action="store_true",
        help="Do not create val; use original train as train and eval as test",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files",
    )
    ap.add_argument(
        "--montage",
        type=str,
        default="standard_1005",
        help="MNE template montage used to derive bipolar coordinates",
    )
    return ap.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def remove_if_overwrite(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists: {path}. Use --overwrite to replace it.")
        if path.is_file():
            path.unlink()


def list_edf_files(split_dir: Path) -> List[Path]:
    if not split_dir.exists():
        return []
    return sorted([p for p in split_dir.rglob("*") if p.is_file() and p.suffix.lower() == ".edf"])


def canonicalize_ref_name(name: str) -> Optional[str]:
    s = name.upper().strip()
    s = s.replace(".", "")
    s = re.sub(r"^EEG\s*", "", s)
    s = s.replace("EEG", "")
    s = s.replace("POL ", "")
    s = s.replace("POL", "")
    s = s.strip()

    s = re.sub(r"[\s\-_]*(REF|LE|AR|AVG|M1|M2|EAR|LINKEDEARS)$", "", s)
    s = s.replace(" ", "").replace("_", "").replace("-", "")

    s = REFERENTIAL_ALIAS_MAP.get(s, s)
    return s if s in REFERENTIAL_SET else None


def build_ref_index_map(ch_names: Sequence[str]) -> Dict[str, int]:
    idx_map: Dict[str, int] = {}
    for idx, ch in enumerate(ch_names):
        canon = canonicalize_ref_name(ch)
        if canon is not None and canon not in idx_map:
            idx_map[canon] = idx
    return idx_map


def load_raw_edf(file_path: Path, preload: bool) -> mne.io.BaseRaw:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = mne.io.read_raw_edf(str(file_path), preload=preload, verbose=False)
    return raw


def infer_tuab_label(edf_path: Path) -> int:
    parts = {part.lower() for part in edf_path.parts}
    if "abnormal" in parts:
        return LABEL_NAME_TO_INDEX["abnormal"]
    if "normal" in parts:
        return LABEL_NAME_TO_INDEX["normal"]
    raise ValueError(f"Could not infer TUAB label from path: {edf_path}")


def subject_id_from_tuab_path(edf_path: Path) -> str:
    """
    TUAB file names are usually subject/session/trial style, e.g.
      aaaaaguk_s002_t001.edf
    We use the subject prefix before _s### when available.
    If that pattern is missing, fall back to the stem before the first underscore,
    and finally to the whole stem.
    """
    stem = edf_path.stem
    m = re.match(r"^(.*?)(?:_s\d+.*)$", stem, flags=re.IGNORECASE)
    if m and m.group(1):
        return m.group(1)
    if "_" in stem:
        left = stem.split("_", 1)[0]
        if left:
            return left
    return stem



def deterministic_subject_split(
    subject_ids: Sequence[str],
    val_ratio: float,
    seed_text: str,
) -> Tuple[set[str], set[str]]:
    """
    CBraMod-style deterministic split:
      1) unique subject IDs
      2) lexical sort
      3) first floor((1-val_ratio) * N) -> train
      4) remaining subjects -> val

    seed_text is intentionally ignored and only retained for CLI compatibility.
    """
    del seed_text

    subjects = sorted(set(subject_ids))
    if len(subjects) == 0:
        return set(), set()
    if val_ratio <= 0.0 or len(subjects) == 1:
        return set(subjects), set()

    n_train = int(len(subjects) * (1.0 - float(val_ratio)))
    n_train = min(len(subjects) - 1, max(1, n_train))

    train_subjects = set(subjects[:n_train])
    val_subjects = set(subjects[n_train:])
    return train_subjects, val_subjects


def build_classwise_train_val_subject_sets(
    train_edfs: Sequence[Path],
    val_ratio: float,
    seed_text: str,
) -> Dict[int, Dict[str, set[str]]]:
    """
    TUAB CBraMod-style split:
      - abnormal training subjects are split independently
      - normal training subjects are split independently
      - each label-specific subject list is lexically sorted before the cut
    """
    by_label: Dict[int, List[str]] = {
        LABEL_NAME_TO_INDEX["normal"]: [],
        LABEL_NAME_TO_INDEX["abnormal"]: [],
    }
    for edf_path in train_edfs:
        label_idx = infer_tuab_label(edf_path)
        by_label[label_idx].append(subject_id_from_tuab_path(edf_path))

    out: Dict[int, Dict[str, set[str]]] = {}
    for label_idx, subject_ids in by_label.items():
        train_subjects, val_subjects = deterministic_subject_split(
            subject_ids=subject_ids,
            val_ratio=val_ratio,
            seed_text=seed_text,
        )
        out[label_idx] = {"train": train_subjects, "val": val_subjects}
    return out


def compute_bipolar_coords(montage_name: str, coords_dtype: np.dtype) -> np.ndarray:
    montage = mne.channels.make_standard_montage(montage_name)
    ch_pos = montage.get_positions()["ch_pos"]

    coords = []
    for a, b in BIPOLAR_PAIRS:
        aa = MONTAGE_NAME_MAP[a]
        bb = MONTAGE_NAME_MAP[b]
        if aa not in ch_pos or bb not in ch_pos:
            raise KeyError(f"Missing montage positions for {aa} or {bb} in {montage_name}")
        coords.append((np.asarray(ch_pos[aa]) + np.asarray(ch_pos[bb])) / 2.0)
    return np.asarray(coords, dtype=coords_dtype)



def resample_factors(orig_sfreq: float, target_sfreq: int) -> Tuple[int, int]:
    frac = Fraction(float(target_sfreq) / float(orig_sfreq)).limit_denominator(1000)
    return frac.numerator, frac.denominator



def expected_resampled_n_times(raw_n_times: int, orig_sfreq: float, target_sfreq: int) -> int:
    if abs(float(orig_sfreq) - float(target_sfreq)) <= 1e-9:
        return int(raw_n_times)
    up, down = resample_factors(orig_sfreq, target_sfreq)
    return int(math.ceil(int(raw_n_times) * up / down))



def preprocess_continuous_bipolar(
    data: np.ndarray,
    orig_sfreq: float,
    target_sfreq: int,
    bandpass_low: float,
    bandpass_high: float,
    notch_freq: float,
    notch_q: float,
) -> np.ndarray:
    x = np.asarray(data, dtype=np.float64)
    nyq = 0.5 * float(orig_sfreq)

    if notch_freq > 0 and notch_freq < nyq:
        b_notch, a_notch = signal.iirnotch(float(notch_freq), float(notch_q), fs=float(orig_sfreq))
        x = signal.filtfilt(b_notch, a_notch, x, axis=-1)

    high = min(float(bandpass_high), nyq - 1e-3)
    low = float(bandpass_low)
    if high <= low:
        raise ValueError(f"Invalid band-pass after nyquist adjustment: low={low}, high={high}, fs={orig_sfreq}")

    sos = signal.butter(4, [low / nyq, high / nyq], btype="bandpass", output="sos")
    x = signal.sosfiltfilt(sos, x, axis=-1)

    if abs(float(orig_sfreq) - float(target_sfreq)) > 1e-9:
        up, down = resample_factors(orig_sfreq, target_sfreq)
        x = signal.resample_poly(x, up, down, axis=-1)

    return np.asarray(x, dtype=np.float32)



def build_bipolar_signals(raw: mne.io.BaseRaw) -> Tuple[np.ndarray, float]:
    idx_map = build_ref_index_map(raw.ch_names)
    missing = [ch for ch in REFERENTIAL_ORDER if ch not in idx_map]
    if missing:
        raise ValueError(f"Missing required channels: {missing}")

    picks = [idx_map[ch] for ch in REFERENTIAL_ORDER]
    ref_data = raw.get_data(picks=picks).astype(np.float32, copy=False)
    ref_name_to_row = {ch: i for i, ch in enumerate(REFERENTIAL_ORDER)}

    bipolar = []
    for a, b in BIPOLAR_PAIRS:
        bipolar.append(ref_data[ref_name_to_row[a]] - ref_data[ref_name_to_row[b]])
    bipolar_data = np.stack(bipolar, axis=0)
    return bipolar_data, float(raw.info["sfreq"])



def zscore_per_window(x: np.ndarray) -> np.ndarray:
    mean = x.mean(axis=-1, keepdims=True)
    std = x.std(axis=-1, keepdims=True)
    return (x - mean) / (std + 1e-8)



def human_bytes(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    x = float(num_bytes)
    for u in units:
        if x < 1024.0 or u == units[-1]:
            return f"{x:.2f}{u}"
        x /= 1024.0
    return f"{x:.2f}TB"



def to_jsonable_counter(counter: Counter) -> Dict[str, int]:
    return {INDEX_TO_LABEL_NAME[int(k)]: int(v) for k, v in sorted(counter.items(), key=lambda kv: int(kv[0]))}



def scan_dataset(
    root_dir: Path,
    target_sr: int,
    window_seconds: float,
    val_ratio: float,
    seed_text: str,
    disable_val_split: bool,
) -> Tuple[Dict[str, List[Path]], Dict[str, dict], Dict[str, set[str]], dict]:
    train_dir = root_dir / "train"
    eval_dir = root_dir / "eval"

    train_edfs = list_edf_files(train_dir)
    eval_edfs = list_edf_files(eval_dir)

    if disable_val_split:
        label_subject_sets: Dict[int, Dict[str, set[str]]] = {
            LABEL_NAME_TO_INDEX["normal"]: {
                "train": {subject_id_from_tuab_path(p) for p in train_edfs if infer_tuab_label(p) == LABEL_NAME_TO_INDEX["normal"]},
                "val": set(),
            },
            LABEL_NAME_TO_INDEX["abnormal"]: {
                "train": {subject_id_from_tuab_path(p) for p in train_edfs if infer_tuab_label(p) == LABEL_NAME_TO_INDEX["abnormal"]},
                "val": set(),
            },
        }
    else:
        label_subject_sets = build_classwise_train_val_subject_sets(
            train_edfs=train_edfs,
            val_ratio=val_ratio,
            seed_text=seed_text,
        )

    split_to_files: Dict[str, List[Path]] = {"train": [], "val": [], "test": []}
    split_to_subjects: Dict[str, set[str]] = {"train": set(), "val": set(), "test": set()}
    file_info: Dict[str, dict] = {}
    summary = {
        "n_train_edf_found": len(train_edfs),
        "n_eval_edf_found": len(eval_edfs),
        "n_files_scanned": 0,
        "n_files_kept": 0,
        "n_files_skipped": 0,
        "skip_reasons": Counter(),
    }

    window_len = int(round(window_seconds * target_sr))
    all_files = [(p, "train_src") for p in train_edfs] + [(p, "eval_src") for p in eval_edfs]

    for edf_path, src_split in tqdm(all_files, desc="Scanning EDF"):
        summary["n_files_scanned"] += 1
        try:
            label_idx = infer_tuab_label(edf_path)
            raw = load_raw_edf(edf_path, preload=False)
            raw_n_times = int(raw.n_times)
            raw_sfreq = float(raw.info["sfreq"])
            duration_sec = float(raw_n_times / raw_sfreq)
            idx_map = build_ref_index_map(raw.ch_names)
            missing = [ch for ch in REFERENTIAL_ORDER if ch not in idx_map]
            if hasattr(raw, "close"):
                raw.close()
        except Exception as e:
            summary["n_files_skipped"] += 1
            summary["skip_reasons"][f"edf_read_error: {type(e).__name__}"] += 1
            file_info[str(edf_path)] = {
                "kept": False,
                "reason": f"edf_read_error: {type(e).__name__}",
            }
            continue

        if missing:
            summary["n_files_skipped"] += 1
            summary["skip_reasons"]["missing_required_channels"] += 1
            file_info[str(edf_path)] = {
                "kept": False,
                "reason": "missing_required_channels",
                "missing_channels": missing,
            }
            continue

        resampled_n = expected_resampled_n_times(raw_n_times, raw_sfreq, target_sr)
        n_segments = int(resampled_n // window_len)
        if n_segments <= 0:
            summary["n_files_skipped"] += 1
            summary["skip_reasons"]["too_short_after_resample"] += 1
            file_info[str(edf_path)] = {
                "kept": False,
                "reason": "too_short_after_resample",
                "duration_sec": duration_sec,
                "raw_sfreq": raw_sfreq,
                "raw_n_times": raw_n_times,
                "resampled_n_times": resampled_n,
            }
            continue

        subject_id = subject_id_from_tuab_path(edf_path)
        if src_split == "train_src":
            label_sets = label_subject_sets[label_idx]
            if subject_id in label_sets["train"]:
                out_split = "train"
            elif subject_id in label_sets["val"]:
                out_split = "val"
            else:
                raise RuntimeError(
                    f"Subject {subject_id} with label {INDEX_TO_LABEL_NAME[label_idx]} was not assigned to train/val. "
                    "Check the class-wise split logic."
                )
        else:
            out_split = "test"

        split_to_files[out_split].append(edf_path)
        split_to_subjects[out_split].add(subject_id)
        summary["n_files_kept"] += 1

        file_info[str(edf_path)] = {
            "kept": True,
            "source_split": src_split,
            "output_split": out_split,
            "subject_id": subject_id,
            "label_idx": int(label_idx),
            "label_name": INDEX_TO_LABEL_NAME[label_idx],
            "duration_sec": duration_sec,
            "raw_sfreq": raw_sfreq,
            "raw_n_times": raw_n_times,
            "resampled_n_times": resampled_n,
            "n_segments": n_segments,
        }

    return split_to_files, file_info, split_to_subjects, summary



def prepare_output_split(
    split_dir: Path,
    n_samples: int,
    eeg_shape_tail: Tuple[int, int],
    eeg_dtype: np.dtype,
    coords_dtype: np.dtype,
    overwrite: bool,
) -> Tuple[np.memmap, np.memmap, np.memmap]:
    ensure_dir(split_dir)

    eeg_path = split_dir / "eeg.npy"
    coords_path = split_dir / "coords.npy"
    label_path = split_dir / "label.npy"

    for p in [eeg_path, coords_path, label_path, split_dir / "meta.json"]:
        remove_if_overwrite(p, overwrite)

    eeg_mm = open_memmap(eeg_path, mode="w+", dtype=eeg_dtype, shape=(n_samples, *eeg_shape_tail))
    coords_mm = open_memmap(coords_path, mode="w+", dtype=coords_dtype, shape=(n_samples, eeg_shape_tail[0], 3))
    label_mm = open_memmap(label_path, mode="w+", dtype=np.int8, shape=(n_samples,))
    return eeg_mm, coords_mm, label_mm



def count_split_samples(split_files: Sequence[Path], file_info: Dict[str, dict]) -> int:
    return int(sum(int(file_info[str(p)]["n_segments"]) for p in split_files))



def write_split_arrays(
    split_name: str,
    split_files: Sequence[Path],
    file_info: Dict[str, dict],
    out_dir: Path,
    coords_template: np.ndarray,
    target_sr: int,
    bandpass_low: float,
    bandpass_high: float,
    notch_freq: float,
    notch_q: float,
    window_seconds: float,
    normalize_per_window: bool,
    eeg_dtype: np.dtype,
    coords_dtype: np.dtype,
    overwrite: bool,
) -> dict:
    n_samples = count_split_samples(split_files, file_info)
    n_chans = len(BIPOLAR_PAIRS)
    window_len = int(round(window_seconds * target_sr))

    split_dir = out_dir / split_name
    eeg_mm, coords_mm, label_mm = prepare_output_split(
        split_dir=split_dir,
        n_samples=n_samples,
        eeg_shape_tail=(n_chans, window_len),
        eeg_dtype=eeg_dtype,
        coords_dtype=coords_dtype,
        overwrite=overwrite,
    )

    write_idx = 0
    class_counts = Counter()
    n_files = 0

    for edf_path in tqdm(split_files, desc=f"Writing {split_name}"):
        info = file_info[str(edf_path)]
        label_idx = int(info["label_idx"])
        expected_segments = int(info["n_segments"])

        try:
            raw = load_raw_edf(edf_path, preload=True)
            bipolar_data, orig_sfreq = build_bipolar_signals(raw)
            if hasattr(raw, "close"):
                raw.close()
        except Exception as e:
            raise RuntimeError(f"Failed while loading/assembling bipolar data for {edf_path}: {e}") from e

        try:
            processed = preprocess_continuous_bipolar(
                data=bipolar_data,
                orig_sfreq=orig_sfreq,
                target_sfreq=target_sr,
                bandpass_low=bandpass_low,
                bandpass_high=bandpass_high,
                notch_freq=notch_freq,
                notch_q=notch_q,
            )
        except Exception as e:
            raise RuntimeError(f"Failed preprocessing {edf_path}: {e}") from e

        n_segments = int(processed.shape[-1] // window_len)
        if n_segments != expected_segments:
            raise RuntimeError(
                f"Segment count mismatch for {edf_path}: expected {expected_segments}, got {n_segments}. "
                f"processed_n_times={processed.shape[-1]}, window_len={window_len}"
            )

        for seg_idx in range(n_segments):
            start = seg_idx * window_len
            stop = start + window_len
            window = processed[:, start:stop]
            if normalize_per_window:
                window = zscore_per_window(window)

            eeg_mm[write_idx] = window.astype(eeg_dtype, copy=False)
            coords_mm[write_idx] = coords_template.astype(coords_dtype, copy=False)
            label_mm[write_idx] = np.int8(label_idx)

            class_counts[label_idx] += 1
            write_idx += 1

        n_files += 1

    if write_idx != n_samples:
        raise RuntimeError(f"Split {split_name}: wrote {write_idx} samples but allocated {n_samples}")

    eeg_mm.flush()
    coords_mm.flush()
    label_mm.flush()

    meta = {
        "split": split_name,
        "n_files": n_files,
        "n_samples": n_samples,
        "sample_construction": f"non-overlapping fixed {window_seconds:g}-second full 16-bipolar windows from each recording",
        "segment_label_rule": "each window inherits the recording-level folder label",
        "label_map": {str(i): INDEX_TO_LABEL_NAME[i] for i in range(len(INDEX_TO_LABEL_NAME))},
        "channels": BIPOLAR_NAMES,
        "eeg": {
            "path": str((split_dir / "eeg.npy").resolve()),
            "dtype": str(np.dtype(eeg_dtype)),
            "shape": [n_samples, n_chans, window_len],
            "size_bytes": int((split_dir / "eeg.npy").stat().st_size),
            "size_h": human_bytes((split_dir / "eeg.npy").stat().st_size),
        },
        "coords": {
            "path": str((split_dir / "coords.npy").resolve()),
            "dtype": str(np.dtype(coords_dtype)),
            "shape": [n_samples, n_chans, 3],
            "size_bytes": int((split_dir / "coords.npy").stat().st_size),
            "size_h": human_bytes((split_dir / "coords.npy").stat().st_size),
        },
        "label": {
            "path": str((split_dir / "label.npy").resolve()),
            "dtype": "int8",
            "shape": [n_samples],
            "size_bytes": int((split_dir / "label.npy").stat().st_size),
            "size_h": human_bytes((split_dir / "label.npy").stat().st_size),
        },
        "class_counts": to_jsonable_counter(class_counts),
    }

    with open(split_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    del eeg_mm, coords_mm, label_mm
    return meta



def main() -> None:
    args = parse_args()
    root_dir = Path(args.root_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    ensure_dir(out_dir)

    if not (root_dir / "train").exists() or not (root_dir / "eval").exists():
        raise FileNotFoundError(f"Expected {root_dir}/train and {root_dir}/eval to exist")

    eeg_dtype = np.dtype(args.eeg_dtype)
    coords_dtype = np.dtype(args.coords_dtype)
    coords_template = compute_bipolar_coords(args.montage, coords_dtype=coords_dtype)

    split_to_files, file_info, split_to_subjects, scan_summary = scan_dataset(
        root_dir=root_dir,
        target_sr=args.target_sr,
        window_seconds=args.window_seconds,
        val_ratio=args.val_ratio,
        seed_text=args.seed_text,
        disable_val_split=args.disable_val_split,
    )

    dataset_meta = {
        "dataset": "TUAB",
        "root_dir": str(root_dir),
        "output_dir": str(out_dir),
        "task": "binary_abnormal_detection",
        "input_channel_kind": "referential EDF channels",
        "derived_channel_kind": "16 bipolar derivations",
        "window_seconds": float(args.window_seconds),
        "window_mode": "non_overlapping",
        "target_sr": int(args.target_sr),
        "bandpass_hz": [float(args.bandpass_low), float(args.bandpass_high)],
        "notch_hz": float(args.notch_freq),
        "notch_q": float(args.notch_q),
        "channels": BIPOLAR_NAMES,
        "n_channels": len(BIPOLAR_NAMES),
        "samples_per_window": int(round(args.window_seconds * args.target_sr)),
        "label_map": {str(i): INDEX_TO_LABEL_NAME[i] for i in range(len(INDEX_TO_LABEL_NAME))},
        "label_rule": {
            "normal": 0,
            "abnormal": 1,
            "source": "directory name in the TUAB tree",
        },
        "normalize_per_window": bool(args.normalize_per_window),
        "eeg_dtype": str(eeg_dtype),
        "coords_dtype": str(coords_dtype),
        "coords_reference_montage": str(args.montage),
        "coords_definition": "average xyz position of each bipolar electrode pair from the MNE template montage",
        "subject_id_rule": "filename prefix before _s### when present; otherwise stem prefix before first underscore; otherwise full stem",
        "split_policy": {
            "train_source": "ROOT_DIR/train",
            "test_source": "ROOT_DIR/eval",
            "val_from_train": False if args.disable_val_split else True,
            "val_ratio": 0.0 if args.disable_val_split else float(args.val_ratio),
            "subject_level": True,
            "ordering": "sorted unique subject ids within each class",
            "assignment": "for abnormal and normal separately, first floor((1-val_ratio)*N) subjects -> train, remainder -> val",
            "seed_text_used": False,
            "classwise_split": True,
        },
        "scan_summary": {
            **scan_summary,
            "skip_reasons": {k: int(v) for k, v in sorted(scan_summary["skip_reasons"].items())},
        },
        "splits": {},
        "subjects": {k: sorted(v) for k, v in split_to_subjects.items()},
    }

    split_order = ["train", "val", "test"]
    for split_name in split_order:
        split_files = split_to_files[split_name]
        if not split_files:
            continue
        meta = write_split_arrays(
            split_name=split_name,
            split_files=split_files,
            file_info=file_info,
            out_dir=out_dir,
            coords_template=coords_template,
            target_sr=args.target_sr,
            bandpass_low=args.bandpass_low,
            bandpass_high=args.bandpass_high,
            notch_freq=args.notch_freq,
            notch_q=args.notch_q,
            window_seconds=args.window_seconds,
            normalize_per_window=args.normalize_per_window,
            eeg_dtype=eeg_dtype,
            coords_dtype=coords_dtype,
            overwrite=args.overwrite,
        )
        dataset_meta["splits"][split_name] = {
            "n_subjects": len(split_to_subjects[split_name]),
            "n_files": len(split_files),
            "n_samples": int(meta["n_samples"]),
            "class_counts": meta["class_counts"],
        }

    with open(out_dir / "dataset_meta.json", "w", encoding="utf-8") as f:
        json.dump(dataset_meta, f, indent=2, ensure_ascii=False)

    with open(out_dir / "scan_report.json", "w", encoding="utf-8") as f:
        json.dump(file_info, f, indent=2, ensure_ascii=False)

    print("\nDone.")
    for split_name in split_order:
        if split_name not in dataset_meta["splits"]:
            continue
        info = dataset_meta["splits"][split_name]
        print(
            f"[{split_name}] subjects={info['n_subjects']} files={info['n_files']} samples={info['n_samples']} class_counts={info['class_counts']}"
        )


if __name__ == "__main__":
    main()
