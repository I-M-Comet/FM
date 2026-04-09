#!/usr/bin/env python3
"""
preprocess_tuev_to_npy.py

Single-file TUEV preprocessing pipeline that directly writes .npy files.

What it does
------------
1) Recursively scans TUEV under:
      ROOT_DIR/
        train/<subject>/**/*.edf
        eval/<subject>/**/*.edf

2) Uses the common TUEV downstream convention from BIOT / CBraMod:
   - start from referential EDF channels (e.g. "EEG FP1-REF")
   - derive the standard 16 bipolar montage channels
   - 0.3-75 Hz band-pass
   - 60 Hz notch
   - resample to 200 Hz
   - 5-second windows
   - one sample per annotation row
   - single 6-class label per sample
   - original annotation channel index preserved separately as offending_channel

3) Supports both:
   - official-style .rec annotations
   - per-channel .lab annotations like:
       aaaaafop_00000001_ch000.lab
       aaaaafop_00000001_ch001.lab
       ...

4) Writes memory-mappable output directly:
      OUT_DIR/
        train/
          eeg.npy                (N, 16, 1000)
          coords.npy             (N, 16, 3)
          label.npy              (N,)
          offending_channel.npy  (N,)
          meta.json
        val/
          ...
        test/
          ...
        dataset_meta.json

Notes
-----
- The EDF channel list in TUH/TUEV is referential (unipolar / REF-style), not already
  the 16 bipolar derivations. The bipolar channels are constructed by subtracting
  predefined referential pairs.
- "Global" classification here follows the common BIOT-style simplification:
  each annotation row yields one full-montage sample with a single class label.
  We do NOT majority-vote or merge all channel labels into a separate global timeline.
- By default, per-window z-score normalization is enabled because that is what you asked for.
# python ./eeg_dataset_preprocess/eval_dataset_preprocess_TUEV.py --root_dir "D:/open_eeg/tuev/edf" --out_dir "D:/open_eeg_eval/TUEV_npy"
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import warnings
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import mne
import numpy as np
import scipy.signal as signal
from numpy.lib.format import open_memmap
from tqdm import tqdm

# -----------------------------------------------------------------------------
# Label definitions (0-based for training)
# -----------------------------------------------------------------------------
LABEL_NAME_TO_INDEX = {
    "spsw": 0,
    "gped": 1,
    "pled": 2,
    "eyem": 3,
    "artf": 4,
    "bckg": 5,
}
INDEX_TO_LABEL_NAME = {v: k.upper() for k, v in LABEL_NAME_TO_INDEX.items()}

LABEL_TOKEN_ALIASES = {
    "spsw": 0,
    "spike": 0,
    "spikewave": 0,
    "gped": 1,
    "pled": 2,
    "eyem": 3,
    "eye": 3,
    "eyemov": 3,
    "eyemovement": 3,
    "artf": 4,
    "artifact": 4,
    "artefact": 4,
    "bckg": 5,
    "background": 5,
    "1": 0,
    "2": 1,
    "3": 2,
    "4": 3,
    "5": 4,
    "6": 5,
}

# -----------------------------------------------------------------------------
# Official BIOT-style 16 bipolar derivations, written in the TUH/TUEV naming
# convention (T3/T4/T5/T6). When we need coordinates from MNE template montages,
# we map T3->T7, T4->T8, T5->P7, T6->P8 internally.
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

# Internal referential name -> MNE built-in montage channel name.
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

# Accept either modern or legacy spellings from the EDF.
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

NUM_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")


@dataclass(frozen=True, slots=True)
class Event:
    start_sec: float
    stop_sec: float
    label_idx: int
    offending_channel: int  # original channel index from .rec or .lab filename chNNN
    source: str             # "rec" or "lab"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="TUEV -> direct NPY preprocessing")
    ap.add_argument("--root_dir", type=str, required=True, help="TUEV edf root that contains train/ and eval/")
    ap.add_argument("--out_dir", type=str, required=True, help="Output directory")
    ap.add_argument("--target_sr", type=int, default=200, help="Target sampling rate")
    ap.add_argument("--bandpass_low", type=float, default=0.3, help="Band-pass low cutoff (Hz)")
    ap.add_argument("--bandpass_high", type=float, default=75.0, help="Band-pass high cutoff (Hz)")
    ap.add_argument("--notch_freq", type=float, default=60.0, help="Notch frequency (Hz). Use <=0 to disable")
    ap.add_argument("--notch_q", type=float, default=30.0, help="Notch Q factor")
    ap.add_argument("--window_seconds", type=float, default=5.0, help="Segment length in seconds")
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
        "--drop_duplicate_global_events",
        action="store_true",
        help="Deduplicate identical (start, stop, label) events across channels. Disabled by default because TUEV labels are per annotation row.",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files",
    )
    ap.add_argument(
        "--disable_val_split",
        action="store_true",
        help="Do not create val; use original train as train and eval as test",
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
    """
    Convert EDF channel names such as:
      EEG FP1-REF, FP1-LE, EEG T3-REF, EEG T7-REF
    into the internal referential channel namespace used to build the
    16 bipolar derivations.
    """
    s = name.upper().strip()
    s = s.replace(".", "")
    s = re.sub(r"^EEG\s*", "", s)
    s = s.replace("EEG", "")
    s = s.replace("POL ", "")
    s = s.replace("POL", "")
    s = s.strip()

    # Remove common reference suffixes before collapsing separators.
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


def subject_id_from_path(edf_path: Path, split_dir: Path) -> str:
    """
    CBraMod-style TUEV subject key:
      filename prefix before the first underscore, e.g.
        aaaaafop_00000001.edf -> aaaaafop

    Fall back to the first directory under split_dir when needed.
    """
    stem = edf_path.stem
    if "_" in stem:
        left = stem.split("_", 1)[0]
        if left:
            return left
    rel = edf_path.relative_to(split_dir)
    if len(rel.parts) >= 2:
        return rel.parts[0]
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


def normalize_label_token(token: str) -> Optional[int]:
    t = re.sub(r"[^a-z0-9]+", "", token.lower())
    if not t:
        return None
    return LABEL_TOKEN_ALIASES.get(t)


def infer_time_pair(
    start_raw: float,
    stop_raw: float,
    duration_sec: float,
    raw_sfreq: float,
) -> Optional[Tuple[float, float]]:
    """
    Heuristic unit inference for .lab files.

    Priority:
      1) already in seconds
      2) TUH 10-microsecond units
      3) milliseconds
      4) samples
    """
    candidates: List[Tuple[str, float, float]] = [
        ("sec", start_raw, stop_raw),
        ("10us", start_raw / 1e5, stop_raw / 1e5),
        ("ms", start_raw / 1e3, stop_raw / 1e3),
    ]
    if raw_sfreq > 0:
        candidates.append(("samples", start_raw / raw_sfreq, stop_raw / raw_sfreq))

    tol = 1.0 / max(raw_sfreq, 1.0)
    for _, s, e in candidates:
        if not (np.isfinite(s) and np.isfinite(e)):
            continue
        if s < -tol:
            continue
        if e <= s:
            continue
        if e <= duration_sec + max(1e-3, tol):
            s = max(0.0, s)
            e = min(duration_sec, e)
            if e > s:
                return float(s), float(e)
    return None


def is_number_token(tok: str) -> bool:
    return bool(NUM_RE.match(tok))


def parse_annotation_line_generic(
    line: str,
    duration_sec: float,
    raw_sfreq: float,
) -> Optional[Tuple[float, float, int]]:
    """
    Generic parser for per-channel .lab lines.

    Accepts lines like:
      1080000 1180000 bckg
      0 100000 SPSW
      0.0 1.0 SPSW
      0,100000,1
      onset=0.0 offset=1.0 label=SPSW
    """
    line = line.strip()
    if not line:
        return None
    if line.startswith("#") or line.startswith("%") or line.startswith(";"):
        return None

    # First try a tolerant whole-line search for textual labels.
    line_lower = line.lower()
    for label_name, idx in LABEL_NAME_TO_INDEX.items():
        if re.search(rf"(?<![a-z0-9]){re.escape(label_name)}(?![a-z0-9])", line_lower):
            numbers = [float(x) for x in re.findall(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", line)]
            if len(numbers) >= 2:
                inferred = infer_time_pair(numbers[0], numbers[1], duration_sec, raw_sfreq)
                if inferred is not None:
                    start_sec, stop_sec = inferred
                    return start_sec, stop_sec, idx

    tokens = [tok for tok in re.split(r"[\s,;\t=]+", line) if tok]
    if len(tokens) < 3:
        return None

    label_idx: Optional[int] = None
    label_pos: Optional[int] = None

    # Prefer textual labels first.
    for i, tok in enumerate(tokens):
        lbl = normalize_label_token(tok)
        if lbl is None:
            continue
        cleaned = re.sub(r"[^a-z0-9]+", "", tok.lower())
        if cleaned in {"spsw", "gped", "pled", "eyem", "artf", "bckg", "spike", "artifact", "artefact", "background", "eyemov", "eyemovement", "eye"}:
            label_idx = lbl
            label_pos = i
            break

    # Fallback: last token can be numeric class code 1..6.
    if label_idx is None:
        last_lbl = normalize_label_token(tokens[-1])
        if last_lbl is not None and re.sub(r"[^a-z0-9]+", "", tokens[-1].lower()) in {"1", "2", "3", "4", "5", "6"}:
            label_idx = last_lbl
            label_pos = len(tokens) - 1

    if label_idx is None or label_pos is None:
        return None

    numeric_before = [float(tok) for tok in tokens[:label_pos] if is_number_token(tok)]
    if len(numeric_before) >= 2:
        start_raw, stop_raw = numeric_before[-2], numeric_before[-1]
    else:
        numeric_all = [float(tok) for i, tok in enumerate(tokens) if i != label_pos and is_number_token(tok)]
        if len(numeric_all) < 2:
            return None
        start_raw, stop_raw = numeric_all[0], numeric_all[1]

    inferred = infer_time_pair(start_raw, stop_raw, duration_sec, raw_sfreq)
    if inferred is None:
        return None
    start_sec, stop_sec = inferred
    return start_sec, stop_sec, label_idx


def parse_rec_annotations(
    rec_path: Path,
    duration_sec: float,
    raw_sfreq: float,
    drop_duplicate_global_events: bool,
) -> List[Event]:
    events: List[Event] = []
    with open(rec_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p for p in re.split(r"[\s,;\t]+", line) if p]
            if len(parts) < 4:
                continue

            try:
                offending_channel = int(float(parts[0]))
            except Exception:
                offending_channel = -1

            try:
                start_raw = float(parts[1])
                stop_raw = float(parts[2])
            except Exception:
                continue

            label_idx = normalize_label_token(parts[3])
            if label_idx is None:
                continue

            inferred = infer_time_pair(start_raw, stop_raw, duration_sec, raw_sfreq)
            if inferred is None:
                continue

            start_sec, stop_sec = inferred
            events.append(Event(start_sec, stop_sec, label_idx, offending_channel, "rec"))

    if drop_duplicate_global_events:
        events = deduplicate_global_events(events)
    return sorted(events, key=lambda e: (e.start_sec, e.stop_sec, e.label_idx, e.offending_channel))


def parse_lab_annotations(
    edf_path: Path,
    duration_sec: float,
    raw_sfreq: float,
    drop_duplicate_global_events: bool,
) -> List[Event]:
    pattern = re.compile(rf"^{re.escape(edf_path.stem)}_ch(\d+)\.lab$", re.IGNORECASE)
    lab_files: List[Tuple[int, Path]] = []
    for p in edf_path.parent.iterdir():
        if not p.is_file():
            continue
        m = pattern.match(p.name)
        if m:
            lab_files.append((int(m.group(1)), p))

    lab_files.sort(key=lambda x: x[0])
    events: List[Event] = []

    for ch_idx, lab_path in lab_files:
        with open(lab_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                parsed = parse_annotation_line_generic(line, duration_sec, raw_sfreq)
                if parsed is None:
                    continue
                start_sec, stop_sec, label_idx = parsed
                events.append(Event(start_sec, stop_sec, label_idx, ch_idx, "lab"))

    if drop_duplicate_global_events:
        events = deduplicate_global_events(events)
    return sorted(events, key=lambda e: (e.start_sec, e.stop_sec, e.label_idx, e.offending_channel))


def deduplicate_global_events(events: Sequence[Event], ndigits: int = 6) -> List[Event]:
    seen = set()
    out: List[Event] = []
    for ev in events:
        key = (round(ev.start_sec, ndigits), round(ev.stop_sec, ndigits), ev.label_idx)
        if key in seen:
            continue
        seen.add(key)
        out.append(ev)
    return out


def parse_annotations_for_edf(
    edf_path: Path,
    duration_sec: float,
    raw_sfreq: float,
    drop_duplicate_global_events: bool,
) -> List[Event]:
    rec_path = edf_path.with_suffix(".rec")
    if rec_path.exists():
        return parse_rec_annotations(rec_path, duration_sec, raw_sfreq, drop_duplicate_global_events)
    return parse_lab_annotations(edf_path, duration_sec, raw_sfreq, drop_duplicate_global_events)


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


def circular_take_window(data: np.ndarray, start_idx: int, length: int) -> np.ndarray:
    n = data.shape[-1]
    if n <= 0:
        raise ValueError("Cannot extract from empty signal")
    idx = (np.arange(length, dtype=np.int64) + int(start_idx)) % n
    return data[:, idx]


def extract_event_window(
    data: np.ndarray,
    start_sec: float,
    stop_sec: float,
    sfreq: int,
    window_seconds: float,
) -> np.ndarray:
    """
    Build a fixed-length event-centered 5-second window.

    This matches the BIOT-style TUEV construction:
    for 1-second annotations, the resulting window contains about 2 seconds
    of context on each side.
    """
    target_len = int(round(window_seconds * sfreq))
    start_idx = int(round(start_sec * sfreq))
    stop_idx = int(round(stop_sec * sfreq))
    event_len = max(1, stop_idx - start_idx)

    if event_len <= target_len:
        pad_total = target_len - event_len
        pad_left = pad_total // 2
        win_start = start_idx - pad_left
    else:
        center = (start_idx + stop_idx) // 2
        win_start = center - target_len // 2

    window = circular_take_window(data, win_start, target_len)
    if window.shape[-1] != target_len:
        raise RuntimeError(f"Internal windowing error: got {window.shape[-1]} != {target_len}")
    return window


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
    val_ratio: float,
    seed_text: str,
    drop_duplicate_global_events: bool,
    disable_val_split: bool,
) -> Tuple[
    Dict[str, List[Path]],
    Dict[str, List[Event]],
    Dict[str, dict],
    Dict[str, set[str]],
    dict,
]:
    train_dir = root_dir / "train"
    eval_dir = root_dir / "eval"

    train_edfs = list_edf_files(train_dir)
    eval_edfs = list_edf_files(eval_dir)

    train_subjects = [subject_id_from_path(p, train_dir) for p in train_edfs]
    if disable_val_split:
        train_subject_set = set(train_subjects)
        val_subject_set: set[str] = set()
    else:
        train_subject_set, val_subject_set = deterministic_subject_split(train_subjects, val_ratio, seed_text)

    split_to_files: Dict[str, List[Path]] = {"train": [], "val": [], "test": []}
    split_to_subjects: Dict[str, set[str]] = {"train": set(), "val": set(), "test": set()}
    events_cache: Dict[str, List[Event]] = {}
    file_info: Dict[str, dict] = {}
    summary = {
        "n_train_edf_found": len(train_edfs),
        "n_eval_edf_found": len(eval_edfs),
        "n_files_scanned": 0,
        "n_files_kept": 0,
        "n_files_skipped": 0,
        "skip_reasons": Counter(),
    }

    all_files = [(p, "train_src") for p in train_edfs] + [(p, "eval_src") for p in eval_edfs]

    for edf_path, src_split in tqdm(all_files, desc="Scanning EDF + annotations"):
        summary["n_files_scanned"] += 1
        try:
            raw = load_raw_edf(edf_path, preload=False)
            duration_sec = float(raw.n_times / raw.info["sfreq"])
            raw_sfreq = float(raw.info["sfreq"])
            idx_map = build_ref_index_map(raw.ch_names)
            missing = [ch for ch in REFERENTIAL_ORDER if ch not in idx_map]
            if hasattr(raw, "close"):
                raw.close()
        except Exception as e:
            summary["n_files_skipped"] += 1
            summary["skip_reasons"][f"edf_read_error: {type(e).__name__}"] += 1
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

        events = parse_annotations_for_edf(
            edf_path=edf_path,
            duration_sec=duration_sec,
            raw_sfreq=raw_sfreq,
            drop_duplicate_global_events=drop_duplicate_global_events,
        )
        if len(events) == 0:
            summary["n_files_skipped"] += 1
            summary["skip_reasons"]["no_valid_annotations"] += 1
            file_info[str(edf_path)] = {
                "kept": False,
                "reason": "no_valid_annotations",
            }
            continue

        if src_split == "train_src":
            subject_id = subject_id_from_path(edf_path, train_dir)
            out_split = "train" if subject_id in train_subject_set else "val"
        else:
            subject_id = subject_id_from_path(edf_path, eval_dir)
            out_split = "test"

        events_cache[str(edf_path)] = events
        split_to_files[out_split].append(edf_path)
        split_to_subjects[out_split].add(subject_id)
        summary["n_files_kept"] += 1

        file_info[str(edf_path)] = {
            "kept": True,
            "source_split": src_split,
            "output_split": out_split,
            "subject_id": subject_id,
            "n_events": len(events),
            "duration_sec": duration_sec,
            "raw_sfreq": raw_sfreq,
            "annotation_source": events[0].source if events else None,
        }

    return split_to_files, events_cache, file_info, split_to_subjects, summary


def prepare_output_split(
    split_dir: Path,
    n_samples: int,
    eeg_shape_tail: Tuple[int, int],
    eeg_dtype: np.dtype,
    coords_dtype: np.dtype,
    overwrite: bool,
) -> Tuple[np.memmap, np.memmap, np.memmap, np.memmap]:
    ensure_dir(split_dir)

    eeg_path = split_dir / "eeg.npy"
    coords_path = split_dir / "coords.npy"
    label_path = split_dir / "label.npy"
    offending_path = split_dir / "offending_channel.npy"

    for p in [eeg_path, coords_path, label_path, offending_path, split_dir / "meta.json"]:
        remove_if_overwrite(p, overwrite)

    eeg_mm = open_memmap(eeg_path, mode="w+", dtype=eeg_dtype, shape=(n_samples, *eeg_shape_tail))
    coords_mm = open_memmap(coords_path, mode="w+", dtype=coords_dtype, shape=(n_samples, eeg_shape_tail[0], 3))
    label_mm = open_memmap(label_path, mode="w+", dtype=np.int8, shape=(n_samples,))
    offending_mm = open_memmap(offending_path, mode="w+", dtype=np.int16, shape=(n_samples,))
    return eeg_mm, coords_mm, label_mm, offending_mm


def write_split_arrays(
    split_name: str,
    split_files: Sequence[Path],
    events_cache: Dict[str, List[Event]],
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
    n_samples = sum(len(events_cache[str(p)]) for p in split_files)
    n_chans = len(BIPOLAR_PAIRS)
    n_times = int(round(window_seconds * target_sr))

    split_dir = out_dir / split_name
    eeg_mm, coords_mm, label_mm, offending_mm = prepare_output_split(
        split_dir=split_dir,
        n_samples=n_samples,
        eeg_shape_tail=(n_chans, n_times),
        eeg_dtype=eeg_dtype,
        coords_dtype=coords_dtype,
        overwrite=overwrite,
    )

    write_idx = 0
    class_counts = Counter()
    annotation_source_counts = Counter()
    n_files = 0

    for edf_path in tqdm(split_files, desc=f"Writing {split_name}"):
        events = events_cache[str(edf_path)]
        if not events:
            continue
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

        for ev in events:
            window = extract_event_window(
                processed,
                start_sec=ev.start_sec,
                stop_sec=ev.stop_sec,
                sfreq=target_sr,
                window_seconds=window_seconds,
            )
            if normalize_per_window:
                window = zscore_per_window(window)

            eeg_mm[write_idx] = window.astype(eeg_dtype, copy=False)
            coords_mm[write_idx] = coords_template.astype(coords_dtype, copy=False)
            label_mm[write_idx] = np.int8(ev.label_idx)
            offending_mm[write_idx] = np.int16(ev.offending_channel)

            class_counts[ev.label_idx] += 1
            annotation_source_counts[ev.source] += 1
            write_idx += 1

        n_files += 1

    if write_idx != n_samples:
        raise RuntimeError(f"Split {split_name}: wrote {write_idx} samples but allocated {n_samples}")

    eeg_mm.flush()
    coords_mm.flush()
    label_mm.flush()
    offending_mm.flush()

    meta = {
        "split": split_name,
        "n_files": n_files,
        "n_samples": n_samples,
        "sample_construction": "one 5-second full 16-bipolar sample per annotation row",
        "offending_channel_definition": "original channel index from .rec column 0 or .lab filename suffix chNNN; not remapped",
        "eeg": {
            "path": str((split_dir / "eeg.npy").resolve()),
            "dtype": str(np.dtype(eeg_dtype)),
            "shape": [n_samples, n_chans, n_times],
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
        "offending_channel": {
            "path": str((split_dir / "offending_channel.npy").resolve()),
            "dtype": "int16",
            "shape": [n_samples],
            "size_bytes": int((split_dir / "offending_channel.npy").stat().st_size),
            "size_h": human_bytes((split_dir / "offending_channel.npy").stat().st_size),
        },
        "class_counts": to_jsonable_counter(class_counts),
        "annotation_source_counts": {k: int(v) for k, v in sorted(annotation_source_counts.items())},
        "label_map": {str(i): INDEX_TO_LABEL_NAME[i] for i in range(len(INDEX_TO_LABEL_NAME))},
        "channels": BIPOLAR_NAMES,
    }

    with open(split_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    del eeg_mm, coords_mm, label_mm, offending_mm
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

    split_to_files, events_cache, file_info, split_to_subjects, scan_summary = scan_dataset(
        root_dir=root_dir,
        val_ratio=args.val_ratio,
        seed_text=args.seed_text,
        drop_duplicate_global_events=args.drop_duplicate_global_events,
        disable_val_split=args.disable_val_split,
    )

    dataset_meta = {
        "dataset": "TUEV",
        "root_dir": str(root_dir),
        "output_dir": str(out_dir),
        "task": "global_6class_event_classification",
        "sample_construction": "one full-montage sample per annotation row from .rec or per-channel .lab",
        "input_channel_kind": "referential EDF channels",
        "derived_channel_kind": "16 bipolar derivations",
        "window_seconds": float(args.window_seconds),
        "target_sr": int(args.target_sr),
        "bandpass_hz": [float(args.bandpass_low), float(args.bandpass_high)],
        "notch_hz": float(args.notch_freq),
        "notch_q": float(args.notch_q),
        "channels": BIPOLAR_NAMES,
        "n_channels": len(BIPOLAR_NAMES),
        "samples_per_window": int(round(args.window_seconds * args.target_sr)),
        "label_map": {str(i): INDEX_TO_LABEL_NAME[i] for i in range(len(INDEX_TO_LABEL_NAME))},
        "drop_duplicate_global_events": bool(args.drop_duplicate_global_events),
        "normalize_per_window": bool(args.normalize_per_window),
        "eeg_dtype": str(eeg_dtype),
        "coords_dtype": str(coords_dtype),
        "coords_reference_montage": str(args.montage),
        "coords_definition": "average xyz position of each bipolar electrode pair from the MNE template montage",
        "offending_channel_definition": "original channel index from .rec column 0 or .lab filename suffix chNNN; not remapped",
        "subject_id_rule": "filename prefix before first underscore when available; otherwise first directory under split root; otherwise full stem",
        "split_policy": {
            "train_source": "ROOT_DIR/train",
            "test_source": "ROOT_DIR/eval",
            "val_from_train": False if args.disable_val_split else True,
            "val_ratio": 0.0 if args.disable_val_split else float(args.val_ratio),
            "subject_level": True,
            "ordering": "sorted unique subject ids",
            "assignment": "first floor((1-val_ratio)*N) subjects -> train, remainder -> val",
            "seed_text_used": False,
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
            events_cache=events_cache,
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
