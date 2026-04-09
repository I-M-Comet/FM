#!/usr/bin/env python3
"""
eval_dataset_preprocess_BCIC4_2a_moabb_ea.py

MOABB(BNCI2014_001)를 사용해 BCI Competition IV 2a를 다운로드/전처리하고
바로 .npy 형태로 저장한다.

핵심 변경점
-----------
1) REVE 설정을 반영해 4초 윈도우([2, 6] s per trial)를 사용
   - BNCI2014_001의 class event(769~772)는 'cue onset'에 해당하므로,
     cue onset 기준 [0, 4] s = trial 기준 [2, 6] s 와 동일하다.
2) Euclidean Alignment(EA) 옵션 추가
   - 기본값: subject 단위 EA 적용
3) band-pass 기본값을 0.5–99.5 Hz로 조정
4) z-score는 기본 OFF
   - REVE 문구에는 포함되지 않으므로 exact reproduction 관점에서 끔
   - 필요하면 CONFIG["APPLY_ZSCORE"] = True 로 켤 수 있음
"""

import os
import json
import time
import math
import warnings
from typing import Dict, List, Tuple

import numpy as np
import scipy.signal as signal
import mne
from moabb.datasets import BNCI2014_001
from tqdm import tqdm


CONFIG = {
    "OUTPUT_DIR": "D:/open_eeg_eval/BCIC_IV_2a_npy/",
    "montage": "standard_1005",

    # REVE-style defaults
    "TARGET_SR": 200,
    "BANDPASS": (0.5, 99.5),
    # 원 데이터는 수집 시점에 50 Hz notch가 이미 적용되어 있음.
    # 추가 notch를 원하면 50.0으로 바꾸면 됨.
    "NOTCH_Q": 30.0,
    "NOTCH_FREQ": None,

    # BNCI2014_001에서 class annotation은 cue onset 기준.
    # trial 기준 [2, 6] s = cue 기준 [0, 4] s
    "WINDOW_FROM_CUE_SECONDS": (0.0, 4.0),
    "WINDOW_FROM_TRIAL_SECONDS": (2.0, 6.0),  # 메타 기록용

    # REVE exactness 관점에서 z-score는 기본 OFF
    "APPLY_ZSCORE": False,
    "CLIP_LIMIT": 15.0,  # z-score를 켰을 때만 사용

    # Euclidean Alignment
    "APPLY_EA": True,
    # "subject": 한 subject의 두 세션 trial 전체로 EA
    # "session": 각 session 별로 EA
    "EA_SCOPE": "subject",

    # 저장 dtype
    "SAVE_DTYPE": "float16",  # "float16" or "float32"
}

MOABB_EVENT_MAP = {
    "left_hand": 0,
    "right_hand": 1,
    "feet": 2,
    "tongue": 3,
}


class SmartEEGPreprocessor:
    """연속 EEG에 대해 filter + resample만 수행."""

    def __init__(self, target_sr: int, bandpass_freq: Tuple[float, float]):
        self.target_sr = int(target_sr)
        self.bandpass_freq = bandpass_freq

    def apply(self, eeg_data: np.ndarray, original_sr: float) -> np.ndarray:
        eeg_data = np.asarray(eeg_data, dtype=np.float32)
        nyq = 0.5 * float(original_sr)
        low_cut, high_cut = self.bandpass_freq
        adjusted_high = (nyq - 1.0) if high_cut >= nyq else high_cut

        line_freq = CONFIG["NOTCH_FREQ"]
        if line_freq and line_freq < nyq:
            b_notch, a_notch = signal.iirnotch(line_freq, Q=CONFIG["NOTCH_Q"], fs=original_sr)
            eeg_data = signal.filtfilt(b_notch, a_notch, eeg_data, axis=-1)

        wn_low = low_cut / nyq
        wn_high = adjusted_high / nyq
        if wn_high >= 1.0:
            wn_high = 0.99

        sos = signal.butter(3, [wn_low, wn_high], btype="band", analog=False, output="sos")
        eeg_data = signal.sosfiltfilt(sos, eeg_data, axis=-1)

        if int(round(original_sr)) != self.target_sr:
            gcd = math.gcd(int(round(original_sr)), self.target_sr)
            up = int(self.target_sr // gcd)
            down = int(int(round(original_sr)) // gcd)
            eeg_data = signal.resample_poly(eeg_data, up, down, axis=-1)

        return eeg_data.astype(np.float32)


def _human_bytes(n: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    x = float(n)
    for u in units:
        if x < 1024.0 or u == units[-1]:
            return f"{x:.2f}{u}"
        x /= 1024.0
    return f"{x:.2f}TB"


def _matrix_inv_sqrt_spd(mat: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """SPD 행렬의 inverse square root."""
    mat = 0.5 * (mat + mat.T)
    c = mat.shape[0]
    reg = eps * max(float(np.trace(mat)) / max(c, 1), 1.0)
    mat = mat + reg * np.eye(c, dtype=mat.dtype)

    eigvals, eigvecs = np.linalg.eigh(mat)
    eigvals = np.clip(eigvals, eps, None)
    inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
    return inv_sqrt.astype(np.float32)


def euclidean_align_trials(trials: np.ndarray) -> np.ndarray:
    """
    trials: [N, C, T]
    subject/session 단위 평균 covariance를 이용해 EA 수행.
    """
    if trials.ndim != 3:
        raise ValueError(f"Expected [N, C, T], got {trials.shape}")

    n_trials, n_ch, _ = trials.shape
    if n_trials == 0:
        return trials

    cov_sum = np.zeros((n_ch, n_ch), dtype=np.float64)
    for x in trials:
        x = x.astype(np.float64, copy=False)
        x = x - x.mean(axis=-1, keepdims=True)
        cov = (x @ x.T) / max(x.shape[-1], 1)
        cov = 0.5 * (cov + cov.T)
        cov_sum += cov

    ref_cov = (cov_sum / n_trials).astype(np.float32)
    ref_inv_sqrt = _matrix_inv_sqrt_spd(ref_cov)

    aligned = np.einsum("ij,njt->nit", ref_inv_sqrt, trials.astype(np.float32), optimize=True)
    return aligned.astype(np.float32)


def apply_per_trial_zscore(trials: np.ndarray, clip_limit: float) -> np.ndarray:
    mean = trials.mean(axis=-1, keepdims=True)
    std = trials.std(axis=-1, keepdims=True)
    trials = (trials - mean) / (std + 1e-8)
    return np.clip(trials, -clip_limit, clip_limit).astype(np.float32)


def save_npy_split(
    eeg_list: List[np.ndarray],
    coords_list: List[np.ndarray],
    label_list: List[int],
    split_dir: str,
    split_name: str,
    subject_ids: List[int],
) -> int:
    if not eeg_list:
        print(f"[{split_name}] No segments found. Skipping.")
        return 0

    os.makedirs(split_dir, exist_ok=True)

    eeg_arr = np.stack(eeg_list)
    coords_arr = np.stack(coords_list)
    label_arr = np.array(label_list, dtype=np.int8)
    subject_arr = np.array(subject_ids, dtype=np.int16)

    save_dtype = np.float16 if CONFIG["SAVE_DTYPE"] == "float16" else np.float32
    eeg_arr = eeg_arr.astype(save_dtype, copy=False)
    coords_arr = coords_arr.astype(np.float16, copy=False)

    eeg_path = os.path.join(split_dir, "eeg.npy")
    coords_path = os.path.join(split_dir, "coords.npy")
    label_path = os.path.join(split_dir, "label.npy")
    subject_path = os.path.join(split_dir, "subject_id.npy")

    np.save(eeg_path, eeg_arr)
    np.save(coords_path, coords_arr)
    np.save(label_path, label_arr)
    np.save(subject_path, subject_arr)

    meta = {
        "source": "MOABB_BNCI2014_001",
        "split": split_name,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "num_segments": int(len(eeg_list)),
        "arrays": {
            "eeg": {"shape": list(eeg_arr.shape), "dtype": str(eeg_arr.dtype)},
            "coords": {"shape": list(coords_arr.shape), "dtype": str(coords_arr.dtype)},
            "label": {"shape": list(label_arr.shape), "dtype": str(label_arr.dtype)},
            "subject_id": {"shape": list(subject_arr.shape), "dtype": str(subject_arr.dtype)},
        },
        "config": CONFIG,
        "label_mapping": MOABB_EVENT_MAP,
    }
    with open(os.path.join(split_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    unique, counts = np.unique(label_arr, return_counts=True)
    dist_str = ", ".join(f"{u}:{c}" for u, c in zip(unique, counts))

    print(f"[{split_name}] Saved {len(eeg_list)} segments")
    print(f"  - eeg shape: {eeg_arr.shape} ({_human_bytes(os.path.getsize(eeg_path))})")
    print(f"  - coords shape: {coords_arr.shape}")
    print(f"  - label dist (0=L, 1=R, 2=F, 3=T): {dist_str}\n")
    return int(len(eeg_list))


def split_key_from_subject(subj: int) -> str:
    if 1 <= subj <= 5:
        return "train"
    if 6 <= subj <= 7:
        return "val"
    return "test"


def extract_trials_from_raw(
    raw: mne.io.BaseRaw,
    preprocessor: SmartEEGPreprocessor,
) -> Tuple[List[np.ndarray], List[int], np.ndarray]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = raw.copy()
        raw.pick_types(eeg=True, eog=False, stim=False, misc=False)
        raw.set_montage(CONFIG["montage"], match_case=False, on_missing="ignore")

    coords = []
    for i in range(len(raw.ch_names)):
        loc = raw.info["chs"][i]["loc"][:3]
        coords.append(loc)
    coords_array = np.array(coords, dtype=np.float32)

    original_sr = float(raw.info["sfreq"])
    data = raw.get_data().astype(np.float32)
    processed_full = preprocessor.apply(data, original_sr)
    total_length = processed_full.shape[-1]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        events, event_dict = mne.events_from_annotations(raw, verbose=False)

    id_to_annot = {v: k for k, v in event_dict.items()}
    start_offset = int(round(CONFIG["WINDOW_FROM_CUE_SECONDS"][0] * CONFIG["TARGET_SR"]))
    end_offset = int(round(CONFIG["WINDOW_FROM_CUE_SECONDS"][1] * CONFIG["TARGET_SR"]))

    trials = []
    labels = []

    for i in range(len(events)):
        onset_orig = int(events[i, 0])
        event_id = int(events[i, 2])
        annot_str = id_to_annot.get(event_id, "")

        if annot_str not in MOABB_EVENT_MAP:
            continue

        final_label = MOABB_EVENT_MAP[annot_str]
        onset_target = int(round(onset_orig * (CONFIG["TARGET_SR"] / original_sr)))
        start_idx = onset_target + start_offset
        end_idx = onset_target + end_offset

        if start_idx < 0 or end_idx > total_length or end_idx <= start_idx:
            continue

        segment = processed_full[:, start_idx:end_idx]
        trials.append(segment.astype(np.float32, copy=False))
        labels.append(final_label)

    return trials, labels, coords_array


def process_subject_subject_level_ea(
    subj: int,
    subj_data: Dict,
    preprocessor: SmartEEGPreprocessor,
) -> Tuple[List[np.ndarray], List[int], List[np.ndarray]]:
    subject_trials: List[np.ndarray] = []
    subject_labels: List[int] = []
    subject_coords: List[np.ndarray] = []

    for _, session_runs in subj_data.items():
        for _, raw in session_runs.items():
            trials, labels, coords = extract_trials_from_raw(raw, preprocessor)
            subject_trials.extend(trials)
            subject_labels.extend(labels)
            subject_coords.extend([coords] * len(trials))

    if len(subject_trials) == 0:
        return [], [], []

    trials_arr = np.stack(subject_trials).astype(np.float32)

    if CONFIG["APPLY_EA"]:
        trials_arr = euclidean_align_trials(trials_arr)

    if CONFIG["APPLY_ZSCORE"]:
        trials_arr = apply_per_trial_zscore(trials_arr, CONFIG["CLIP_LIMIT"])

    out_trials = [trials_arr[i] for i in range(trials_arr.shape[0])]
    return out_trials, subject_labels, subject_coords


def process_subject_session_level_ea(
    subj: int,
    subj_data: Dict,
    preprocessor: SmartEEGPreprocessor,
) -> Tuple[List[np.ndarray], List[int], List[np.ndarray]]:
    subject_trials: List[np.ndarray] = []
    subject_labels: List[int] = []
    subject_coords: List[np.ndarray] = []

    for _, session_runs in subj_data.items():
        session_trials: List[np.ndarray] = []
        session_labels: List[int] = []
        session_coords: List[np.ndarray] = []

        for _, raw in session_runs.items():
            trials, labels, coords = extract_trials_from_raw(raw, preprocessor)
            session_trials.extend(trials)
            session_labels.extend(labels)
            session_coords.extend([coords] * len(trials))

        if len(session_trials) == 0:
            continue

        trials_arr = np.stack(session_trials).astype(np.float32)

        if CONFIG["APPLY_EA"]:
            trials_arr = euclidean_align_trials(trials_arr)

        if CONFIG["APPLY_ZSCORE"]:
            trials_arr = apply_per_trial_zscore(trials_arr, CONFIG["CLIP_LIMIT"])

        subject_trials.extend([trials_arr[i] for i in range(trials_arr.shape[0])])
        subject_labels.extend(session_labels)
        subject_coords.extend(session_coords)

    return subject_trials, subject_labels, subject_coords


if __name__ == "__main__":
    os.makedirs(CONFIG["OUTPUT_DIR"], exist_ok=True)

    print("[init] Downloading/Loading BCI Comp IV 2a via MOABB...")
    dataset = BNCI2014_001()
    preprocessor = SmartEEGPreprocessor(CONFIG["TARGET_SR"], CONFIG["BANDPASS"])

    splits = {
        "train": {"eeg": [], "coords": [], "label": [], "subject_id": []},
        "val": {"eeg": [], "coords": [], "label": [], "subject_id": []},
        "test": {"eeg": [], "coords": [], "label": [], "subject_id": []},
    }

    expected_total = 9 * 2 * 288  # 5184

    for subj in tqdm(range(1, 10), desc="Processing Subjects"):
        split_key = split_key_from_subject(subj)
        subj_data = dataset.get_data(subjects=[subj])[subj]

        if CONFIG["EA_SCOPE"] == "session":
            eegs, labels, coords = process_subject_session_level_ea(subj, subj_data, preprocessor)
        else:
            eegs, labels, coords = process_subject_subject_level_ea(subj, subj_data, preprocessor)

        splits[split_key]["eeg"].extend(eegs)
        splits[split_key]["coords"].extend(coords)
        splits[split_key]["label"].extend(labels)
        splits[split_key]["subject_id"].extend([subj] * len(labels))

        print(
            f"[subject {subj:02d}] split={split_key} "
            f"segments={len(labels)} "
            f"label_dist={dict(zip(*np.unique(np.array(labels, dtype=np.int8), return_counts=True))) if labels else {}}"
        )

    print("\n" + "=" * 60)
    print("Saving to .npy files...")
    total_segments = 0
    per_split_counts = {}

    for split_key in ["train", "val", "test"]:
        n = save_npy_split(
            splits[split_key]["eeg"],
            splits[split_key]["coords"],
            splits[split_key]["label"],
            os.path.join(CONFIG["OUTPUT_DIR"], split_key),
            split_key,
            splits[split_key]["subject_id"],
        )
        total_segments += n
        per_split_counts[split_key] = n

    dataset_meta = {
        "source": "MOABB_BNCI2014_001",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "expected_total_trials_from_dataset_description": expected_total,
        "actual_total_segments": total_segments,
        "split_counts": per_split_counts,
        "config": CONFIG,
        "label_mapping": MOABB_EVENT_MAP,
        "note": (
            "Class events in BNCI2014_001 correspond to cue onset. Therefore, "
            "taking [0, 4] seconds from cue onset is equivalent to taking [2, 6] "
            "seconds from trial onset in the original BCI IV 2a protocol."
        ),
    }
    with open(os.path.join(CONFIG["OUTPUT_DIR"], "dataset_meta.json"), "w", encoding="utf-8") as f:
        json.dump(dataset_meta, f, indent=2, ensure_ascii=False)

    print("=" * 60)
    print(f"[Done] All splits processed. Total segments: {total_segments} / expected: {expected_total}")
    print(f"[Done] Split counts: {per_split_counts}")
    print("=" * 60)
