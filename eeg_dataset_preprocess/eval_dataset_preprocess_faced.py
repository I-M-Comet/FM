#!/usr/bin/env python3
"""
eval_dataset_preprocess_FACED.py

FACED 데이터셋 전처리 → .npy 직접 저장

입력:
  sub000.pkl ~ sub122.pkl  (각 파일: 28 trials × 32 channels × time_samples)

출력:
  OUT_DIR/
    train/  (sub000~079)
      eeg.npy, coords.npy, label.npy, meta.json
    val/    (sub080~099)
      ...
    test/   (sub100~122)
      ...

Usage:
  python eval_dataset_preprocess_FACED.py
"""

import os
import glob
import re
import json
import time
import math
import pickle
import warnings
import numpy as np
import scipy.signal as signal
import mne
from multiprocessing import Pool
from tqdm import tqdm

# ==============================================================================
# [설정 영역]
# ==============================================================================
CONFIG = {
    "ROOT_DIR": "D:/open_eeg/faced/processed_data",
    "OUTPUT_DIR": "D:/open_eeg_eval/FACED_npy/",
    "FILE_EXT": "sub*.pkl",

    "ORIGINAL_SR": 250,      # FACED 원본 샘플링 레이트 (데이터에 맞게 수정)
    "TARGET_SR": 200,
    "BANDPASS": (0.5, 75.0),
    "NOTCH_Q": 30.0,
    "NOTCH_FREQ": 50.0,      # FACED는 중국 데이터 → 50Hz 노치
    "CLIP_LIMIT": 15.0,

    "SEGMENT_SECONDS": 10.0,  # 전처리 후 trial을 이 길이로 잘라서 세그먼트화

    "NUM_WORKERS": max(1, os.cpu_count() - 2),
}

# ──────────────────────────────────────────────────────────────────────────────
# FACED 28 trials에 대한 고정 라벨 시퀀스 (9종류: 0~8)
# 0001112223334444555666777888
# ──────────────────────────────────────────────────────────────────────────────
TRIAL_LABELS = [int(c) for c in "0001112223334444555666777888"]

# FACED 32채널 이름 (데이터에 맞게 수정)
CHANNELS_32 = [
    'Fp1', 'Fp2', 'AF3', 'AF4', 'F3', 'F4', 'F7', 'F8',
    'FC1', 'FC2', 'FC5', 'FC6', 'C3', 'C4', 'T7', 'T8',
    'CP1', 'CP2', 'CP5', 'CP6', 'P3', 'P4', 'P7', 'P8',
    'PO3', 'PO4', 'O1', 'O2', 'Fz', 'Cz', 'Pz', 'Oz'
]


# ==============================================================================
# 1. 전처리기 (원본과 동일 로직)
# ==============================================================================
class SmartEEGPreprocessor:
    def __init__(self, target_sr, bandpass_freq, clip_limit):
        self.target_sr = target_sr
        self.bandpass_freq = bandpass_freq
        self.clip_limit = clip_limit

    def apply(self, eeg_data, original_sr):
        """
        eeg_data: (channels, time)
        → 마지막 축(time) 기준으로 필터링/리샘플/정규화
        """
        nyq = 0.5 * original_sr
        low_cut, high_cut = self.bandpass_freq
        adjusted_high = (nyq - 1.0) if high_cut >= nyq else high_cut

        # Notch filter
        line_freq = CONFIG["NOTCH_FREQ"]
        if line_freq and line_freq < nyq:
            b_notch, a_notch = signal.iirnotch(line_freq, Q=CONFIG["NOTCH_Q"], fs=original_sr)
            eeg_data = signal.filtfilt(b_notch, a_notch, eeg_data, axis=-1)

        # Bandpass filter
        Wn_low, Wn_high = low_cut / nyq, adjusted_high / nyq
        if Wn_high >= 1.0:
            Wn_high = 0.99
        # sos = signal.butter(3, [Wn_low, Wn_high], btype='band', analog=False, output='sos')
        sos = signal.butter(3, Wn_low, btype='highpass', analog=False, output='sos')
        eeg_data = signal.sosfiltfilt(sos, eeg_data, axis=-1)

        # Resample
        if original_sr != self.target_sr:
            gcd = math.gcd(int(original_sr), int(self.target_sr))
            eeg_data = signal.resample_poly(
                eeg_data, int(self.target_sr // gcd), int(original_sr // gcd), axis=-1
            )

        # Z-score 정규화 + 클리핑
        mean = np.mean(eeg_data, axis=-1, keepdims=True)
        std = np.std(eeg_data, axis=-1, keepdims=True)
        eeg_data = (eeg_data - mean) / (std + 1e-8)
        return np.clip(eeg_data.astype(np.float16), -self.clip_limit, self.clip_limit)


# ==============================================================================
# 2. 채널 좌표 생성 (montage 기반, 파일마다 실행)
# ==============================================================================
def build_channel_coords(channel_names, montage_name="standard_1005"):
    """표준 몽타주에서 채널 3D 좌표를 가져옵니다. 피험자마다 채널이 다를 수 있으므로 매번 호출."""
    montage = mne.channels.make_standard_montage(montage_name)
    pos = montage.get_positions()["ch_pos"]

    coords = []
    for ch in channel_names:
        matched = False
        for m_name, m_pos in pos.items():
            if m_name.upper() == ch.upper():
                coords.append(m_pos[:3])
                matched = True
                break
        if not matched:
            warnings.warn(f"Channel '{ch}' not found in montage, using (0,0,0)")
            coords.append(np.array([0.0, 0.0, 0.0]))

    return np.array(coords, dtype=np.float16)


# ==============================================================================
# 3. 헬퍼: 피험자 번호 추출
# ==============================================================================
def get_subject_id(file_path):
    """sub000.pkl → 0, sub122.pkl → 122"""
    filename = os.path.basename(file_path)
    match = re.search(r'sub(\d+)', filename, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


# ==============================================================================
# 4. Worker 함수 (파일 1개 → 세그먼트 리스트)
# ==============================================================================
_PREPROCESSOR = None


def _init_worker():
    """워커 프로세스 초기화: 전처리기 세팅"""
    global _PREPROCESSOR
    _PREPROCESSOR = SmartEEGPreprocessor(
        CONFIG["TARGET_SR"], CONFIG["BANDPASS"], CONFIG["CLIP_LIMIT"]
    )


def process_single_file(file_path):
    """
    pkl 파일 1개 처리
    pkl 내부:
      - dict인 경우: data['eeg'] = (trials, ch, time), data['ch_names'] = [...] (optional)
      - ndarray인 경우: (trials, ch, time), 채널명은 CHANNELS_32 사용
    반환: (eeg_segments, coord_segments, labels) 또는 None
    """
    global _PREPROCESSOR
    try:
        subj_id = get_subject_id(file_path)
        if subj_id is None:
            return None

        with open(file_path, "rb") as f:
            raw_pkl = pickle.load(f)

        # ── 데이터 & 채널명 추출 ──
        ch_names = None
        if isinstance(raw_pkl, dict):
            # dict → 채널명이 있으면 가져오기
            ch_names = raw_pkl.get("ch_names", raw_pkl.get("channels", None))
            data = raw_pkl.get("eeg", raw_pkl.get("data", None))
            if data is None:
                print(f"[SKIP] {file_path}: dict에서 'eeg'/'data' 키를 찾을 수 없음")
                return None
        else:
            data = raw_pkl

        data = np.array(data, dtype=np.float32)

        if data.ndim != 3:
            print(f"[SKIP] {file_path}: expected 3D (trials, ch, time), got {data.shape}")
            return None

        num_trials, num_channels, _ = data.shape

        # ── 채널 좌표: 매 파일마다 생성 (피험자별 몽타주 차이 대응) ──
        if ch_names is not None:
            if isinstance(ch_names, np.ndarray):
                ch_names = ch_names.tolist()
            ch_names = [str(c) for c in ch_names]
        else:
            # 채널명 정보가 없으면 기본 목록 사용 (채널 수에 맞게 슬라이스)
            ch_names = CHANNELS_32[:num_channels]

        coords_array = build_channel_coords(ch_names)

        # ── 라벨 할당 ──
        usable_trials = min(num_trials, len(TRIAL_LABELS))
        if num_trials != len(TRIAL_LABELS):
            print(f"[WARN] {file_path}: {num_trials} trials vs {len(TRIAL_LABELS)} labels → {usable_trials}개만 사용")

        eeg_segments, coord_segments, labels = [], [], []

        # 전처리 후 세그먼트 길이 (샘플 수)
        seg_samples = int(CONFIG["SEGMENT_SECONDS"] * CONFIG["TARGET_SR"])

        for t in range(usable_trials):
            trial_data = data[t]  # (channels, time_samples)
            processed = _PREPROCESSOR.apply(trial_data, CONFIG["ORIGINAL_SR"])

            # 전처리된 trial을 SEGMENT_SECONDS 단위로 분할
            total_samples = processed.shape[-1]
            num_segs = total_samples // seg_samples  # 나머지는 버림

            for s in range(num_segs):
                start = s * seg_samples
                end = start + seg_samples
                eeg_segments.append(processed[:, start:end])
                coord_segments.append(coords_array)
                labels.append(TRIAL_LABELS[t])

        if not eeg_segments:
            return None
        return eeg_segments, coord_segments, labels

    except Exception as e:
        print(f"[ERROR] {file_path}: {e}")
        return None


# ==============================================================================
# 5. 직접 .npy 저장
# ==============================================================================
def _human_bytes(n: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    x = float(n)
    for u in units:
        if x < 1024.0 or u == units[-1]:
            return f"{x:.2f}{u}"
        x /= 1024.0
    return f"{x:.2f}TB"


def process_and_save_npy(file_list, split_dir, split_name):
    """파일 목록 전처리 → eeg.npy / coords.npy / label.npy 저장. 저장된 세그먼트 수를 반환."""
    if not file_list:
        print(f"[{split_name}] No files. Skipping.")
        return 0

    os.makedirs(split_dir, exist_ok=True)

    all_eegs, all_coords, all_labels = [], [], []

    with Pool(
        CONFIG["NUM_WORKERS"],
        initializer=_init_worker,
    ) as pool:
        for results in tqdm(
            pool.imap_unordered(process_single_file, file_list),
            total=len(file_list),
            desc=f"[{split_name}]",
        ):
            if results:
                eeg_segs, coord_segs, labs = results
                all_eegs.extend(eeg_segs)
                all_coords.extend(coord_segs)
                all_labels.extend(labs)

    if not all_eegs:
        print(f"[{split_name}] No valid segments. Skipping.")
        return 0

    eeg_arr = np.stack(all_eegs)                      # (N, C, T)
    coords_arr = np.stack(all_coords)                  # (N, C, 3)
    label_arr = np.array(all_labels, dtype=np.int8)    # (N,)

    eeg_path = os.path.join(split_dir, "eeg.npy")
    coords_path = os.path.join(split_dir, "coords.npy")
    label_path = os.path.join(split_dir, "label.npy")

    np.save(eeg_path, eeg_arr)
    np.save(coords_path, coords_arr)
    np.save(label_path, label_arr)

    # meta.json
    meta = {
        "source": CONFIG["ROOT_DIR"],
        "split": split_name,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "num_segments": len(all_eegs),
        "num_subjects": len(file_list),
        "arrays": {
            "eeg":    {"shape": list(eeg_arr.shape),    "dtype": str(eeg_arr.dtype)},
            "coords": {"shape": list(coords_arr.shape), "dtype": str(coords_arr.dtype)},
            "label":  {"shape": list(label_arr.shape),  "dtype": str(label_arr.dtype)},
        },
        "files": {
            "eeg":    {"path": eeg_path,    "size": _human_bytes(os.path.getsize(eeg_path))},
            "coords": {"path": coords_path, "size": _human_bytes(os.path.getsize(coords_path))},
            "label":  {"path": label_path,  "size": _human_bytes(os.path.getsize(label_path))},
        },
        "config": {
            "original_sr": CONFIG["ORIGINAL_SR"],
            "target_sr": CONFIG["TARGET_SR"],
            "bandpass": list(CONFIG["BANDPASS"]),
            "notch_freq": CONFIG["NOTCH_FREQ"],
            "clip_limit": CONFIG["CLIP_LIMIT"],
            "segment_seconds": CONFIG["SEGMENT_SECONDS"],
        },
        "label_sequence": TRIAL_LABELS,
        "channels": CHANNELS_32,
    }
    with open(os.path.join(split_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    # 라벨 분포 출력
    unique, counts = np.unique(label_arr, return_counts=True)
    dist_str = ", ".join(f"{u}:{c}" for u, c in zip(unique, counts))

    print(f"[{split_name}] Saved {len(all_eegs)} segments ({len(file_list)} subjects)")
    print(f"  - eeg   : {_human_bytes(os.path.getsize(eeg_path))}  shape={eeg_arr.shape}")
    print(f"  - coords: {_human_bytes(os.path.getsize(coords_path))}  shape={coords_arr.shape}")
    print(f"  - label : {_human_bytes(os.path.getsize(label_path))}  shape={label_arr.shape}")
    print(f"  - label dist: {dist_str}")
    print()

    return len(all_eegs)


# ==============================================================================
# 6. 메인 실행부
# ==============================================================================
if __name__ == "__main__":
    os.makedirs(CONFIG["OUTPUT_DIR"], exist_ok=True)

    # 파일 수집 및 정렬
    all_files = sorted(
        glob.glob(os.path.join(CONFIG["ROOT_DIR"], CONFIG["FILE_EXT"]))
    )
    print(f"[init] Found {len(all_files)} pkl files")
    print(f"[init] Labels per trial ({len(TRIAL_LABELS)} trials): {TRIAL_LABELS}")
    print(f"[init] Segment: {CONFIG['SEGMENT_SECONDS']}s → {int(CONFIG['SEGMENT_SECONDS'] * CONFIG['TARGET_SR'])} samples @ {CONFIG['TARGET_SR']}Hz\n")

    # Split 분배: sub000~079 → train, sub080~099 → val, sub100~122 → test
    train_files, val_files, test_files = [], [], []
    for f in all_files:
        subj_id = get_subject_id(f)
        if subj_id is None:
            continue
        if 0 <= subj_id <= 79:
            train_files.append(f)
        elif 80 <= subj_id <= 99:
            val_files.append(f)
        else:  # 100~122
            test_files.append(f)

    print(f"Split: Train={len(train_files)}, Val={len(val_files)}, Test={len(test_files)}\n")

    total = 0
    total += process_and_save_npy(train_files, os.path.join(CONFIG["OUTPUT_DIR"], "train"), "train")
    total += process_and_save_npy(val_files, os.path.join(CONFIG["OUTPUT_DIR"], "val"), "val")
    total += process_and_save_npy(test_files, os.path.join(CONFIG["OUTPUT_DIR"], "test"), "test")

    print("=" * 60)
    print(f"[Done] All splits processed. Total segments: {total}")
    print("=" * 60)