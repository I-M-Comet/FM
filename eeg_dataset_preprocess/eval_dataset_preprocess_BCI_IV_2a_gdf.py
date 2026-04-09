#!/usr/bin/env python3
"""
eval_dataset_preprocess_BCIC4_2a.py

BCI Competition IV 2a 데이터셋 전처리 → .npy 직접 저장

입력:
  "A{num}T.gdf" (num=01~09, T=Training만 사용)
  이벤트 기반 에포킹: 769→0(left), 770→1(right), 771→2(foot), 772→3(tongue)
  각 이벤트 onset부터 4초 자르기

출력:
  OUT_DIR/
    train/  (Subject 1~5)
      eeg.npy, coords.npy, label.npy, meta.json
    val/    (Subject 6~7)
      ...
    test/   (Subject 8~9)
      ...

Usage:
  python eval_dataset_preprocess_BCIC4_2a.py
"""

import os
import glob
import re
import json
import time
import math
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
    "ROOT_DIR": "D:/open_eeg/BCIC_IV_2a",
    "OUTPUT_DIR": "D:/open_eeg_eval/BCIC_IV_2a_npy/",
    "montage": "standard_1005",
    "FILE_EXT": "*.gdf",

    "TARGET_SR": 200,
    "BANDPASS": (0.5, 75.0),
    "NOTCH_Q": 30.0,
    "NOTCH_FREQ": 50.0,
    "CLIP_LIMIT": 15.0,

    "EPOCH_SECONDS": 4.0,

    "NUM_WORKERS": max(1, os.cpu_count() - 2),
}

# 이벤트 ID → 라벨 매핑
EVENT_LABEL_MAP = {
    769: 0,  # Cue onset left
    770: 1,  # Cue onset right
    771: 2,  # Cue onset foot
    772: 3,  # Cue onset tongue
}

# EEG-{숫자} → 표준 채널명 매핑
NUM_TO_CH_NAME = {
    '0': 'FC3', '1': 'FC1', '2': 'FCz', '3': 'FC2', '4': 'FC4',
    '5': 'C5',  '6': 'C1',  '7': 'C2',  '8': 'C6',
    '9': 'CP3', '10': 'CP1', '11': 'CPz', '12': 'CP2', '13': 'CP4',
    '14': 'P1', '15': 'P2', '16': 'POz',
}


# ==============================================================================
# 1. 전처리기
# ==============================================================================
class SmartEEGPreprocessor:
    def __init__(self, target_sr, bandpass_freq, clip_limit):
        self.target_sr = target_sr
        self.bandpass_freq = bandpass_freq
        self.clip_limit = clip_limit

    def apply(self, eeg_data, original_sr):
        nyq = 0.5 * original_sr
        low_cut, high_cut = self.bandpass_freq
        adjusted_high = (nyq - 1.0) if high_cut >= nyq else high_cut

        line_freq = CONFIG["NOTCH_FREQ"]
        if line_freq and line_freq < nyq:
            b_notch, a_notch = signal.iirnotch(line_freq, Q=CONFIG["NOTCH_Q"], fs=original_sr)
            eeg_data = signal.filtfilt(b_notch, a_notch, eeg_data, axis=-1)

        Wn_low, Wn_high = low_cut / nyq, adjusted_high / nyq
        if Wn_high >= 1.0:
            Wn_high = 0.99
        sos = signal.butter(3, [Wn_low, Wn_high], btype='band', analog=False, output='sos')
        eeg_data = signal.sosfiltfilt(sos, eeg_data, axis=-1)

        if original_sr != self.target_sr:
            gcd = math.gcd(int(original_sr), int(self.target_sr))
            eeg_data = signal.resample_poly(
                eeg_data, int(self.target_sr // gcd), int(original_sr // gcd), axis=-1
            )

        mean = np.mean(eeg_data, axis=-1, keepdims=True)
        std = np.std(eeg_data, axis=-1, keepdims=True)
        eeg_data = (eeg_data - mean) / (std + 1e-8)
        return np.clip(eeg_data.astype(np.float16), -self.clip_limit, self.clip_limit)


# ==============================================================================
# 2. 채널 정리
# ==============================================================================
def clean_bcic_channels(raw):
    """
    BCIC IV 2a 전용 채널 정리:
    1) 'EEG-Fz' → 'Fz', 'EEG-0' → 'FC3', ..., 'EEG-16' → 'POz'
    2) EOG 채널 제거
    """
    rename_map = {}
    eog_channels = []

    for ch_name in raw.ch_names:
        if ch_name.upper().startswith('EOG'):
            eog_channels.append(ch_name)
            continue

        if ch_name.startswith('EEG-'):
            suffix = ch_name[4:]  # "EEG-" 제거

            # 숫자면 표준 채널명으로 변환
            if suffix in NUM_TO_CH_NAME:
                rename_map[ch_name] = NUM_TO_CH_NAME[suffix]
            else:
                # Fz 등 이미 이름이 있는 경우 그대로
                rename_map[ch_name] = suffix

    # 채널명 변경
    if rename_map:
        try:
            raw.rename_channels(rename_map)
        except Exception as e:
            print(f"[WARN] Channel rename failed: {e}")

    # EOG 채널 제거
    if eog_channels:
        raw.drop_channels(eog_channels)

    return raw


def build_channel_coords(channel_names, montage_name="standard_1005"):
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
            coords.append(np.array([0.0, 0.0, 0.0]))

    return np.array(coords, dtype=np.float16)


# ==============================================================================
# 3. 파일명 파싱
# ==============================================================================
def parse_filename(file_path):
    """
    파일명: "A{num}T.gdf" 또는 "A{num}E.gdf"
    반환: (subject_num, task_char) 또는 (None, None)
    """
    filename = os.path.basename(file_path)
    match = re.match(r'A(\d+)([TE])', filename, re.IGNORECASE)
    if match:
        subj_num = int(match.group(1))
        task = match.group(2).upper()
        return subj_num, task
    return None, None


# ==============================================================================
# 4. Worker 함수 (전처리 → 이벤트 기반 에포킹)
# ==============================================================================
_PREPROCESSOR = None


def _init_worker():
    global _PREPROCESSOR
    _PREPROCESSOR = SmartEEGPreprocessor(
        CONFIG["TARGET_SR"], CONFIG["BANDPASS"], CONFIG["CLIP_LIMIT"]
    )


def process_single_file(file_path):
    """
    GDF 파일 1개 → 연속 데이터 전처리 → 이벤트 기반 4초 에포킹
    반환: (eeg_segments, coord_segments, labels) 또는 None
    """
    global _PREPROCESSOR
    try:
        subj_num, task = parse_filename(file_path)
        if subj_num is None:
            return None
        if task != 'T':
            return None

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = mne.io.read_raw_gdf(file_path, preload=True, verbose=False)

        # 채널 정리: EEG-{num} → 표준명, EOG 제거
        raw = clean_bcic_channels(raw)

        # 몽타주 설정 (파일마다)
        try:
            montage = mne.channels.make_standard_montage(CONFIG["montage"])
            raw.set_montage(montage, match_case=False, on_missing='ignore')
        except Exception:
            pass

        # 이벤트 추출
        events, event_dict = mne.events_from_annotations(raw, verbose=False)
        # event_dict: {'769': id1, '770': id2, ...} → 역매핑
        id_to_annotation = {v: k for k, v in event_dict.items()}

        original_sr = raw.info['sfreq']

        # 채널 좌표 생성
        valid_indices = []
        valid_coords = []
        for i, ch_name in enumerate(raw.ch_names):
            loc = raw.info['chs'][i]['loc'][:3]
            if not np.all(np.isnan(loc)) and not np.all(loc == 0):
                valid_indices.append(i)
                valid_coords.append(loc)

        if valid_indices:
            coords_array = np.array(valid_coords, dtype=np.float16)
            data = raw.get_data()[valid_indices].astype(np.float32)
        else:
            coords_array = build_channel_coords(raw.ch_names)
            data = raw.get_data().astype(np.float32)

        # 연속 데이터 전체 전처리 (필터링, 리샘플링, 정규화)
        processed_full = _PREPROCESSOR.apply(data, original_sr)

        epoch_samples = int(CONFIG["EPOCH_SECONDS"] * CONFIG["TARGET_SR"])
        total_length = processed_full.shape[-1]

        if processed_full.shape[0] != len(coords_array):
            print(f"[SKIP] {file_path}: channel mismatch "
                  f"data={processed_full.shape[0]} coords={len(coords_array)}")
            return None

        eeg_segments, coord_segments, labels = [], [], []

        # 이벤트 기반 에포킹
        for i in range(len(events)):
            onset_orig = events[i, 0]  # 원본 SR 기준 샘플 인덱스
            event_id = events[i, 2]

            # annotation 문자열 가져오기
            annotation_str = id_to_annotation.get(event_id, '')

            # annotation이 숫자 문자열일 수 있음: '769', '770' 등
            try:
                event_code = int(annotation_str)
            except (ValueError, TypeError):
                continue

            if event_code not in EVENT_LABEL_MAP:
                continue

            final_label = EVENT_LABEL_MAP[event_code]

            # 리샘플된 데이터에 맞춰 onset 변환
            onset_target = int(onset_orig * (CONFIG["TARGET_SR"] / original_sr))
            end_idx = onset_target + epoch_samples

            if end_idx > total_length:
                continue

            segment_data = processed_full[:, onset_target:end_idx]

            eeg_segments.append(segment_data)
            coord_segments.append(coords_array)
            labels.append(final_label)

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
    """파일 목록 전처리 → eeg.npy / coords.npy / label.npy 저장. 세그먼트 수 반환."""
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
        "num_files": len(file_list),
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
            "target_sr": CONFIG["TARGET_SR"],
            "bandpass": list(CONFIG["BANDPASS"]),
            "notch_freq": CONFIG["NOTCH_FREQ"],
            "clip_limit": CONFIG["CLIP_LIMIT"],
            "epoch_seconds": CONFIG["EPOCH_SECONDS"],
        },
        "label_mapping": {
            "769_left": 0, "770_right": 1,
            "771_foot": 2, "772_tongue": 3,
        },
        "channel_mapping": NUM_TO_CH_NAME,
    }
    with open(os.path.join(split_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    # 라벨 분포 출력
    unique, counts = np.unique(label_arr, return_counts=True)
    dist_str = ", ".join(f"{u}:{c}" for u, c in zip(unique, counts))

    print(f"[{split_name}] Saved {len(all_eegs)} segments ({len(file_list)} files)")
    print(f"  - eeg   : {_human_bytes(os.path.getsize(eeg_path))}  shape={eeg_arr.shape}")
    print(f"  - coords: {_human_bytes(os.path.getsize(coords_path))}  shape={coords_arr.shape}")
    print(f"  - label : {_human_bytes(os.path.getsize(label_path))}  shape={label_arr.shape}")
    print(f"  - label dist (0=left,1=right,2=foot,3=tongue): {dist_str}")
    print()

    return len(all_eegs)


# ==============================================================================
# 6. 메인 실행부
# ==============================================================================
if __name__ == "__main__":
    os.makedirs(CONFIG["OUTPUT_DIR"], exist_ok=True)

    # 파일 수집 (T 파일만)
    all_files = sorted(
        glob.glob(os.path.join(CONFIG["ROOT_DIR"], "**", CONFIG["FILE_EXT"]), recursive=True)
    )
    print(f"[init] Found {len(all_files)} GDF files (A*T.gdf)")
    print(f"[init] Epoch: {CONFIG['EPOCH_SECONDS']}s from event onset → "
          f"{int(CONFIG['EPOCH_SECONDS'] * CONFIG['TARGET_SR'])} samples @ {CONFIG['TARGET_SR']}Hz")
    print(f"[init] Events: 769→left(0), 770→right(1), 771→foot(2), 772→tongue(3)\n")

    # Subject 기반 split: 1-5 train, 6-7 val, 8-9 test
    train_files, val_files, test_files = [], [], []

    for f in all_files:
        subj_num, task = parse_filename(f)
        if subj_num is None or task != 'T':
            continue
        if 1 <= subj_num <= 5:
            train_files.append(f)
        elif 6 <= subj_num <= 7:
            val_files.append(f)
        elif 8 <= subj_num <= 9:
            test_files.append(f)

    print(f"[split] Train: {len(train_files)} files (Subject 1-5)")
    print(f"[split] Val  : {len(val_files)} files (Subject 6-7)")
    print(f"[split] Test : {len(test_files)} files (Subject 8-9)\n")

    total = 0
    total += process_and_save_npy(train_files, os.path.join(CONFIG["OUTPUT_DIR"], "train"), "train")
    total += process_and_save_npy(val_files, os.path.join(CONFIG["OUTPUT_DIR"], "val"), "val")
    total += process_and_save_npy(test_files, os.path.join(CONFIG["OUTPUT_DIR"], "test"), "test")

    print("=" * 60)
    print(f"[Done] All splits processed. Total segments: {total}")
    print("=" * 60)