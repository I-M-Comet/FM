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
    "ROOT_DIR": "D:/open_eeg/mat",
    "OUTPUT_DIR": "D:/open_eeg_eval/MAT_npy/",
    "montage": "standard_1005",
    "FILE_EXT": "*.edf",

    "TARGET_SR": 200,
    "BANDPASS": (0.5, 45.0),
    "NOTCH_Q": 30.0,
    "NOTCH_FREQ": 50.0,
    "CLIP_LIMIT": 15.0,

    "SEGMENT_SECONDS": 5.0,

    "NUM_WORKERS": max(1, os.cpu_count() - 2),
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
# 2. 채널 정리 & 좌표
# ==============================================================================
def clean_channel_names(raw):
    mapping = {}
    for ch_name in raw.ch_names:
        clean_name = re.sub(r'[^A-Za-z0-9]', '', ch_name).strip().upper()
        clean_name = re.sub(r'^EEG', '', clean_name)
        name_map = {
            'FP1': 'Fp1', 'FP2': 'Fp2', 'FZ': 'Fz', 'CZ': 'Cz', 'PZ': 'Pz', 'OZ': 'Oz',
            'T3': 'T7', 'T4': 'T8', 'T5': 'P7', 'T6': 'P8',
            'T7': 'T7', 'T8': 'T8',
        }
        final_name = name_map.get(clean_name, clean_name.capitalize())
        if ch_name != final_name:
            mapping[ch_name] = final_name

    final_mapping = {k: v for k, v in mapping.items() if v not in raw.ch_names or k == v}
    try:
        raw.rename_channels(final_mapping)
    except Exception:
        pass
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


def get_valid_channel_data_and_coords(raw):
    """몽타주 매칭된 채널만 추출하여 (data, coords) 반환"""
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
        # 몽타주 매칭 실패 시 전체 채널 + 이름 기반 좌표 생성
        coords_array = build_channel_coords(raw.ch_names)
        data = raw.get_data().astype(np.float32)

    return data, coords_array


# ==============================================================================
# 3. 파일명 파싱
# ==============================================================================
def parse_filename(file_path):
    """
    파일명: "Subject{num}_{task}.edf"
    반환: (subject_num, task_num) 또는 (None, None)
    """
    filename = os.path.basename(file_path)
    match = re.match(r'Subject(\d+)_(\d+)', filename, re.IGNORECASE)
    if match:
        subj_num = int(match.group(1))
        task_num = int(match.group(2))
        return subj_num, task_num
    return None, None


# ==============================================================================
# 4. Worker 함수
# ==============================================================================
_PREPROCESSOR = None


def _init_worker():
    global _PREPROCESSOR
    _PREPROCESSOR = SmartEEGPreprocessor(
        CONFIG["TARGET_SR"], CONFIG["BANDPASS"], CONFIG["CLIP_LIMIT"]
    )


def process_single_file(file_path):
    """
    EDF 파일 1개 → 전처리 → 5초 세그먼트 분할
    반환: (eeg_segments, coord_segments, labels) 또는 None
    """
    global _PREPROCESSOR
    try:
        subj_num, task_num = parse_filename(file_path)
        if subj_num is None:
            return None

        # 라벨: task 1 → 0, task 2 → 1
        label = task_num - 1

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)

        raw = clean_channel_names(raw)
        # print(raw.ch_names)
        # 몽타주 설정 (파일마다)
        try:
            montage = mne.channels.make_standard_montage(CONFIG["montage"])
            raw.set_montage(montage, match_case=False, on_missing='ignore')
        except Exception:
            pass

        original_sr = raw.info['sfreq']
        data, coords_array = get_valid_channel_data_and_coords(raw)

        # 전처리
        processed = _PREPROCESSOR.apply(data, original_sr)

        # 5초 세그먼트 분할
        seg_samples = int(CONFIG["SEGMENT_SECONDS"] * CONFIG["TARGET_SR"])
        total_samples = processed.shape[-1]
        num_segs = total_samples // seg_samples

        if num_segs == 0:
            return None

        if processed.shape[0] != len(coords_array):
            print(f"[SKIP] {file_path}: channel mismatch data={processed.shape[0]} coords={len(coords_array)}")
            return None

        eeg_segments, coord_segments, labels = [], [], []

        for s in range(num_segs):
            start = s * seg_samples
            end = start + seg_samples
            eeg_segments.append(processed[:, start:end])
            coord_segments.append(coords_array)
            labels.append(label)

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
            "segment_seconds": CONFIG["SEGMENT_SECONDS"],
        },
        "label_mapping": {"task_1": 0, "task_2": 1},
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
    print(f"  - label dist (0=task1, 1=task2): {dist_str}")
    print()

    return len(all_eegs)


# ==============================================================================
# 6. 메인 실행부
# ==============================================================================
if __name__ == "__main__":
    os.makedirs(CONFIG["OUTPUT_DIR"], exist_ok=True)

    # 파일 수집
    all_files = sorted(
        glob.glob(os.path.join(CONFIG["ROOT_DIR"], "**", CONFIG["FILE_EXT"]), recursive=True)
    )
    print(f"[init] Found {len(all_files)} EDF files")
    print(f"[init] Segment: {CONFIG['SEGMENT_SECONDS']}s → "
          f"{int(CONFIG['SEGMENT_SECONDS'] * CONFIG['TARGET_SR'])} samples @ {CONFIG['TARGET_SR']}Hz")
    print(f"[init] Labels: task 1 → 0, task 2 → 1\n")

    # Subject 번호 기반 split
    # Subject 1~28 → train, 29~32 → val, 33~36 → test
    train_files, val_files, test_files = [], [], []

    for f in all_files:
        subj_num, task_num = parse_filename(f)
        if subj_num is None:
            continue
        if 1 <= subj_num <= 28:
            train_files.append(f)
        elif 29 <= subj_num <= 32:
            val_files.append(f)
        elif 33 <= subj_num <= 36:
            test_files.append(f)

    print(f"[split] Train: {len(train_files)} files (Subject 1-28)")
    print(f"[split] Val  : {len(val_files)} files (Subject 29-32)")
    print(f"[split] Test : {len(test_files)} files (Subject 33-36)\n")

    total = 0
    total += process_and_save_npy(train_files, os.path.join(CONFIG["OUTPUT_DIR"], "train"), "train")
    total += process_and_save_npy(val_files, os.path.join(CONFIG["OUTPUT_DIR"], "val"), "val")
    total += process_and_save_npy(test_files, os.path.join(CONFIG["OUTPUT_DIR"], "test"), "test")

    print("=" * 60)
    print(f"[Done] All splits processed. Total segments: {total}")
    print("=" * 60)