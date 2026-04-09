#!/usr/bin/env python3
"""
eval_dataset_preprocess_BCIC4_2a_moabb.py

MOABB 라이브러리를 사용하여 BCI Competition IV 2a 데이터셋을 다운로드 및 전처리합니다.
Train과 Eval(정답 복원됨) 세션을 모두 사용하여 데이터를 2배로 확보합니다.
"""

import os
import json
import time
import math
import warnings
import numpy as np
import scipy.signal as signal
import mne
from moabb.datasets import BNCI2014_001
from tqdm import tqdm

# ==============================================================================
# [설정 영역]
# ==============================================================================
CONFIG = {
    "OUTPUT_DIR": "D:/open_eeg_eval/BCIC_IV_2a_npy/",
    "montage": "standard_1005",

    "TARGET_SR": 200,
    "BANDPASS": (0.5, 75.0),
    "NOTCH_Q": 30.0,
    "NOTCH_FREQ": 50.0,
    "CLIP_LIMIT": 15.0,

    "EPOCH_SECONDS": 4.0,
}

# MOABB에서 부여하는 표준 Annotation 라벨 맵핑
MOABB_EVENT_MAP = {
    'left_hand': 0,
    'right_hand': 1,
    'feet': 2,
    'tongue': 3
}

# ==============================================================================
# 1. 전처리기 (기존과 동일)
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
# 2. 직접 .npy 저장 (구조 변경)
# ==============================================================================
def _human_bytes(n: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    x = float(n)
    for u in units:
        if x < 1024.0 or u == units[-1]:
            return f"{x:.2f}{u}"
        x /= 1024.0
    return f"{x:.2f}TB"

def save_npy_split(eeg_list, coords_list, label_list, split_dir, split_name):
    if not eeg_list:
        print(f"[{split_name}] No segments found. Skipping.")
        return 0

    os.makedirs(split_dir, exist_ok=True)

    eeg_arr = np.stack(eeg_list)
    coords_arr = np.stack(coords_list)
    label_arr = np.array(label_list, dtype=np.int8)

    eeg_path = os.path.join(split_dir, "eeg.npy")
    coords_path = os.path.join(split_dir, "coords.npy")
    label_path = os.path.join(split_dir, "label.npy")

    np.save(eeg_path, eeg_arr)
    np.save(coords_path, coords_arr)
    np.save(label_path, label_arr)

    meta = {
        "source": "MOABB_BNCI2014_001",
        "split": split_name,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "num_segments": len(eeg_list),
        "arrays": {
            "eeg": {"shape": list(eeg_arr.shape), "dtype": str(eeg_arr.dtype)},
            "coords": {"shape": list(coords_arr.shape), "dtype": str(coords_arr.dtype)},
            "label": {"shape": list(label_arr.shape), "dtype": str(label_arr.dtype)},
        },
        "config": CONFIG,
        "label_mapping": MOABB_EVENT_MAP
    }
    with open(os.path.join(split_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    unique, counts = np.unique(label_arr, return_counts=True)
    dist_str = ", ".join(f"{u}:{c}" for u, c in zip(unique, counts))

    print(f"[{split_name}] Saved {len(eeg_list)} segments")
    print(f"  - eeg shape: {eeg_arr.shape}")
    print(f"  - label dist (0=L, 1=R, 2=F, 3=T): {dist_str}\n")
    return len(eeg_list)

# ==============================================================================
# 3. 메인 실행부
# ==============================================================================
if __name__ == "__main__":
    os.makedirs(CONFIG["OUTPUT_DIR"], exist_ok=True)
    
    # 1. MOABB 데이터셋 로드 (BCI Comp IV 2a)
    print("[init] Downloading/Loading BCI Comp IV 2a via MOABB...")
    dataset = BNCI2014_001()
    
    preprocessor = SmartEEGPreprocessor(CONFIG["TARGET_SR"], CONFIG["BANDPASS"], CONFIG["CLIP_LIMIT"])
    epoch_samples = int(CONFIG["EPOCH_SECONDS"] * CONFIG["TARGET_SR"])

    # Split별 데이터 저장용 리스트
    splits = {
        "train": {"eeg": [], "coords": [], "label": []}, # Subj 1~5
        "val":   {"eeg": [], "coords": [], "label": []}, # Subj 6~7
        "test":  {"eeg": [], "coords": [], "label": []}  # Subj 8~9
    }

    # MOABB 캐시 충돌을 막기 위해 순차적(Sequential)으로 처리합니다.
    # (어차피 메모리에 로드된 데이터를 처리하므로 매우 빠릅니다)
    for subj in tqdm(range(1, 10), desc="Processing Subjects"):
        # 대상 스플릿 결정
        if 1 <= subj <= 5: split_key = "train"
        elif 6 <= subj <= 7: split_key = "val"
        else: split_key = "test"

        # 피험자 데이터 가져오기 (Train/Eval 세션 모두 포함됨!)
        subj_data = dataset.get_data(subjects=[subj])[subj]

        for session_name, session_runs in subj_data.items():
            for run_name, raw in session_runs.items():
                
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    # 1. EOG 채널 제거 및 EEG만 선택 (MOABB가 이미 EOG를 식별해둠)
                    raw.pick_types(eeg=True, eog=False)
                    
                    # 2. 몽타주 설정 및 좌표 추출
                    raw.set_montage(CONFIG["montage"], match_case=False, on_missing='ignore')
                    valid_coords = [raw.info['chs'][i]['loc'][:3] for i in range(len(raw.ch_names))]
                    coords_array = np.array(valid_coords, dtype=np.float16)
                    
                    # 3. 연속 데이터 전체 전처리
                    original_sr = raw.info['sfreq']
                    data = raw.get_data().astype(np.float32)
                    processed_full = preprocessor.apply(data, original_sr)
                    total_length = processed_full.shape[-1]

                    # 4. 이벤트(라벨) 추출 (MOABB가 평가용 데이터 라벨도 전부 복원해 줌)
                    events, event_dict = mne.events_from_annotations(raw, verbose=False)
                    id_to_annot = {v: k for k, v in event_dict.items()}

                # 5. 에포킹 (Epoching)
                for i in range(len(events)):
                    onset_orig = events[i, 0]
                    event_id = events[i, 2]
                    annot_str = id_to_annot.get(event_id, '')

                    # MOABB 라벨 문자열('left_hand' 등)을 숫자로 매핑
                    if annot_str not in MOABB_EVENT_MAP:
                        continue
                    final_label = MOABB_EVENT_MAP[annot_str]

                    # 리샘플링된 SR에 맞게 Onset 조정
                    onset_target = int(onset_orig * (CONFIG["TARGET_SR"] / original_sr))
                    end_idx = onset_target + epoch_samples

                    if end_idx > total_length:
                        continue

                    # 잘라낸 세그먼트를 해당 스플릿에 추가
                    segment_data = processed_full[:, onset_target:end_idx]
                    splits[split_key]["eeg"].append(segment_data)
                    splits[split_key]["coords"].append(coords_array)
                    splits[split_key]["label"].append(final_label)

    print("\n" + "=" * 60)
    print("Saving to .npy files...")
    total_segments = 0
    for split_key in ["train", "val", "test"]:
        total_segments += save_npy_split(
            splits[split_key]["eeg"],
            splits[split_key]["coords"],
            splits[split_key]["label"],
            os.path.join(CONFIG["OUTPUT_DIR"], split_key),
            split_key
        )
    
    print("=" * 60)
    print(f"[Done] All splits processed. Total segments: {total_segments}")
    print("=" * 60)