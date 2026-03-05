import os
import glob
import warnings
import re
import numpy as np
import scipy.signal as signal
import math
import mne
from multiprocessing import Pool
from tqdm import tqdm

# ==============================================================================
# [설정 영역]
# ==============================================================================
CONFIG = {
    "ROOT_DIR": "D:/open_eeg_eval/physionet_MI",           
    "OUTPUT_DIR": "D:/open_eeg_eval/physionet_MI_pp/", 
    "montage": "standard_1005",
    "FILE_EXT": "*.edf", 

    "TARGET_SR": 200,        
    "BANDPASS": (0.5, 75.0), 
    "NOTCH_Q": 30.0,         
    "NOTCH_FREQ": 60.0,      
    "CLIP_LIMIT": 15.0,      

    # [NEW] MI 데이터는 이벤트(자극) 발생 후 몇 초를 볼 것인지가 중요합니다.
    # PhysioNet MI의 Task 길이는 대략 4.1초입니다.
    "EPOCH_SECONDS": 4.0,    
    "NUM_WORKERS": max(1, os.cpu_count() - 2)
}

# 19개 표준 타겟 채널 (이전 설정 유지)
TARGET_CHANNELS = [
    'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
    'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 'Fz', 'Cz', 'Pz'
]

# ==============================================================================
# 1. 헬퍼 함수 (경로 분석 및 레이블 매핑)
# ==============================================================================
def get_subject_and_run(file_path):
    """ 파일 경로에서 피험자 번호와 Run 번호를 추출합니다. (예: S001R03.edf -> subj: 1, run: 3) """
    filename = os.path.basename(file_path)
    match = re.search(r'S(\d+)R(\d+)', filename, re.IGNORECASE)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None

def map_event_to_label(event_str, run_num):
    """ Run 번호와 Annotation(T0, T1, T2)에 따라 최종 레이블을 0~4로 매핑합니다. """
    if event_str == 'T0': return None # Rest
    
    # Left / Right Fist Task
    if run_num in [4, 8, 12]:
    # if run_num in [3, 4, 7, 8, 11, 12]:
        if event_str == 'T1': return 0 # Left fist
        if event_str == 'T2': return 1 # Right fist
        
    # Both Fists / Both Feet Task
    elif run_num in [6, 10, 14]:
    # elif run_num in [5, 6, 9, 10, 13, 14]:
        if event_str == 'T1': return 2 # Both fists
        if event_str == 'T2': return 3 # Both feet
        
    return None # Run 1, 2(Baseline)이거나 알 수 없는 경우 None

# (SmartEEGPreprocessor 클래스는 이전 답변과 동일하므로 생략 없이 그대로 유지한다고 가정)
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
        if Wn_high >= 1.0: Wn_high = 0.99 
        
        sos = signal.butter(3, [Wn_low, Wn_high], btype='band', analog=False, output='sos')
        eeg_data = signal.sosfiltfilt(sos, eeg_data, axis=-1)

        if original_sr != self.target_sr:
            gcd = math.gcd(int(original_sr), int(self.target_sr))
            eeg_data = signal.resample_poly(eeg_data, int(self.target_sr // gcd), int(original_sr // gcd), axis=-1)

        mean = np.mean(eeg_data, axis=-1, keepdims=True)
        std = np.std(eeg_data, axis=-1, keepdims=True)
        eeg_data = (eeg_data - mean) / (std + 1e-8)
        return np.clip(eeg_data.astype(np.float16), -self.clip_limit, self.clip_limit)

def clean_channel_names(raw):
    mapping = {}
    for ch_name in raw.ch_names:
        # PhysioNet 특유의 온점(.)이나 쓸데없는 문자 제거
        clean_name = re.sub(r'[^A-Za-z0-9]', '', ch_name).strip().upper()
        
        name_map = {
            'FP1': 'Fp1', 'FP2': 'Fp2', 'FZ': 'Fz', 'CZ': 'Cz', 'PZ': 'Pz', 'OZ': 'Oz',
            'T3': 'T7', 'T4': 'T8', 'T5': 'P7', 'T6': 'P8',
            'T7': 'T7', 'T8': 'T8' 
        }
        final_name = name_map.get(clean_name, clean_name.capitalize()) 
        if ch_name != final_name: mapping[ch_name] = final_name

    final_mapping = {k: v for k, v in mapping.items() if v not in raw.ch_names or k == v}
    try: raw.rename_channels(final_mapping)
    except: pass 
    return raw

# ==============================================================================
# 2. Worker 함수 (Event 기반 Slicing)
# ==============================================================================
def process_single_file(file_path):
    try:
        subj, run = get_subject_and_run(file_path)
        if subj is None or run is None: return None
        
        # Run 1, 2는 Baseline(눈 감기/뜨기)이므로 모델의 MI 분류를 위해 스킵
        if run in [1, 2]: return None

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)

        raw = clean_channel_names(raw)
        
        missing_channels = [ch for ch in TARGET_CHANNELS if ch not in raw.ch_names]
        if missing_channels: return None
        
        raw.pick(TARGET_CHANNELS)
        raw.reorder_channels(TARGET_CHANNELS)
        
        # [NEW] 어노테이션에서 이벤트 추출
        # events_from_annotations는 (이벤트배열, 이벤트ID딕셔너리)를 반환합니다.
        events, event_dict = mne.events_from_annotations(raw, verbose=False)
        
        # MNE가 임의로 부여한 ID(1, 2, 3 등)를 다시 'T0', 'T1', 'T2' 문자열로 뒤집기 위한 딕셔너리
        id_to_event = {v: k for k, v in event_dict.items()}

        data = raw.get_data().astype(np.float32) 
        original_sr = raw.info['sfreq']
        
        # 연속 데이터 전체 전처리 (필터링, 리샘플링)
        preprocessor = SmartEEGPreprocessor(CONFIG["TARGET_SR"], CONFIG["BANDPASS"], CONFIG["CLIP_LIMIT"])
        processed_full = preprocessor.apply(data, original_sr)
        
        epoch_samples = int(CONFIG["EPOCH_SECONDS"] * CONFIG["TARGET_SR"])
        total_length = processed_full.shape[-1]
        
        # 고정 좌표 추출
        valid_coords = [raw.info['chs'][raw.ch_names.index(ch)]['loc'][:3] for ch in TARGET_CHANNELS]
        coords_array = np.array(valid_coords, dtype=np.float16)
        
        eeg_segments, coord_segments, labels = [], [], []
        
        # [NEW] 이벤트 발생 시점을 기준으로 데이터 Slicing
        for i in range(len(events)):
            onset_orig = events[i, 0] # 원본 샘플링 레이트 기준 발생 인덱스
            event_id = events[i, 2]   
            
            event_str = id_to_event.get(event_id, '')
            final_label = map_event_to_label(event_str, run)
            
            if final_label is None: continue
            
            # 리샘플링된 데이터에 맞춰 onset 인덱스 변환
            onset_target = int(onset_orig * (CONFIG["TARGET_SR"] / original_sr))
            
            # Epoch 길이만큼 자르기
            end_idx = onset_target + epoch_samples
            if end_idx > total_length: continue # 파일 끝자락에서 발생한 이벤트는 스킵
            
            segment_data = processed_full[:, onset_target:end_idx]
            
            eeg_segments.append(segment_data)
            coord_segments.append(coords_array)
            labels.append(final_label)
            
        return eeg_segments, coord_segments, labels

    except Exception as e:
        return None

# ==============================================================================
# 3. 메인 실행부 (Train/Val/Test 분리)
# ==============================================================================
def process_and_save(file_list, save_path):
    if not file_list: return
    
    all_eegs, all_coords, all_labels = [], [], []
    with Pool(CONFIG["NUM_WORKERS"]) as pool:
        for results in tqdm(pool.imap_unordered(process_single_file, file_list), total=len(file_list), desc=os.path.basename(save_path)):
            if results:
                eeg_segs, coord_segs, labs = results
                all_eegs.extend(eeg_segs)
                all_coords.extend(coord_segs)
                all_labels.extend(labs)

    if all_eegs:
        np.savez_compressed(save_path, eeg=np.stack(all_eegs), coords=np.stack(all_coords), label=np.array(all_labels, dtype=np.int8))
        print(f"Saved {len(all_eegs)} segments to {save_path}\n")

if __name__ == "__main__":
    os.makedirs(CONFIG["OUTPUT_DIR"], exist_ok=True)

    all_files = glob.glob(os.path.join(CONFIG["ROOT_DIR"], "**", CONFIG["FILE_EXT"]), recursive=True)
    
    train_files, val_files, test_files = [], [], []
    
    # 파일 경로에서 피험자 번호를 추출하여 정확하게 분배
    for f in all_files:
        subj, _ = get_subject_and_run(f)
        if subj is None: continue
        
        if 1 <= subj <= 70: train_files.append(f)
        elif 71 <= subj <= 89: val_files.append(f)
        elif 90 <= subj <= 109: test_files.append(f)

    print(f"Total: {len(all_files)} files -> Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}\n")

    process_and_save(train_files, os.path.join(CONFIG["OUTPUT_DIR"], "train.npz"))
    process_and_save(val_files, os.path.join(CONFIG["OUTPUT_DIR"], "val.npz"))
    process_and_save(test_files, os.path.join(CONFIG["OUTPUT_DIR"], "test.npz"))