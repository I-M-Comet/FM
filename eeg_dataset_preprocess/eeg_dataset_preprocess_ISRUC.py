import os
import glob
import re
import warnings
import numpy as np
import pandas as pd
import scipy.signal as signal
import math
import mne
from multiprocessing import Pool
from tqdm import tqdm

# ==============================================================================
# [설정 영역]
# ==============================================================================
CONFIG = {
    "ROOT_DIR": "D:\\open_eeg_eval\\ISRUC\\archive",           
    "OUTPUT_DIR": "D:\\open_eeg_eval\\ISRUC_pp", 
    "montage": "standard_1005",

    "TARGET_SR": 200,        
    "BANDPASS": (0.5, 75.0), 
    "NOTCH_Q": 30.0,         
    "NOTCH_FREQ": 60.0,      
    "CLIP_LIMIT": 15.0,      

    "EPOCH_SECONDS": 30.0,   # 수면 1 Epoch 길이
    "NUM_WORKERS": max(1, os.cpu_count() - 2)
}

TARGET_CHANNELS = [
    'F3', 'C3', 'O1', 'F4', 'C4', 'O2'
]

# ==============================================================================
# 1. 헬퍼 함수 및 전처리 클래스
# ==============================================================================
def map_stage_to_label(stage_str):
    """ Excel의 수면 단계를 0~4 라벨로 매핑합니다. """
    stage_str = str(stage_str).strip().upper()
    stage_map = {
        'W': 0,
        'N1': 1,
        'N2': 2,
        'N3': 3,
        'R': 4
    }
    return stage_map.get(stage_str, None)

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
        active_ch = ch_name.split('-')[0]
        clean_name = re.sub(r'[^A-Za-z0-9]', '', active_ch).strip().upper()
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
# 2. Worker 함수 (피험자 1명 전체 처리)
# ==============================================================================
def process_subject(subject_id):
    try:
        raw_dir = os.path.join(CONFIG["ROOT_DIR"], "raw_data")
        target_dir = os.path.join(CONFIG["ROOT_DIR"], "target_data")
        
        # 1. 파일 경로 탐색 (.rec뿐만 아니라 이름이 이미 바뀐 .edf도 찾도록 수정)
        eeg_files = glob.glob(os.path.join(raw_dir, f"*{subject_id}*.rec")) + \
                    glob.glob(os.path.join(raw_dir, f"*{subject_id}*.edf"))
                    
        # 정확히 {subject_id}.rec 또는 .edf 로 끝나는 파일 찾기
        eeg_path = next((f for f in eeg_files if re.search(r'(?<!\d)' + str(subject_id) + r'\.(rec|edf)$', f, re.IGNORECASE)), None)
        label_path = os.path.join(target_dir, f"{subject_id}_1.xlsx")
        if not eeg_path or not os.path.exists(label_path):
            return None 

        # 2. 정답지(Excel) 로드
        df = pd.read_excel(label_path, engine='openpyxl')
        stages = df.iloc[:, 1].values
        
        # 3. MNE 에러 우회를 위한 확장자 강제 변경 (.rec -> .edf)
        if eeg_path.lower().endswith('.rec'):
            edf_path = eeg_path[:-4] + ".edf"
            os.rename(eeg_path, edf_path) # 실제 디스크의 파일 이름을 변경해버림!
            eeg_path = edf_path           # 경로 업데이트
        
        # 4. 뇌파 로드
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # 이제 무조건 .edf 확장자로 들어가므로 MNE가 에러를 뱉지 않습니다.
            raw = mne.io.read_raw_edf(eeg_path, preload=True, verbose=False)
        if subject_id == 8:
            tqdm.write(f'{subject_id} :: {raw.ch_names}')
            return None
        raw = clean_channel_names(raw)
        
        missing_channels = [ch for ch in TARGET_CHANNELS if ch not in raw.ch_names]
        if missing_channels: 
            tqdm.write(f"⚠️  Subject {subject_id}: Missing channels: {missing_channels}. Skipping... {raw.ch_names}")
            return None
            
        raw.pick(TARGET_CHANNELS)
        raw.reorder_channels(TARGET_CHANNELS)
        
        try:
            montage = mne.channels.make_standard_montage(CONFIG["montage"])
            raw.set_montage(montage, match_case=False, on_missing='ignore')
        except: pass

        data = raw.get_data().astype(np.float32) 
        original_sr = raw.info['sfreq']
        
        preprocessor = SmartEEGPreprocessor(CONFIG["TARGET_SR"], CONFIG["BANDPASS"], CONFIG["CLIP_LIMIT"])
        processed_full = preprocessor.apply(data, original_sr)
        
        epoch_samples = int(CONFIG["EPOCH_SECONDS"] * CONFIG["TARGET_SR"])
        total_length = processed_full.shape[-1]
        
        valid_coords = [raw.info['chs'][raw.ch_names.index(ch)]['loc'][:3] for ch in TARGET_CHANNELS]
        coords_array = np.array(valid_coords, dtype=np.float16)
        
        eeg_segments, coord_segments, labels = [], [], []
        
        # 3. 30초 단위로 슬라이싱하며 라벨 매핑
        for idx, stage in enumerate(stages):
            label = map_stage_to_label(stage)
            if label is None: continue # 알 수 없는 라벨(Unknown 등)은 버림
                
            start_idx = idx * epoch_samples
            end_idx = start_idx + epoch_samples
            
            # 정답지 개수가 뇌파 길이보다 많을 경우 (측정 종료 후의 정답지) 방어
            if end_idx > total_length:
                break
                
            segment_data = processed_full[:, start_idx:end_idx]
            
            eeg_segments.append(segment_data)
            coord_segments.append(coords_array)
            labels.append(label)
            
        return eeg_segments, coord_segments, labels

    except Exception as e:
        tqdm.write(f"❌ Subject {subject_id}: 처리 중 오류 발생: {e}")
        return None

# ==============================================================================
# 3. 메인 실행부
# ==============================================================================
def process_and_save(subject_ids, save_path):
    if not subject_ids: return
    
    all_eegs, all_coords, all_labels = [], [], []
    with Pool(CONFIG["NUM_WORKERS"]) as pool:
        for results in tqdm(pool.imap_unordered(process_subject, subject_ids), total=len(subject_ids), desc=os.path.basename(save_path)):
            if results:
                eeg_segs, coord_segs, labs = results
                all_eegs.extend(eeg_segs)
                all_coords.extend(coord_segs)
                all_labels.extend(labs)

    if all_eegs:
        np.savez_compressed(save_path, eeg=np.stack(all_eegs), coords=np.stack(all_coords), label=np.array(all_labels, dtype=np.int8))
        print(f"Saved {len(all_eegs)} epochs to {save_path}\n")
    else:
        print(f"No valid data extracted for {save_path}\n")

if __name__ == "__main__":
    os.makedirs(CONFIG["OUTPUT_DIR"], exist_ok=True)

    # 논문에 명시된 기준대로 1~100번 피험자 분할
    train_subjects = list(range(1, 81))       # 1 ~ 80
    val_subjects = list(range(81, 91))        # 81 ~ 90
    test_subjects = list(range(91, 101))      # 91 ~ 100

    print("Starting ISRUC Processing...")
    process_and_save(train_subjects, os.path.join(CONFIG["OUTPUT_DIR"], "train.npz"))
    process_and_save(val_subjects, os.path.join(CONFIG["OUTPUT_DIR"], "val.npz"))
    process_and_save(test_subjects, os.path.join(CONFIG["OUTPUT_DIR"], "test.npz"))