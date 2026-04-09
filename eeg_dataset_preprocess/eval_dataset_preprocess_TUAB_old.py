import os
import glob
import warnings
import numpy as np
import scipy.signal as signal
import math
import mne
import re
from multiprocessing import Pool
from tqdm import tqdm

# ==============================================================================
# [설정 영역] 사용자의 환경에 맞게 수정하세요.
# ==============================================================================
CONFIG = {
    # --------------------------------------------------------------------------
    # 파일 경로 및 데이터 소스 설정
    # --------------------------------------------------------------------------
    # TUEV 데이터셋 최상위 폴더 (하위에 train/, eval/ 폴더가 있어야 함)
    "ROOT_DIR": "D:\\One_한양대학교\\private object minsu\\coding\\data\\TUAB\\v3.0.1\\edf\\",           
    
    # NPZ 파일이 저장될 출력 폴더
    "OUTPUT_DIR": "D:/open_eeg_eval/tuab", 
    
    "montage": "standard_1005", # 채널 좌표 매핑을 위한 몽타주 이름
    
    # 처리할 파일 확장자 
    "FILE_EXT": "*.edf", 

    # --------------------------------------------------------------------------
    # 전처리 파라미터 (공통)
    # --------------------------------------------------------------------------
    "TARGET_SR": 200,        # 목표 샘플링 레이트
    "BANDPASS": (0.5, 75.0), # (Low cut, High cut)
    "NOTCH_Q": 30.0,         # Notch Filter Q factor
    "NOTCH_FREQ": 60.0,      # None이면 자동 감지
    "CLIP_LIMIT": 15.0,      # Z-score 후 클리핑

    # 세그멘테이션 설정
    "WINDOW_SECONDS": 10,    # 10초 단위로 자르기
    "DROP_LAST": True,       # 자투리 버림
    
    # 병렬 처리 설정
    "NUM_WORKERS": max(1, os.cpu_count() - 2)
}

# ==============================================================================
# 1. 통합 데이터 로더
# ==============================================================================
def load_data_to_mne(file_path):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
        return raw
    except Exception as e:
        # print(f"[Load Error] {file_path}: {e}")
        return None

# ==============================================================================
# 2. EEG 전처리 로직 (Signal Processing)
# ==============================================================================
class SmartEEGPreprocessor:
    def __init__(self, target_sr, bandpass_freq, clip_limit):
        self.target_sr = target_sr
        self.bandpass_freq = bandpass_freq
        self.clip_limit = clip_limit

    def detect_line_noise(self, eeg_data, fs):
        try:
            if fs <= 120: return None 
            freqs, psd = signal.welch(eeg_data, fs, nperseg=int(fs), axis=-1)
            mean_psd = np.mean(psd, axis=0)
            
            idx_50 = np.argmin(np.abs(freqs - 50))
            idx_60 = np.argmin(np.abs(freqs - 60))
            if idx_50 >= len(mean_psd) or idx_60 >= len(mean_psd): return None

            power_50, power_60 = mean_psd[idx_50], mean_psd[idx_60]
            baseline_50 = np.median(mean_psd[np.argmin(np.abs(freqs - 45)):np.argmin(np.abs(freqs - 55))])
            baseline_60 = np.median(mean_psd[np.argmin(np.abs(freqs - 55)):np.argmin(np.abs(freqs - 65))])

            threshold = 5.0 
            is_50_noise = power_50 > (baseline_50 * threshold)
            is_60_noise = power_60 > (baseline_60 * threshold)

            if is_60_noise and not is_50_noise: return 60.0
            elif is_50_noise and not is_60_noise: return 50.0
            elif is_50_noise and is_60_noise:
                ratio_50 = power_50 / (baseline_50 + 1e-12)
                ratio_60 = power_60 / (baseline_60 + 1e-12)
                return 60.0 if ratio_60 > ratio_50 else 50.0
            return None
        except Exception:
            return None

    def apply(self, eeg_data, original_sr):
        nyq = 0.5 * original_sr
        low_cut, high_cut = self.bandpass_freq

        if high_cut >= nyq:
            adjusted_high = nyq - 1.0 
            if adjusted_high <= low_cut: adjusted_high = nyq - 0.1
        else:
            adjusted_high = high_cut

        line_freq = CONFIG["NOTCH_FREQ"] if CONFIG["NOTCH_FREQ"] is not None else self.detect_line_noise(eeg_data, original_sr)
        if line_freq and line_freq < nyq:
            b_notch, a_notch = signal.iirnotch(line_freq, Q=CONFIG["NOTCH_Q"], fs=original_sr)
            eeg_data = signal.filtfilt(b_notch, a_notch, eeg_data, axis=-1)

        Wn_low = low_cut / nyq
        Wn_high = adjusted_high / nyq
        if Wn_high >= 1.0: Wn_high = 0.99 
        
        sos = signal.butter(3, [Wn_low, Wn_high], btype='band', analog=False, output='sos')
        eeg_data = signal.sosfiltfilt(sos, eeg_data, axis=-1)

        if original_sr != self.target_sr:
            gcd = math.gcd(int(original_sr), int(self.target_sr))
            up = int(self.target_sr // gcd)
            down = int(original_sr // gcd)
            eeg_data = signal.resample_poly(eeg_data, up, down, axis=-1)

        mean = np.mean(eeg_data, axis=-1, keepdims=True)
        std = np.std(eeg_data, axis=-1, keepdims=True)
        eeg_data = (eeg_data - mean) / (std + 1e-8)

        eeg_data = eeg_data.astype(np.float16)
        eeg_data = np.clip(eeg_data, -self.clip_limit, self.clip_limit)
        return eeg_data

# ==============================================================================
# 3. 채널 이름 정규화 
# ==============================================================================
def clean_channel_names(raw):
    # 기존 원본 로직 유지
    mapping = {}
    for ch_name in raw.ch_names:
        clean_name = re.sub(r'(?i)(EEG|[-_]?(REF|LE|MON|AVG|M1|M2|A1|A2)$)', '', ch_name).strip()
        clean_name = clean_name.upper()
        
        name_map = {
            'FP1': 'Fp1', 'FP2': 'Fp2', 'FZ': 'Fz', 'CZ': 'Cz', 'PZ': 'Pz', 'OZ': 'Oz',
            'T3': 'T7', 'T4': 'T8', 'T5': 'P7', 'T6': 'P8',
            'T7': 'T7', 'T8': 'T8', # 'T1': 'FT9', 'T2': 'FT10',
        }
        final_name = name_map.get(clean_name, clean_name.capitalize()) 
        
        if ch_name != final_name:
            mapping[ch_name] = final_name

    final_mapping = {k: v for k, v in mapping.items() if v not in raw.ch_names or k == v}
    try:
        raw.rename_channels(final_mapping)
    except:
        pass 
    return raw

def get_valid_channel_indices(raw):
    # 기존 원본 로직 유지
    valid_names = []
    valid_coords = []
    
    for ch_name in raw.ch_names:
        if ch_name not in raw.info['chs'][raw.ch_names.index(ch_name)]['ch_name']: continue
        ch_idx = raw.ch_names.index(ch_name)
        loc = raw.info['chs'][ch_idx]['loc'][:3]
        
        if not np.all(np.isnan(loc)) and not np.all(loc == 0):
            valid_names.append(ch_name)
            valid_coords.append(loc)
            
    return valid_names, valid_coords

# ==============================================================================
# 4. Worker 함수 (단일 파일 처리 -> EEG 세그먼트, 좌표, Label 반환)
# ==============================================================================
def process_single_file(file_path):
    try:
        # [NEW] 경로를 분석하여 Label 추출 (abnormal = 1, normal = 0)
        # normality_folder = os.path.basename(os.path.dirname(file_path)).lower()
        if 'abnormal' in file_path:
            label = 1
        elif 'normal' in file_path:
            label = 0
        else:
            print(f"[Warning] Cannot determine label from path: {file_path}")
            return None

        raw = load_data_to_mne(file_path)
        if raw is None: return None

        if raw._data.dtype == np.float64:
             raw._data = raw._data.astype(np.float32)

        raw = clean_channel_names(raw)
        
        if 'eeg' in raw:
            try: raw.pick("eeg", exclude="bads")
            except ValueError: pass
        # unwanted_chs = ['Roc', 'Loc', 'Ekg1', 'Photic', 'Ibi', 'Bursts', 'Suppr']
        # raw.drop_channels([ch for ch in unwanted_chs if ch in raw.ch_names])
        # 1. 사용할 19개 표준 채널 리스트 정의 (순서도 이대로 고정됩니다)
        TARGET_CHANNELS = [
            'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
            'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 'Fz', 'Cz', 'Pz'
        ]

        # 2. 현재 raw 데이터에 타겟 채널이 모두 존재하는지 확인
        missing_channels = [ch for ch in TARGET_CHANNELS if ch not in raw.ch_names]

        if missing_channels:
            print(f"누락된 채널이 있습니다: {missing_channels}")
            # 파일 스킵 처리 (return None 등)
        else:
            # 3. 원하는 19개 채널만 뽑아내기 (나머지 A1, A2, T1 등은 다 버려짐)
            raw.pick(TARGET_CHANNELS)
            # 4. [매우 중요] 채널 순서를 TARGET_CHANNELS 리스트와 똑같이 강제 정렬
            raw.reorder_channels(TARGET_CHANNELS)
    
        try:
            montage = mne.channels.make_standard_montage(CONFIG["montage"])
            raw.set_montage(montage, match_case=False, on_missing='ignore')
        except: pass
        
        if len(raw.ch_names) < 3: return None



        valid_names, valid_coords = get_valid_channel_indices(raw)
        if len(valid_names) < 3: return None

        raw.pick(valid_names)

        data = raw.get_data() 
        sfreq = raw.info['sfreq']
        
        preprocessor = SmartEEGPreprocessor(CONFIG["TARGET_SR"], CONFIG["BANDPASS"], CONFIG["CLIP_LIMIT"])
        processed_full = preprocessor.apply(data, sfreq)

        window_samples = int(CONFIG["WINDOW_SECONDS"] * CONFIG["TARGET_SR"])
        total_length = processed_full.shape[-1]
        
        if total_length < window_samples: return None
        
        num_segments = total_length // window_samples
        coords_array = np.array(valid_coords, dtype=np.float16)
        
        if processed_full.shape[-2] != len(coords_array): return None
        
        # 분할된 데이터를 담을 리스트
        eeg_segments = []
        coord_segments = []
        labels = []
        
        for i in range(num_segments):
            start_idx = i * window_samples
            end_idx = start_idx + window_samples
            
            segment_data = processed_full[:, start_idx:end_idx]
            
            eeg_segments.append(segment_data)
            coord_segments.append(coords_array)
            labels.append(label)
            
        return eeg_segments, coord_segments, labels

    except Exception as e:
        # print(f"[Error] {file_path}: {e}")
        return None

# ==============================================================================
# 5. 메인 실행부 (병렬 처리 후 npz 저장)
# ==============================================================================
def process_and_save(file_list, save_path):
    all_eegs = []
    all_coords = []
    all_labels = []

    if not file_list:
        print(f"No files to process for {save_path}")
        return

    print(f"\nProcessing {len(file_list)} files for {os.path.basename(save_path)}...")
    with Pool(CONFIG["NUM_WORKERS"]) as pool:
        for results in tqdm(pool.imap_unordered(process_single_file, file_list), total=len(file_list)):
            if results is None:
                continue
            
            eeg_segs, coord_segs, labels = results
            all_eegs.extend(eeg_segs)
            all_coords.extend(coord_segs)
            all_labels.extend(labels)

    if not all_eegs:
        print(f"No valid data extracted for {save_path}")
        return

    # 리스트를 하나의 numpy 배열로 변환
    final_eeg = np.stack(all_eegs)     # Shape: (N_samples, Channels, Time)
    final_coords = np.stack(all_coords) # Shape: (N_samples, Channels, 3)
    final_labels = np.array(all_labels, dtype=np.int8) # Shape: (N_samples,)

    print(f"Saving {final_eeg.shape[0]} segments to {save_path}...")
    np.savez_compressed(
        save_path, 
        eeg=final_eeg, 
        coords=final_coords, 
        label=final_labels
    )
    print(f"Done! {final_eeg.shape[0]} Samples, File size: {os.path.getsize(save_path) / (1024*1024):.2f} MB")

if __name__ == "__main__":
    os.makedirs(CONFIG["OUTPUT_DIR"], exist_ok=True)

    # 1. 파일 검색
    search_pattern = os.path.join(CONFIG["ROOT_DIR"], "**", CONFIG["FILE_EXT"])
    all_files = glob.glob(search_pattern, recursive=True)
    
    # 2. Train / Test 리스트 분리
    train_files = [f for f in all_files if 'train' in f.lower()]
    test_files  = [f for f in all_files if 'eval' in f.lower()]
    
    print(f"Total files found: {len(all_files)}")
    print(f" - Train files: {len(train_files)}")
    print(f" - Eval(Test) files: {len(test_files)}")

    # 3. 각각 처리 및 저장
    process_and_save(train_files, os.path.join(CONFIG["OUTPUT_DIR"], "train.npz"))
    process_and_save(test_files, os.path.join(CONFIG["OUTPUT_DIR"], "test.npz"))