import os
import glob
import re
import warnings
import numpy as np
import scipy.signal as signal
import scipy.io as sio  # [NEW] mat 파일용
import pandas as pd     # [NEW] tsv/csv 파일용
import math
import mne
import webdataset as wds
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# ==============================================================================
# [설정 영역] 사용자의 환경에 맞게 수정하세요.
# ==============================================================================
CONFIG = {
    # --------------------------------------------------------------------------
    # [NEW] 파일 형식 및 데이터 소스 설정
    # --------------------------------------------------------------------------
    "ROOT_DIR": "D:/open_eeg/siena",           # 데이터가 있는 최상위 폴더
    "OUTPUT_PATTERN": "D:/open_eeg_pp/physionet_siena/eeg-%06d.tar", # 결과 파일 패턴

    "montage": "standard_1005", # 채널 좌표 매핑을 위한 몽타주 이름 (MNE에서 지원하는 몽타주 사용 권장)
    # "montage": "biosemi64", # 채널 좌표 매핑을 위한 몽타주 이름 (MNE에서 지원하는 몽타주 사용 권장)
    
    # 처리할 파일 확장자 (".edf", ".set", ".mat", ".tsv", ".csv", ".txt", ".bdf", ".vhdr", ".gdf" 등)
    "FILE_EXT": "*.edf", 

    # [중요] EDF가 아닌 파일(MAT, TSV)을 위한 강제 설정
    # EDF 파일은 아래 두 설정이 무시됩니다 (파일 헤더 정보 사용).
    "FORCE_SFREQ": 512,   # 원본 데이터의 샘플링 레이트 (Hz)
    
    # MAT/TSV 데이터의 채널 순서대로 이름 지정 (Standard-1005 몽타주 이름 권장)
    # 데이터의 열(Column) 또는 행(Row) 순서와 일치해야 합니다.
    "FORCE_CH_NAMES": [
        'Fz', 'FCz', 'Cz', 'CPz', 'Pz', 'C1', 'C3', 'C5', 'C2', 'C4', 'C6', 'F4',
        'FC2', 'FC4', 'FC6', 'CP2', 'CP4', 'CP6', 'P4', 'F3', 'FC1', 'FC3', 'FC5',
        'CP1', 'CP3', 'CP5', 'P3'
    ],

    # [MAT 파일 전용] 데이터가 들어있는 변수 키 (예: 'data', 'X', 'eeg_data')
    "MAT_KEY": "data",

    # [TSV/CSV 파일 전용] 구분자 (탭: '\t', 쉼표: ',')
    "SEPARATOR": "\t", 
    "HAS_HEADER": False, # 파일 첫 줄에 헤더(이름)가 있으면 True, 없으면 False

    # --------------------------------------------------------------------------
    # 전처리 파라미터 (공통)
    # --------------------------------------------------------------------------
    "TARGET_SR": 200,        # 목표 샘플링 레이트
    "BANDPASS": (0.5, 75.0), # (Low cut, High cut)
    "NOTCH_Q": 30.0,         # Notch Filter Q factor
    "NOTCH_FREQ": 50.0,      # None이면 자동 감지
    "CLIP_LIMIT": 15.0,      # Z-score 후 클리핑

    # 세그멘테이션 설정
    "WINDOW_SECONDS": 60,    # 10초 단위로 자르기
    "DROP_LAST": True,       # 자투리 버림

    # 저장 설정
    "SHARD_MAX_SIZE": 1024 ** 3 * 1, # 1GB
    "SHARD_MAX_COUNT": 100000,
    
    # 병렬 처리 설정
    # "NUM_WORKERS": max(1, cpu_count() - 2)
    "NUM_WORKERS": 2
}

# ==============================================================================
# 1. 통합 데이터 로더 (EDF / MAT / TSV 지원)
# ==============================================================================
def load_data_to_mne(file_path):
    """
    다양한 포맷의 파일을 읽어 MNE Raw 객체로 변환합니다.
    """
    ext = os.path.splitext(file_path)[-1].lower()
    data = None
    sfreq = CONFIG["FORCE_SFREQ"]
    ch_names = list(CONFIG["FORCE_CH_NAMES"]) # 복사해서 사용

    try:
        # [Case 1] EDF 파일 (MNE native)
        if ext == '.edf':
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
            return raw
        
        elif ext == '.set':
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # eeglab 포맷은 .set 파일과 .fdt(데이터) 파일이 같은 폴더에 있어야 정상 로드됩니다.
                raw = mne.io.read_raw_eeglab(file_path, preload=True, verbose=False)
            return raw
        
        # [Case 1.6] BDF 파일 (BioSemi native) - 새로 추가
        elif ext == '.bdf':
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # infer_types=True를 주면 EEG와 Trigger 채널을 더 정확히 구분합니다.
                raw = mne.io.read_raw_bdf(file_path, preload=True, infer_types=True, verbose=False)
            return raw
        
        # [Case 1.7] BrainVision 파일 (.vhdr) - 새로 추가
        elif ext == '.vhdr':
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # vhdr 파일을 넘기면 같은 폴더의 .eeg와 .vmrk를 자동으로 불러옵니다.
                raw = mne.io.read_raw_brainvision(file_path, preload=True, verbose=False)
            return raw
        
        # [Case 1.8] GDF 파일 (.gdf) - 혜성 추가
        elif ext == '.gdf':
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                raw = mne.io.read_raw_gdf(file_path, preload=True, verbose=False)
            return raw
        
        # [Case 2] MAT 파일
        elif ext == '.mat':
            mat = sio.loadmat(file_path)
            key = CONFIG["MAT_KEY"]
            if key not in mat:
                # 키를 못 찾으면 가장 큰 변수를 자동으로 찾음 (편의 기능)
                valid_keys = [k for k in mat.keys() if not k.startswith('__')]
                largest_key = max(valid_keys, key=lambda k: mat[k].size if hasattr(mat[k], 'size') else 0)
                # print(f"[Info] Key '{key}' not found. Using '{largest_key}' instead.")
                data = mat[largest_key]
            else:
                data = mat[key]

        # [Case 3] TSV / CSV / TXT
        elif ext in ['.tsv', '.csv', '.txt']:
            header = 0 if CONFIG["HAS_HEADER"] else None
            df = pd.read_csv(file_path, sep=CONFIG["SEPARATOR"], header=header)
            data = df.values



        else:
            print(f"[Error] Unsupported extension: {ext}")
            return None

        # [공통 처리] Numpy Array -> MNE Raw 변환
        if data is None: return None

        # [안전한 Transpose 로직]
        # 설정된 채널 개수와 일치하는 차원을 찾습니다.
        target_n_ch = len(ch_names)
        shape = data.shape
        
        # 경우의 수 1: (N_CH, Time) -> 그대로
        if shape[0] == target_n_ch:
            pass
        # 경우의 수 2: (Time, N_CH) -> 전치
        elif shape[1] == target_n_ch:
            data = data.T
        # 경우의 수 3: 둘 다 안 맞지만, 일반적인 긴 축을 시간으로 간주
        elif shape[0] < shape[1]: # 행이 짧고 열이 길면 (채널, 시간)으로 간주
            pass
        else: # (시간, 채널)로 간주하고 전치
            data = data.T
        
        # 채널 개수 검증
        n_channels = data.shape[0]
        if n_channels != len(ch_names):
            # 설정된 이름보다 데이터 채널이 적으면 앞에서부터 자름
            if n_channels < len(ch_names):
                ch_names = ch_names[:n_channels]
            # 데이터 채널이 더 많으면 임시 이름 생성
            else:
                extra = [f"CH{i}" for i in range(len(ch_names), n_channels)]
                ch_names = ch_names + extra

        # Info 객체 생성
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
        raw = mne.io.RawArray(data, info, verbose=False)
        return raw

    except Exception as e:
        print(f"[Load Error] {file_path}: {e}")
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
            # 1. 샘플링 레이트가 너무 낮아 60Hz를 제대로 관측할 수 없는 경우 방어
            if fs <= 120: 
                return None 

            freqs, psd = signal.welch(eeg_data, fs, nperseg=int(fs), axis=-1)
            mean_psd = np.mean(psd, axis=0)
            
            # 2. 타겟 주파수(50Hz, 60Hz) 인덱스 및 파워 추출
            idx_50 = np.argmin(np.abs(freqs - 50))
            idx_60 = np.argmin(np.abs(freqs - 60))
            
            if idx_50 >= len(mean_psd) or idx_60 >= len(mean_psd): 
                return None

            power_50 = mean_psd[idx_50]
            power_60 = mean_psd[idx_60]

            # 3. 주변 주파수(Local Baseline) 대역 인덱스 설정 (±5Hz 구간)
            idx_45 = np.argmin(np.abs(freqs - 45))
            idx_55 = np.argmin(np.abs(freqs - 55))
            idx_65 = np.argmin(np.abs(freqs - 65))

            # np.median을 사용하면 정중앙의 노이즈 피크 값이 중간값에 영향을 주지 않음
            baseline_50 = np.median(mean_psd[idx_45:idx_55])
            baseline_60 = np.median(mean_psd[idx_55:idx_65])

            # 4. 주변 베이스라인 대비 몇 배나 솟아올라야 노이즈로 볼 것인지 (Threshold)
            # 선형 스케일(PSD) 기준 보통 4.0 ~ 5.0 배 이상 튀면 명확한 라인 노이즈로 간주
            threshold = 5.0 
            
            is_50_noise = power_50 > (baseline_50 * threshold)
            is_60_noise = power_60 > (baseline_60 * threshold)

            # 5. 결과 반환
            if is_60_noise and not is_50_noise:
                return 60.0
            elif is_50_noise and not is_60_noise:
                return 50.0
            elif is_50_noise and is_60_noise:
                # 만약 둘 다 노이즈 기준을 넘었다면, 주변 대비 더 심하게 튀어오른 쪽을 반환
                ratio_50 = power_50 / (baseline_50 + 1e-12) # 0 나누기 방지
                ratio_60 = power_60 / (baseline_60 + 1e-12)
                return 60.0 if ratio_60 > ratio_50 else 50.0
            return None

        except Exception:
            return None

    def apply(self, eeg_data, original_sr):
        nyq = 0.5 * original_sr
        low_cut = self.bandpass_freq[0]
        high_cut = self.bandpass_freq[1]

        if high_cut >= nyq:
            adjusted_high = nyq - 1.0 
            if adjusted_high <= low_cut: adjusted_high = nyq - 0.1
        else:
            adjusted_high = high_cut

        # Notch Filter
        line_freq = CONFIG["NOTCH_FREQ"] if CONFIG["NOTCH_FREQ"] is not None else self.detect_line_noise(eeg_data, original_sr)
        if line_freq and line_freq < nyq:
            b_notch, a_notch = signal.iirnotch(line_freq, Q=CONFIG["NOTCH_Q"], fs=original_sr)
            eeg_data = signal.filtfilt(b_notch, a_notch, eeg_data, axis=-1)

        # Bandpass Filter
        Wn_low = low_cut / nyq
        Wn_high = adjusted_high / nyq
        if Wn_high >= 1.0: Wn_high = 0.99 
        
        sos = signal.butter(3, [Wn_low, Wn_high], btype='band', analog=False, output='sos')
        eeg_data = signal.sosfiltfilt(sos, eeg_data, axis=-1)

        # Resample
        if original_sr != self.target_sr:
            gcd = math.gcd(int(original_sr), int(self.target_sr))
            up = int(self.target_sr // gcd)
            down = int(original_sr // gcd)
            eeg_data = signal.resample_poly(eeg_data, up, down, axis=-1)

        # Z-score Normalization
        mean = np.mean(eeg_data, axis=-1, keepdims=True)
        std = np.std(eeg_data, axis=-1, keepdims=True)
        eeg_data = (eeg_data - mean) / (std + 1e-8)

        # Type Casting & Clipping
        eeg_data = eeg_data.astype(np.float16)
        eeg_data = np.clip(eeg_data, -self.clip_limit, self.clip_limit)

        return eeg_data

# ==============================================================================
# 3. 채널 이름 정규화 및 좌표 추출 헬퍼
# ==============================================================================
def clean_channel_names(raw):
    mapping = {}
    for ch_name in raw.ch_names:
        # 1. 특수문자 및 불필요한 접미사 제거 (더 강력하게)
        # 예: "EEG Fp1-REF" -> "Fp1", "C3-M1" -> "C3"
        clean_name = re.sub(r'(?i)(EEG|[-_]?(REF|LE|MON|AVG|M1|M2|A1|A2)$)', '', ch_name).strip()
        clean_name = clean_name.upper()
        
        # 2. 표준 이름 매핑 (필요한 경우 추가)
        name_map = {
            'FP1': 'Fp1', 'FP2': 'Fp2', 'FZ': 'Fz', 'CZ': 'Cz', 'PZ': 'Pz', 'OZ': 'Oz',
            'T3': 'T7', 'T4': 'T8', 'T5': 'P7', 'T6': 'P8',
            'T7': 'T7', 'T8': 'T8' # 이미 표준인 경우 유지
        }
        final_name = name_map.get(clean_name, clean_name.capitalize()) # 나머지는 첫글자만 대문자로
        
        # 원본 이름과 다르면 매핑에 추가
        if ch_name != final_name:
            mapping[ch_name] = final_name

    # 중복 이름 방지 (예: T3->T7로 바꿨는데 이미 T7이 있는 경우)
    # MNE rename_channels는 중복을 허용하지 않으므로, 있는 이름은 제외하고 변경
    final_mapping = {k: v for k, v in mapping.items() if v not in raw.ch_names or k == v}
    
    try:
        raw.rename_channels(final_mapping)
    except Exception as e:
        pass # 이름 변경 실패 시 경고 없이 넘어감 (좌표 매핑에서 걸러짐)
    return raw

def biosemi64_mapping():
    biosemi64_mapping = {
        # Left hemisphere & Midline (A bundle)
        'A1': 'Fp1', 'A2': 'AF7', 'A3': 'AF3', 'A4': 'F1', 'A5': 'F3', 'A6': 'F5', 'A7': 'F7', 'A8': 'FT7',
        'A9': 'FC5', 'A10': 'FC3', 'A11': 'FC1', 'A12': 'C1', 'A13': 'C3', 'A14': 'C5', 'A15': 'T7', 'A16': 'TP7',
        'A17': 'CP5', 'A18': 'CP3', 'A19': 'CP1', 'A20': 'P1', 'A21': 'P3', 'A22': 'P5', 'A23': 'P7', 'A24': 'P9',
        'A25': 'PO7', 'A26': 'PO3', 'A27': 'O1', 'A28': 'Iz', 'A29': 'Oz', 'A30': 'POz', 'A31': 'Pz', 'A32': 'CPz',
        
        # Right hemisphere & Midline (B bundle)
        'B1': 'Fpz', 'B2': 'Fp2', 'B3': 'AF8', 'B4': 'AF4', 'B5': 'AFz', 'B6': 'Fz', 'B7': 'F2', 'B8': 'F4',
        'B9': 'F6', 'B10': 'F8', 'B11': 'FT8', 'B12': 'FC6', 'B13': 'FC4', 'B14': 'FC2', 'B15': 'FCz', 'B16': 'Cz',
        'B17': 'C2', 'B18': 'C4', 'B19': 'C6', 'B20': 'T8', 'B21': 'TP8', 'B22': 'CP6', 'B23': 'CP4', 'B24': 'CP2',
        'B25': 'P2', 'B26': 'P4', 'B27': 'P6', 'B28': 'P8', 'B29': 'P10', 'B30': 'PO8', 'B31': 'PO4', 'B32': 'O2'
    }
    return biosemi64_mapping

def get_valid_channel_indices(raw):
    valid_names = []
    valid_coords = []
    
    # 몽타주 정보에서 좌표 추출
    for ch_name in raw.ch_names:
        if ch_name not in raw.info['chs'][raw.ch_names.index(ch_name)]['ch_name']:
            continue
            
        # MNE 내부 좌표 가져오기
        ch_idx = raw.ch_names.index(ch_name)
        loc = raw.info['chs'][ch_idx]['loc'][:3]
        
        # 좌표가 유효한지 확인 (NaN이 아니고, 0,0,0이 아닌 경우)
        if not np.all(np.isnan(loc)) and not np.all(loc == 0):
            valid_names.append(ch_name)
            valid_coords.append(loc)
            
    return valid_names, valid_coords

# ==============================================================================
# 4. Worker 함수 (병렬 처리 단위)
# ==============================================================================
def process_single_file(file_path):
    try:
        # [NEW] 통합 로더 사용
        raw = load_data_to_mne(file_path)
        if raw is None: return None

        # 메모리 최적화: float32 변환
        if raw._data.dtype == np.float64:
             raw._data = raw._data.astype(np.float32)

        # 1. 채널 이름 정리 (EDF는 필수, 나머지도 포맷팅 위해 수행)
        raw = clean_channel_names(raw)

        try:
        # 3. Standard-1005 몽타주 적용 (좌표 매핑의 핵심)
            montage = mne.channels.make_standard_montage(CONFIG["montage"])
            if CONFIG["montage"] == 'biosemi64':
                raw.rename_channels(biosemi64_mapping())
            raw.set_montage(montage, match_case=False, on_missing='ignore')
        except: pass
        
        # 2. EEG 채널 필터링 (EDF인 경우 잡다한 채널 제거)
        if 'eeg' in raw:
            try:
                raw.pick("eeg", exclude="bads")
            except ValueError: pass


        if len(raw.ch_names) < 3: 
            print(f"[Skip] {file_path}: Not enough valid channels ({len(raw.ch_names)})")
            return None

        # 4. 좌표 유효성 검사 (좌표 없는 채널 제거)
        valid_names, valid_coords = get_valid_channel_indices(raw)
        
        if len(valid_names) < 3:
            print(f"[Skip] {file_path}: Not enough valid channels ({len(valid_names)})")
            return None

        # 5. Raw 객체에서 유효 채널만 남김
        raw.pick(valid_names)

        # 6. 신호 전처리 및 세그멘테이션
        data = raw.get_data() # (Channels, Time)
        sfreq = raw.info['sfreq']
        
        preprocessor = SmartEEGPreprocessor(
            target_sr=CONFIG["TARGET_SR"],
            bandpass_freq=CONFIG["BANDPASS"],
            clip_limit=CONFIG["CLIP_LIMIT"]
        )
        
        # processed_full = preprocessor.apply(data[:,int(sfreq*20):-int(sfreq*20)], sfreq) # 혹시나 앞에 노이즈 많을까봐 5초(200*5샘플) 자르고 시작하는 옵션 (필요시 활성화)
        processed_full = preprocessor.apply(data, sfreq)


        # 세그멘테이션
        samples_list = []
        window_samples = int(CONFIG["WINDOW_SECONDS"] * CONFIG["TARGET_SR"])
        total_length = processed_full.shape[-1]
        
        if total_length < window_samples: 
            print(f"[Skip] {file_path}: Not enough data for one segment ({total_length} samples at {CONFIG['TARGET_SR']} Hz)")
            return None
        
        # 파일명 추출 로직 (확장자 제거)
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        parent_folder = os.path.basename(os.path.dirname(file_path))
        
        num_segments = total_length // window_samples
        coords_array = np.array(valid_coords, dtype=np.float16)
        if processed_full.shape[-2] != len(coords_array):
            print(f"{file_path}: Shape mismatch after preprocessing, maybe channel error. Expected {len(coords_array)} channels, got {processed_full.shape[-2]}")
            return None
        
        for i in range(num_segments):
            start_idx = i * window_samples
            end_idx = start_idx + window_samples
            
            segment_data = processed_full[:, start_idx:end_idx]
            key = f"{parent_folder}_{file_name}_seg{i:04d}"
            
            sample_dict = {
                "key": key,
                "eeg": segment_data,
                "coords": coords_array,
                "meta": {
                    "source": file_name,
                    "segment_idx": i,
                    "start_sec": i * CONFIG["WINDOW_SECONDS"],
                    "n_channels": len(valid_names),
                    "ch_names": valid_names
                }
            }
            samples_list.append(sample_dict)
            
        return samples_list

    except Exception as e:
        print(f"[Error Processing] {file_path}: {e}")
        return None

# ==============================================================================
# 5. 메인 실행부
# ==============================================================================
if __name__ == "__main__":
    out_path = CONFIG["OUTPUT_PATTERN"]
    if out_path.startswith("file:"):
        out_path = out_path.replace("file:", "")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # [NEW] 설정된 확장자에 따라 파일 검색 (재귀)
    search_pattern = os.path.join(CONFIG["ROOT_DIR"], "**", CONFIG["FILE_EXT"])
    target_files = glob.glob(search_pattern, recursive=True)
    
    print(f"Workers: {CONFIG['NUM_WORKERS']}")
    print(f"Target Pattern: {CONFIG['FILE_EXT']}")
    print(f"Found Files: {len(target_files)}")
    
    writer = wds.ShardWriter(
        "file:" + out_path, 
        maxsize=CONFIG["SHARD_MAX_SIZE"], 
        maxcount=CONFIG["SHARD_MAX_COUNT"]
    )

    print(f"Starting processing... (Window: {CONFIG['WINDOW_SECONDS']}s)")
    
    total_segments = 0
    
    with Pool(CONFIG["NUM_WORKERS"]) as pool:
        for results in tqdm(pool.imap_unordered(process_single_file, target_files), total=len(target_files)):
            if results is None:
                continue
            
            for sample in results:
                writer.write({
                    "__key__": sample["key"],
                    "eeg.npy": sample["eeg"],
                    "coords.npy": sample["coords"],
                    "info.json": sample["meta"]
                })
                total_segments += 1

    writer.close()
    print(f"Done. Total segments saved: {total_segments}")
    total_hours = (total_segments * CONFIG["WINDOW_SECONDS"]) / 3600
    print(f"Total EEG hours in dataset: {total_hours:.2f} hours")