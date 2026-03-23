import math
import os
import re
import json
import warnings
import traceback
import numpy as np
import scipy.signal as signal
import mne
import webdataset as wds
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# MOABB 관련 임포트
import moabb
from moabb.datasets import Liu2024, Schirrmeister2017, BNCI2015_001

# 불필요한 로그 끄기
mne.set_log_level("WARNING")
moabb.set_log_level("info")

# ==============================================================================
# [설정 영역]
# ==============================================================================
CONFIG = {
    # [MOABB 설정] 사용할 데이터셋 클래스 지정
    # 예: BNCI2014001(), PhysionetMI(), Schirrmeister2017() 등
    "MOABB_DATASET": BNCI2015_001(), 
    
    # 출력 경로
    "OUTPUT_PATTERN": "D:/open_eeg_pp/moabb_bnci2015_001/eeg-%06d.tar",
    
    # 전처리 파라미터 (데이터셋 특성에 맞게 조절 필요)
    "TARGET_SR": 200,        # 목표 샘플링 레이트
    "BANDPASS": (0.5, 75.0), # (Low, High)
    "NOTCH_Q": 30.0,
    "CLIP_LIMIT": 15.0,


    # 세그멘테이션 설정
    "WINDOW_SECONDS": 10,     # BCI 데이터는 보통 짧으므로 4초로 예시 설정 (조절 가능)
    "DROP_LAST": True,

    # 저장 설정
    "SHARD_MAX_SIZE": 1024 ** 3 * 1, 
    "SHARD_MAX_COUNT": 100000,       
    
    # 병렬 처리 설정
    # "NUM_WORKERS": max(1, cpu_count() - 2) 
    "NUM_WORKERS": 4
}

# ==============================================================================
# 1. EEG 전처리 로직 (Signal Processing) - 기존과 동일
# ==============================================================================
class SmartEEGPreprocessor:
    def __init__(self, target_sr, bandpass_freq, clip_limit):
        self.target_sr = target_sr
        self.bandpass_freq = bandpass_freq
        self.clip_limit = clip_limit

    def detect_line_noise(self, eeg_data, fs):
        try:
            freqs, psd = signal.welch(eeg_data, fs, nperseg=int(fs), axis=-1)
            mean_psd = np.mean(psd, axis=0)
            idx_50 = np.argmin(np.abs(freqs - 50))
            idx_60 = np.argmin(np.abs(freqs - 60))
            if idx_50 >= len(mean_psd) or idx_60 >= len(mean_psd): return None
            power_50 = mean_psd[idx_50]
            power_60 = mean_psd[idx_60]
            if power_60 > power_50 * 1.5: return 60.0
            elif power_50 > power_60 * 1.5: return 50.0
            return None
        except: return None

    def apply(self, eeg_data, original_sr):
        nyq = 0.5 * original_sr
        low_cut = self.bandpass_freq[0]
        high_cut = self.bandpass_freq[1]

        if high_cut >= nyq:
            adjusted_high = nyq - 1.0 
            if adjusted_high <= low_cut: adjusted_high = nyq - 0.1
        else:
            adjusted_high = high_cut

        line_freq = self.detect_line_noise(eeg_data, original_sr)
        if line_freq and line_freq < nyq:
            b_notch, a_notch = signal.iirnotch(line_freq, Q=30.0, fs=original_sr)
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
# 2. 헬퍼 함수
# ==============================================================================
def clean_channel_names(raw):
    mapping = {}
    for ch_name in raw.ch_names:
        # MOABB 데이터셋마다 채널명이 다르므로 정규식 보정
        new_name = re.sub(r'(?i)(EEG|[-_]REF|[-_]LE|[-_]MON|[-_]AVG)', '', ch_name).strip()
        new_name = new_name.upper()
        # 필요한 경우 매핑 추가
        name_map = {
            'FP1': 'Fp1', 'FP2': 'Fp2', 'FZ': 'Fz', 'CZ': 'Cz', 'PZ': 'Pz', 'OZ': 'Oz',
            'T3': 'T7', 'T4': 'T8', 'T5': 'P7', 'T6': 'P8',
            'C3': 'C3', 'C4': 'C4' # 대소문자 이슈 방지
        }
        final_name = name_map.get(new_name, new_name.capitalize())
        mapping[ch_name] = final_name
    
    # 중복 채널명이 생길 경우를 대비해 rename_channels 사용 시 안전장치 필요
    # 여기서는 단순 매핑만 적용
    try:
        raw.rename_channels(mapping)
    except:
        pass # 이미 변경되었거나 매핑 실패시 무시
    return raw

def get_valid_channel_indices(raw):
    valid_names = []
    valid_coords = []
    
    # 몽타주가 없을 경우를 대비해 설정
    if raw.get_montage() is None:
        try:
            montage = mne.channels.make_standard_montage('standard_1005')
            raw.set_montage(montage, match_case=False, match_alias=True, on_missing='ignore')
        except:
            pass # 몽타주 설정 실패 시 좌표 없는 채널 제외 로직으로 넘어감

    for ch_name in raw.ch_names:
        idx = raw.ch_names.index(ch_name)
        # loc 정보가 있는지 확인
        if raw.info['chs'][idx]['loc'] is not None:
            loc = raw.info['chs'][idx]['loc'][:3]
            if not np.all(np.isnan(loc)) and not np.all(loc == 0):
                valid_names.append(ch_name)
                valid_coords.append(loc)
    return valid_names, valid_coords

# ==============================================================================
# 3. Worker 함수 (MOABB 전용)
# ==============================================================================
def process_subject_data(subject_id):
    """
    Subject ID를 받아서 해당 피험자의 모든 세션/런 데이터를 전처리하여 리스트로 반환
    """
    samples_list = []
    
    try:
        # [중요] Worker 프로세스 안에서 데이터셋 객체를 통해 데이터를 로드합니다.
        # 전역변수 CONFIG에서 데이터셋 클래스를 가져옵니다.
        dataset = CONFIG["MOABB_DATASET"]
        
        # MOABB 데이터 로드: {subject_id: {session: {run: Raw}}}
        # 이미 다운로드된 데이터는 로컬에서 빠르게 로드됩니다.
        data_dict = dataset.get_data(subjects=[subject_id])
        
        if subject_id not in data_dict:
            print(f"Subject {subject_id} data not found.")
            return None

        # 세션 및 런(Run) 순회
        for session_name, session_runs in data_dict[subject_id].items():
            for run_name, raw in session_runs.items():
                
                # 원본 보호를 위해 복사
                raw = raw.copy()
                # raw = raw.get_data().astype(np.float32)
                
                # 1. 채널 이름 정리
                raw = clean_channel_names(raw)
                
                # 2. EEG 채널 필터링 (EOG, Stim 등 제외)
                try:
                    if "eeg" in raw:
                        raw.pick("eeg", exclude="bads")
                    else:
                        # 타입 정보가 없는 경우, 채널명으로 추측하거나 모든 채널 사용
                        pass 
                except ValueError as e:
                    print(f"{subject_id} session {session_name} run {run_name}: ValueError during channel filtering: {e}")
                    continue

                # 3. 좌표 유효성 검사 및 채널 동기화
                valid_names, valid_coords = get_valid_channel_indices(raw)
                if len(valid_names) < 3: 
                    print(f"{subject_id} session {session_name} run {run_name}: Not enough valid channels.")
                    continue
                
                raw.pick(valid_names)

                # 4. 신호 전처리
                data = raw.get_data() # (Channels, Time)
                sfreq = raw.info['sfreq']
                
                preprocessor = SmartEEGPreprocessor(
                    target_sr=CONFIG["TARGET_SR"],
                    bandpass_freq=CONFIG["BANDPASS"],
                    clip_limit=CONFIG["CLIP_LIMIT"]
                )
                
                # 실제 전처리 수행
                processed_full = preprocessor.apply(data, sfreq)

                # 좌표 배열 (모든 세그먼트 공유)
                coords_array = np.array(valid_coords, dtype=np.float16)

                if processed_full.shape[-2] != len(coords_array):
                    print(f"{subject_id} session {session_name} run {run_name}: Shape mismatch after preprocessing, maybe channel error.")
                    continue
                
                # 5. Segmentation
                window_samples = int(CONFIG["WINDOW_SECONDS"] * CONFIG["TARGET_SR"])
                total_length = processed_full.shape[-1]
                
                if total_length < window_samples:
                    print(f"{subject_id} session {session_name} run {run_name} skipped: data too short.")
                    continue

                
                # 고유 키 생성: [데이터셋]_[피험자]_[세션]_[런]
                base_key = f"{dataset.code}_S{subject_id:03d}_{session_name}_{run_name}"
                
                num_segments = total_length // window_samples
                
                for i in range(num_segments):
                    start_idx = i * window_samples
                    end_idx = start_idx + window_samples
                    
                    segment_data = processed_full[:, start_idx:end_idx]
                    
                    # 최종 키: ..._seg0001
                    key = f"{base_key}_seg{i:04d}"
                    
                    sample_dict = {
                        "key": key,
                        "eeg": segment_data,
                        "coords": coords_array,
                        "meta": {
                            "dataset": dataset.code,
                            "subject": subject_id,
                            "session": session_name,
                            "run": run_name,
                            "segment_idx": i,
                            "ch_names": valid_names
                        }
                    }
                    samples_list.append(sample_dict)
                    
        return samples_list

    except Exception as e:
        # 에러 발생 시 해당 피험자 스킵 (로그 출력 가능)
        # traceback.print_exc()
        print(f"Error processing subject {subject_id}: {e}")
        return None

# ==============================================================================
# 4. 메인 실행부
# ==============================================================================
if __name__ == "__main__":
    out_path = CONFIG["OUTPUT_PATTERN"]
    if out_path.startswith("file:"):
        out_path = out_path.replace("file:", "")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    dataset = CONFIG["MOABB_DATASET"]
    print(f"Dataset: {dataset.code}")
    
    # 데이터셋 다운로드 (최초 1회 실행 시 시간 소요됨)
    print("Checking/Downloading dataset...")
    try:
        dataset.download()
    except Exception as e:
        print(f"Download Warning (check internet or path): {e}")
    
    # 피험자 리스트 가져오기
    subjects = dataset.subject_list
    print(f"Found {len(subjects)} subjects.")
    
    writer = wds.ShardWriter(
        "file:" + out_path, 
        maxsize=CONFIG["SHARD_MAX_SIZE"], 
        maxcount=CONFIG["SHARD_MAX_COUNT"]
    )

    print(f"Starting processing... (Window: {CONFIG['WINDOW_SECONDS']}s) w/ {CONFIG['NUM_WORKERS']} workers")
    total_segments = 0
    
    # 병렬 처리
    # MOABB은 Subject 단위로 병렬화하는 것이 가장 효율적입니다.
    valid_segments = 0
    with Pool(CONFIG["NUM_WORKERS"]) as pool:
        for results in tqdm(pool.imap_unordered(process_subject_data, subjects), total=len(subjects)):
            if results is None:
                # print("A subject processing returned None, skipping.")
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