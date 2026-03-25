"""
TUH EEG Corpus 전처리 파이프라인
=====================================
- 대상: /media/hdd3/tuh/ 하위의 모든 .edf 파일
- 메모리 안전: 파일 전체를 preload하지 않고 청크 단위로 읽어 처리
- 파이프라인: [notch 60Hz → bandpass → resample] (청크+패딩) → [z-score → clip] (세그먼트별)
- 이어하기: 완료된 파일 목록을 progress 파일에 기록, 재실행 시 건너뜀
- 하드웨어: HP Z8 G5 Fury / Xeon W5-3425 (12C/24T) / 128GB DDR5 / HDD 기준 최적화

사용법:
    python tuh_preprocess.py
    python tuh_preprocess.py --resume          # 중단 지점부터 이어서
    python tuh_preprocess.py --reset           # progress 초기화 후 처음부터
"""

import os
import sys
import re
import json
import glob
import math
import time
import fcntl
import hashlib
import logging
import warnings
import argparse
import traceback
from pathlib import Path
from multiprocessing import Pool, cpu_count

import numpy as np
import scipy.signal as signal
import mne
import webdataset as wds
from tqdm import tqdm


# ==============================================================================
# 설정
# ==============================================================================
CONFIG = {
    # 경로
    "ROOT_DIR": "/media/hdd3/tuh",
    "OUTPUT_PATTERN": "/media/hdd1/tuh_preprocessed/tuh-%06d.tar",
    "PROGRESS_FILE": "/media/hdd1/tuh_preprocessed/.progress.jsonl",   # 완료 기록
    "LOG_FILE": "/media/hdd1/tuh_preprocessed/preprocess.log",

    # 몽타주
    "MONTAGE": "standard_1005",

    # 전처리 파라미터
    "TARGET_SR": 200,             # 목표 샘플링 레이트 (Hz)
    "BANDPASS": (0.5, 75.0),      # (Low cut, High cut) Hz
    "NOTCH_FREQ": 60.0,           # 미국 전력선 주파수 (Hz)
    "NOTCH_Q": 30.0,              # Notch filter Q factor
    "CLIP_LIMIT": 15.0,           # Z-score 후 클리핑 범위

    # 세그멘테이션
    "WINDOW_SECONDS": 60,         # 세그먼트 길이 (초)
    "FILTER_PAD_SECONDS": 5.0,    # 필터 edge artifact 방지 패딩 (초)
    "DROP_LAST": True,            # 60초 미만 자투리 버림

    # 채널 최소 요구
    "MIN_CHANNELS": 3,

    # 저장
    "SHARD_MAX_SIZE": 1024 ** 3,  # 1 GB per shard
    "SHARD_MAX_COUNT": 100000,

    # 병렬 처리 — HDD 기반 I/O 병목 고려
    # Xeon W5-3425 = 12C/24T, 128GB RAM, HDD
    # HDD random I/O 병목 → 너무 많은 worker는 역효과
    # 각 worker는 chunk 단위 로딩이라 메모리 사용량 낮음 (~수십 MB/worker)
    "NUM_WORKERS": 8,
}


# ==============================================================================
# 로깅 설정
# ==============================================================================
def setup_logging():
    os.makedirs(os.path.dirname(CONFIG["LOG_FILE"]), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(CONFIG["LOG_FILE"], encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    # MNE 로그 억제
    mne.set_log_level("ERROR")


# ==============================================================================
# Progress 관리 (이어하기 지원)
# ==============================================================================
class ProgressTracker:
    """
    완료된 파일 경로를 JSONL 파일에 한 줄씩 기록.
    재실행 시 이미 처리된 파일을 건너뛴다.
    멀티프로세스 환경에서 file lock으로 안전하게 기록.
    """

    def __init__(self, progress_path):
        self.path = progress_path
        self.completed = set()
        self._load()

    def _load(self):
        if not os.path.exists(self.path):
            return
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    self.completed.add(entry["file"])
                except (json.JSONDecodeError, KeyError):
                    continue
        logging.info(f"Progress 로드: 이전에 완료된 파일 {len(self.completed)}개")

    def is_done(self, file_path):
        return file_path in self.completed

    def mark_done(self, file_path, n_segments, status="ok"):
        """파일 처리 완료를 기록 (file lock으로 프로세스 안전)"""
        entry = json.dumps({
            "file": file_path,
            "segments": n_segments,
            "status": status,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }, ensure_ascii=False)

        with open(self.path, "a", encoding="utf-8") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            f.write(entry + "\n")
            fcntl.flock(f, fcntl.LOCK_UN)

        self.completed.add(file_path)

    def reset(self):
        if os.path.exists(self.path):
            os.remove(self.path)
        self.completed.clear()
        logging.info("Progress 초기화 완료")


# ==============================================================================
# TUH 파일 탐색
# ==============================================================================
def discover_edf_files(root_dir):
    """
    /media/hdd3/tuh/000~150/subject/session/**/*.edf 구조 탐색
    """
    edf_files = []
    root = Path(root_dir)

    for folder_num in sorted(root.iterdir()):
        if not folder_num.is_dir():
            continue
        # 000 ~ 150 같은 숫자 폴더 또는 기타 폴더 모두 탐색
        for edf_path in folder_num.rglob("*.edf"):
            edf_files.append(str(edf_path))

    # glob 대비 정렬하여 재현성 확보
    edf_files.sort()
    return edf_files


# ==============================================================================
# 채널 이름 정규화 (TUH 전용)
# ==============================================================================
def clean_channel_names_tuh(raw):
    """
    TUH EDF 채널 이름 정리.
    예: "EEG FP1-REF" → "Fp1", "EEG T3-LE" → "T7"
    """
    # 구식 → 신식 10-20 매핑
    OLD_TO_NEW = {
        "T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8",
    }

    # 표준 대소문자 매핑 (MNE standard_1005 기준)
    CASE_MAP = {
        "FP1": "Fp1", "FP2": "Fp2", "FPZ": "Fpz",
        "FZ": "Fz", "CZ": "Cz", "PZ": "Pz", "OZ": "Oz",
        "FCZ": "FCz", "CPZ": "CPz", "POZ": "POz",
        "AFZ": "AFz", "FT7": "FT7", "FT8": "FT8",
        "TP7": "TP7", "TP8": "TP8",
    }

    mapping = {}
    for ch_name in raw.ch_names:
        # 1) "EEG ", "EMG ", "ECG ", "EOG " 접두어 제거
        clean = re.sub(r"^(?:EEG|EMG|ECG|EOG|PHOTIC|IBI|BURSTS|SUPPR)\s+", "", ch_name, flags=re.IGNORECASE)
        # 2) "-REF", "-LE", "-AR", "-AVG" 등 접미사 제거
        clean = re.sub(r"[-_]?(REF|LE|AR|AVG|M1|M2|A1|A2|MON)$", "", clean, flags=re.IGNORECASE).strip()
        # 3) 대문자화 후 매핑
        upper = clean.upper()
        # 구식 이름 변환
        if upper in OLD_TO_NEW:
            upper = OLD_TO_NEW[upper]
        # 대소문자 정규화
        final = CASE_MAP.get(upper, clean.capitalize() if len(clean) <= 3 else clean)
        # 대소문자 매핑에 없으면 upper를 그대로 capitalize
        if upper not in CASE_MAP and upper not in OLD_TO_NEW:
            # 예: "F3" → "F3", "AF7" → "AF7"
            # 숫자가 포함된 경우 첫 글자만 대문자
            if any(c.isdigit() for c in clean):
                final = clean.upper()
                # F3, C4 등은 그대로, AF7 등도 그대로
                # Fp1, Fp2만 특수 처리 (이미 CASE_MAP에 있음)
            else:
                final = clean.capitalize()

        if ch_name != final:
            mapping[ch_name] = final

    # 중복 이름 방지
    existing = set(raw.ch_names)
    safe_mapping = {}
    used_names = set()
    for old, new in mapping.items():
        if new in existing and old != new and new not in mapping:
            # 이미 동일한 이름이 존재 → 건너뜀
            continue
        if new in used_names:
            continue
        safe_mapping[old] = new
        used_names.add(new)

    if safe_mapping:
        try:
            raw.rename_channels(safe_mapping)
        except Exception:
            pass

    return raw


# ==============================================================================
# 유효 채널 추출 (몽타주 좌표 기반)
# ==============================================================================
def get_valid_eeg_channels(raw):
    """
    몽타주 적용 후 3D 좌표가 유효한 EEG 채널만 추출.
    반환: (valid_names, valid_coords)
    """
    valid_names = []
    valid_coords = []

    for idx, ch_name in enumerate(raw.ch_names):
        loc = raw.info["chs"][idx]["loc"][:3]
        if not np.all(np.isnan(loc)) and not np.all(loc == 0):
            valid_names.append(ch_name)
            valid_coords.append(loc)

    return valid_names, np.array(valid_coords, dtype=np.float32)


# ==============================================================================
# 신호 처리 함수 (청크 단위)
# ==============================================================================
def apply_filters(data, sfreq, target_sr, bandpass, notch_freq, notch_q):
    """
    Notch → Bandpass → Resample 을 numpy 배열에 적용.
    data shape: (n_channels, n_samples)
    반환: (filtered_resampled_data, target_sr)
    """
    nyq = 0.5 * sfreq
    low_cut, high_cut = bandpass

    # Nyquist 보정
    if high_cut >= nyq:
        high_cut = nyq - 1.0
        if high_cut <= low_cut:
            high_cut = nyq - 0.1

    # 1) Notch Filter (60 Hz)
    if notch_freq and notch_freq < nyq:
        b_notch, a_notch = signal.iirnotch(notch_freq, Q=notch_q, fs=sfreq)
        data = signal.filtfilt(b_notch, a_notch, data, axis=-1)
        # 120 Hz 하모닉도 제거 (가능한 경우)
        if 2 * notch_freq < nyq:
            b2, a2 = signal.iirnotch(2 * notch_freq, Q=notch_q, fs=sfreq)
            data = signal.filtfilt(b2, a2, data, axis=-1)

    # 2) Bandpass Filter
    Wn_low = low_cut / nyq
    Wn_high = high_cut / nyq
    if Wn_high >= 1.0:
        Wn_high = 0.99
    sos = signal.butter(3, [Wn_low, Wn_high], btype="band", analog=False, output="sos")
    data = signal.sosfiltfilt(sos, data, axis=-1)

    # 3) Resample
    if int(sfreq) != int(target_sr):
        gcd = math.gcd(int(sfreq), int(target_sr))
        up = int(target_sr) // gcd
        down = int(sfreq) // gcd
        data = signal.resample_poly(data, up, down, axis=-1)

    return data.astype(np.float32)


def apply_segment_norm(segment, clip_limit):
    """
    세그먼트별 Z-score 정규화 + 클리핑.
    segment shape: (n_channels, n_samples)
    """
    mean = np.mean(segment, axis=-1, keepdims=True)
    std = np.std(segment, axis=-1, keepdims=True)
    segment = (segment - mean) / (std + 1e-8)
    segment = np.clip(segment, -clip_limit, clip_limit)
    return segment.astype(np.float16)


# ==============================================================================
# 단일 파일 처리 (Worker 함수)
# ==============================================================================
def process_single_file(file_path):
    """
    EDF 파일 하나를 청크 단위로 읽어 전처리된 세그먼트 리스트 반환.

    파이프라인:
        1) preload=False로 메타데이터만 로드
        2) 채널 이름 정리 → 몽타주 적용 → 유효 채널 선별
        3) 60초 + 패딩 청크 단위로 데이터 로드
        4) 청크별: notch → bandpass → resample
        5) 패딩 제거 후 세그먼트별: z-score → clip
    """
    try:
        # ── 1. 메타데이터 로드 (데이터는 아직 안 읽음) ──
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = mne.io.read_raw_edf(file_path, preload=False, verbose=False)

        sfreq = raw.info["sfreq"]
        total_samples = raw.n_times
        total_seconds = total_samples / sfreq

        # 최소 60초 이상이어야 세그먼트 1개라도 생성 가능
        if total_seconds < CONFIG["WINDOW_SECONDS"]:
            return {"file": file_path, "status": "skip_short", "segments": []}

        # ── 2. 채널 준비 (메모리 부담 없음) ──
        raw = clean_channel_names_tuh(raw)

        # 채널 타입이 'eeg'로 안 잡힌 채널도 있을 수 있으므로, 
        # 일단 몽타주 매칭으로 EEG 채널을 판별
        try:
            montage = mne.channels.make_standard_montage(CONFIG["MONTAGE"])
            raw.set_montage(montage, match_case=False, on_missing="ignore")
        except Exception:
            pass

        # EEG 타입 채널만 우선 시도
        eeg_ch_names = [ch for ch in raw.ch_names
                        if raw.get_channel_types([ch])[0] == "eeg"]
        if len(eeg_ch_names) >= CONFIG["MIN_CHANNELS"]:
            raw.pick(eeg_ch_names)
        # EEG 타입이 부족하면 좌표 기반으로 유효 채널 판별

        valid_names, valid_coords = get_valid_eeg_channels(raw)
        if len(valid_names) < CONFIG["MIN_CHANNELS"]:
            return {"file": file_path, "status": "skip_few_channels", "segments": []}

        raw.pick(valid_names)
        n_channels = len(valid_names)

        # ── 3. 청크 단위 처리 ──
        window_sec = CONFIG["WINDOW_SECONDS"]
        pad_sec = CONFIG["FILTER_PAD_SECONDS"]
        target_sr = CONFIG["TARGET_SR"]
        window_samples_out = int(window_sec * target_sr)  # 리샘플 후 기대 샘플 수

        segments = []
        seg_global_idx = 0

        # 파일 식별자 생성 (경로 기반)
        rel_path = os.path.relpath(file_path, CONFIG["ROOT_DIR"])
        file_id = rel_path.replace(os.sep, "_").replace(".edf", "")

        t = 0.0
        while t + window_sec <= total_seconds:
            # 패딩 포함 범위 계산
            load_start = max(0.0, t - pad_sec)
            load_end = min(total_seconds, t + window_sec + pad_sec)

            # 실제 패딩 크기 (파일 경계에서 잘릴 수 있음)
            actual_pad_left = t - load_start
            actual_pad_right = load_end - (t + window_sec)

            # 샘플 인덱스로 변환
            start_sample = int(load_start * sfreq)
            stop_sample = int(load_end * sfreq)
            # 범위 보정
            stop_sample = min(stop_sample, total_samples)

            # 데이터 로드 (이 시점에서만 디스크 I/O 발생)
            try:
                data = raw.get_data(start=start_sample, stop=stop_sample)
            except Exception as e:
                logging.warning(f"[Chunk Load Error] {file_path} t={t:.1f}s: {e}")
                t += window_sec
                continue

            # float32 변환 (메모리 절약)
            data = data.astype(np.float32)

            # ── 4. 필터링 + 리샘플 (패딩 포함 청크에 적용) ──
            data = apply_filters(
                data, sfreq, target_sr,
                CONFIG["BANDPASS"], CONFIG["NOTCH_FREQ"], CONFIG["NOTCH_Q"]
            )

            # ── 5. 패딩 제거 (리샘플 후 샘플 수 기준) ──
            pad_left_samples = int(actual_pad_left * target_sr)
            pad_right_samples = int(actual_pad_right * target_sr)

            if pad_right_samples > 0:
                data = data[:, pad_left_samples: -pad_right_samples]
            else:
                data = data[:, pad_left_samples:]

            # 길이 보정 (리샘플 반올림 오차)
            if data.shape[-1] > window_samples_out:
                data = data[:, :window_samples_out]
            elif data.shape[-1] < window_samples_out:
                # 약간 짧으면 zero-pad (1~2 샘플 차이만 허용)
                diff = window_samples_out - data.shape[-1]
                if diff <= 3:
                    data = np.pad(data, ((0, 0), (0, diff)), mode="constant")
                else:
                    # 너무 많이 부족하면 건너뜀
                    t += window_sec
                    continue

            # ── 6. 세그먼트별 정규화 ──
            segment = apply_segment_norm(data, CONFIG["CLIP_LIMIT"])

            key = f"{file_id}_seg{seg_global_idx:04d}"
            segments.append({
                "key": key,
                "eeg": segment,                              # (n_ch, 12000) float16
                "coords": valid_coords.astype(np.float16),   # (n_ch, 3)    float16
                "meta": {
                    "source": rel_path,
                    "segment_idx": seg_global_idx,
                    "start_sec": t,
                    "n_channels": n_channels,
                    "ch_names": valid_names,
                    "original_sfreq": sfreq,
                },
            })

            seg_global_idx += 1
            t += window_sec

        return {"file": file_path, "status": "ok", "segments": segments}

    except Exception as e:
        logging.error(f"[File Error] {file_path}: {e}\n{traceback.format_exc()}")
        return {"file": file_path, "status": f"error: {e}", "segments": []}


# ==============================================================================
# 메인
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="TUH EEG 전처리 파이프라인")
    parser.add_argument("--resume", action="store_true", default=True,
                        help="이전 진행 상황에서 이어서 처리 (기본값: True)")
    parser.add_argument("--reset", action="store_true",
                        help="진행 상황 초기화 후 처음부터 시작")
    parser.add_argument("--workers", type=int, default=CONFIG["NUM_WORKERS"],
                        help=f"병렬 워커 수 (기본값: {CONFIG['NUM_WORKERS']})")
    parser.add_argument("--root", type=str, default=CONFIG["ROOT_DIR"],
                        help="TUH 데이터 루트 디렉토리")
    args = parser.parse_args()

    CONFIG["NUM_WORKERS"] = args.workers
    CONFIG["ROOT_DIR"] = args.root

    setup_logging()
    logger = logging.getLogger(__name__)

    # 출력 디렉토리 생성
    out_dir = os.path.dirname(CONFIG["OUTPUT_PATTERN"])
    os.makedirs(out_dir, exist_ok=True)

    # Progress 관리
    tracker = ProgressTracker(CONFIG["PROGRESS_FILE"])
    if args.reset:
        tracker.reset()

    # 파일 탐색
    logger.info(f"EDF 파일 탐색 중: {CONFIG['ROOT_DIR']}")
    all_files = discover_edf_files(CONFIG["ROOT_DIR"])
    logger.info(f"전체 EDF 파일: {len(all_files)}개")

    # 이미 완료된 파일 제외
    pending_files = [f for f in all_files if not tracker.is_done(f)]
    skipped = len(all_files) - len(pending_files)
    if skipped > 0:
        logger.info(f"이전 완료분 건너뜀: {skipped}개 → 남은 파일: {len(pending_files)}개")

    if not pending_files:
        logger.info("처리할 파일이 없습니다. 모두 완료!")
        return

    # ── ShardWriter 설정 ──
    # 기존 shard가 있으면 이어서 쓰기 위해 다음 번호 계산
    existing_shards = glob.glob(os.path.join(out_dir, "tuh-*.tar"))
    start_shard = len(existing_shards) if existing_shards else 0

    # start_shard_count를 반영하기 위해 패턴 조정
    # webdataset은 자동으로 번호를 매기므로, 기존 파일과 겹치지 않게
    # 이어쓰기 시에는 기존 shard 뒤에 이어서 번호가 매겨짐
    out_pattern = CONFIG["OUTPUT_PATTERN"]

    writer = wds.ShardWriter(
        "file:" + out_pattern,
        maxsize=CONFIG["SHARD_MAX_SIZE"],
        maxcount=CONFIG["SHARD_MAX_COUNT"],
        start_shard=start_shard,
    )

    logger.info("=" * 70)
    logger.info(f"TUH EEG 전처리 시작")
    logger.info(f"  Workers:        {CONFIG['NUM_WORKERS']}")
    logger.info(f"  Window:         {CONFIG['WINDOW_SECONDS']}초")
    logger.info(f"  Target SR:      {CONFIG['TARGET_SR']} Hz")
    logger.info(f"  Bandpass:       {CONFIG['BANDPASS']} Hz")
    logger.info(f"  Notch:          {CONFIG['NOTCH_FREQ']} Hz")
    logger.info(f"  Clip:           ±{CONFIG['CLIP_LIMIT']} (z-score)")
    logger.info(f"  Resume shard:   #{start_shard}")
    logger.info(f"  대기 파일:      {len(pending_files)}개")
    logger.info("=" * 70)

    total_segments = 0
    total_errors = 0
    total_skipped = 0
    start_time = time.time()

    with Pool(CONFIG["NUM_WORKERS"]) as pool:
        pbar = tqdm(
            pool.imap_unordered(process_single_file, pending_files),
            total=len(pending_files),
            desc="Processing",
            unit="file",
            smoothing=0.1,
        )

        for result in pbar:
            file_path = result["file"]
            status = result["status"]
            segments = result["segments"]
            n_seg = len(segments)

            if status.startswith("error"):
                total_errors += 1
                tracker.mark_done(file_path, 0, status=status)
                continue

            if n_seg == 0:
                total_skipped += 1
                tracker.mark_done(file_path, 0, status=status)
                continue

            # WebDataset에 기록
            for sample in segments:
                writer.write({
                    "__key__": sample["key"],
                    "eeg.npy": sample["eeg"],
                    "coords.npy": sample["coords"],
                    "info.json": sample["meta"],
                })
                total_segments += 1

            tracker.mark_done(file_path, n_seg, status="ok")

            # 진행 상황 표시
            elapsed = time.time() - start_time
            rate = total_segments / max(elapsed, 1)
            pbar.set_postfix(
                segs=total_segments,
                err=total_errors,
                skip=total_skipped,
                rate=f"{rate:.1f}seg/s",
            )

    writer.close()

    # ── 최종 통계 ──
    elapsed = time.time() - start_time
    total_hours_eeg = (total_segments * CONFIG["WINDOW_SECONDS"]) / 3600

    logger.info("=" * 70)
    logger.info("전처리 완료!")
    logger.info(f"  처리 파일:      {len(pending_files) - total_errors - total_skipped}개")
    logger.info(f"  에러 파일:      {total_errors}개")
    logger.info(f"  스킵 파일:      {total_skipped}개")
    logger.info(f"  총 세그먼트:    {total_segments}개")
    logger.info(f"  EEG 총 시간:    {total_hours_eeg:.2f}시간")
    logger.info(f"  소요 시간:      {elapsed / 3600:.2f}시간")
    logger.info(f"  처리 속도:      {total_segments / max(elapsed, 1):.1f} seg/s")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
