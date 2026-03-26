"""
TUH EEG Corpus 전처리 파이프라인
=====================================
- 대상: /media/hdd3/tuh/ 하위의 모든 .edf 파일
- 메모리 안전: 파일 전체를 preload하지 않고 block + context 단위로 읽어 처리
- 파이프라인:
    [50/60Hz notch → 0.5~70Hz bandpass → resample] (block+context) →
    [context 제거 → 60초 segmentation → z-score → clip]
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
import logging
import warnings
import argparse
import traceback
from pathlib import Path
from multiprocessing import Pool

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
    "TARGET_SR": 200,                  # 목표 샘플링 레이트 (Hz)
    "BANDPASS": (0.5, 70.0),           # (Low cut, High cut) Hz
    "NOTCH_FREQS": (50.0, 60.0),       # 전력선/환경 노이즈 주파수 (Hz)
    "NOTCH_Q": 30.0,                   # Notch filter Q factor
    "CLIP_LIMIT": 15.0,                # Z-score 후 클리핑 범위

    # 세그멘테이션 / block 처리
    "WINDOW_SECONDS": 60,              # 최종 저장 세그먼트 길이 (초)
    "BLOCK_SECONDS": 300,              # 처리 block(core) 길이 (초)
    "CONTEXT_SECONDS": 15.0,           # block 좌우 context 길이 (초)
    "INITIAL_DISCARD_SECONDS": 10.0,    # 파일 시작부 discard (초)
    "DROP_LAST": True,                 # 60초 미만 자투리 버림

    # 채널 최소 요구
    "MIN_CHANNELS": 3,

    # 저장
    "SHARD_MAX_SIZE": 1024 ** 3,       # 1 GB per shard
    "SHARD_MAX_COUNT": 100000,

    # 병렬 처리 — HDD 기반 I/O 병목 고려
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
        for edf_path in folder_num.rglob("*.edf"):
            edf_files.append(str(edf_path))

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
    OLD_TO_NEW = {
        "T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8",
    }

    CASE_MAP = {
        "FP1": "Fp1", "FP2": "Fp2", "FPZ": "Fpz",
        "FZ": "Fz", "CZ": "Cz", "PZ": "Pz", "OZ": "Oz",
        "FCZ": "FCz", "CPZ": "CPz", "POZ": "POz",
        "AFZ": "AFz", "FT7": "FT7", "FT8": "FT8",
        "TP7": "TP7", "TP8": "TP8",
    }

    mapping = {}
    for ch_name in raw.ch_names:
        clean = re.sub(r"^(?:EEG|EMG|ECG|EOG|PHOTIC|IBI|BURSTS|SUPPR)\s+", "", ch_name, flags=re.IGNORECASE)
        clean = re.sub(r"[-_]?(REF|LE|AR|AVG|M1|M2|A1|A2|MON)$", "", clean, flags=re.IGNORECASE).strip()
        upper = clean.upper()

        if upper in OLD_TO_NEW:
            upper = OLD_TO_NEW[upper]

        final = CASE_MAP.get(upper, clean.capitalize() if len(clean) <= 3 else clean)
        if upper not in CASE_MAP and upper not in OLD_TO_NEW:
            if any(c.isdigit() for c in clean):
                final = clean.upper()
            else:
                final = clean.capitalize()

        if ch_name != final:
            mapping[ch_name] = final

    existing = set(raw.ch_names)
    safe_mapping = {}
    used_names = set()
    for old, new in mapping.items():
        if new in existing and old != new and new not in mapping:
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
# 신호 처리 함수
# ==============================================================================
def crop_or_pad_last_dim(data, target_len, pad_mode="edge"):
    """마지막 축 길이를 target_len으로 맞춘다."""
    cur_len = data.shape[-1]
    if cur_len == target_len:
        return data
    if cur_len > target_len:
        return data[..., :target_len]

    diff = target_len - cur_len
    if cur_len == 0:
        return np.zeros((*data.shape[:-1], target_len), dtype=data.dtype)

    if pad_mode == "constant":
        return np.pad(data, [(0, 0)] * (data.ndim - 1) + [(0, diff)], mode="constant")

    return np.pad(data, [(0, 0)] * (data.ndim - 1) + [(0, diff)], mode=pad_mode)



def reflect_pad_2d(data, left_pad, right_pad):
    """(n_channels, n_samples) 배열의 시간축을 reflect padding 한다."""
    if left_pad <= 0 and right_pad <= 0:
        return data

    if data.shape[-1] < 2:
        mode = "edge"
    else:
        mode = "reflect"

    return np.pad(data, ((0, 0), (left_pad, right_pad)), mode=mode)



def load_block_with_context(raw, start_sec, core_sec, context_sec, sfreq, available_start_sec, total_seconds):
    """
    block(core) + context를 로드한다.
    - available_start_sec 이전 구간은 이미 discard된 것으로 간주한다.
    - 실제 데이터가 없는 부분만 reflect padding으로 채운다.
    """
    wanted_start = start_sec - context_sec
    wanted_end = start_sec + core_sec + context_sec

    load_start = max(available_start_sec, wanted_start)
    load_end = min(total_seconds, wanted_end)

    start_sample = int(round(load_start * sfreq))
    stop_sample = int(round(load_end * sfreq))
    data = raw.get_data(start=start_sample, stop=stop_sample).astype(np.float32)

    missing_left_sec = max(0.0, available_start_sec - wanted_start)
    missing_right_sec = max(0.0, wanted_end - total_seconds)

    missing_left = int(round(missing_left_sec * sfreq))
    missing_right = int(round(missing_right_sec * sfreq))

    if missing_left > 0 or missing_right > 0:
        data = reflect_pad_2d(data, missing_left, missing_right)

    expected_len = int(round((core_sec + 2.0 * context_sec) * sfreq))
    data = crop_or_pad_last_dim(data, expected_len, pad_mode="edge")
    return data



def apply_filters(data, sfreq, target_sr, bandpass, notch_freqs, notch_q):
    """
    Multi-notch → Bandpass → Resample 을 numpy 배열에 적용.
    data shape: (n_channels, n_samples)
    직접 context를 붙여서 들어온 배열을 처리하므로 filtfilt 내부 padding은 끈다.
    """
    sfreq_int = int(round(sfreq))
    target_sr_int = int(round(target_sr))
    nyq = 0.5 * sfreq
    low_cut, high_cut = bandpass

    if high_cut >= nyq:
        high_cut = nyq - 1.0
        if high_cut <= low_cut:
            high_cut = nyq - 0.1

    for f0 in notch_freqs:
        if f0 is None:
            continue
        if f0 <= 0 or f0 >= nyq:
            continue
        b_notch, a_notch = signal.iirnotch(f0, Q=notch_q, fs=sfreq)
        data = signal.filtfilt(b_notch, a_notch, data, axis=-1, padtype=None)

    wn_low = low_cut / nyq
    wn_high = min(high_cut / nyq, 0.99)
    sos = signal.butter(3, [wn_low, wn_high], btype="band", analog=False, output="sos")
    data = signal.sosfiltfilt(sos, data, axis=-1, padtype=None)

    if sfreq_int != target_sr_int:
        gcd = math.gcd(sfreq_int, target_sr_int)
        up = target_sr_int // gcd
        down = sfreq_int // gcd
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
    EDF 파일 하나를 block + context 단위로 읽어 전처리된 세그먼트 리스트 반환.

    파이프라인:
        1) preload=False로 메타데이터만 로드
        2) 채널 이름 정리 → 몽타주 적용 → 유효 채널 선별
        3) 파일 시작부 discard
        4) 300초 core + 좌우 context 단위로 로드
        5) block별: 50/60 notch → 0.5~70 bandpass → resample
        6) context 제거 후 core 내부를 60초 세그먼트로 분할
        7) 세그먼트별: z-score → clip
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = mne.io.read_raw_edf(file_path, preload=False, verbose=False)

        sfreq = raw.info["sfreq"]
        total_samples = raw.n_times
        total_seconds = total_samples / sfreq

        initial_discard_sec = CONFIG["INITIAL_DISCARD_SECONDS"]
        usable_seconds = total_seconds - initial_discard_sec
        if usable_seconds < CONFIG["WINDOW_SECONDS"]:
            return {"file": file_path, "status": "skip_short", "segments": []}

        raw = clean_channel_names_tuh(raw)

        try:
            montage = mne.channels.make_standard_montage(CONFIG["MONTAGE"])
            raw.set_montage(montage, match_case=False, on_missing="ignore")
        except Exception:
            pass

        eeg_ch_names = [ch for ch in raw.ch_names if raw.get_channel_types([ch])[0] == "eeg"]
        if len(eeg_ch_names) >= CONFIG["MIN_CHANNELS"]:
            raw.pick(eeg_ch_names)

        valid_names, valid_coords = get_valid_eeg_channels(raw)
        if len(valid_names) < CONFIG["MIN_CHANNELS"]:
            return {"file": file_path, "status": "skip_few_channels", "segments": []}

        raw.pick(valid_names)
        n_channels = len(valid_names)

        window_sec = float(CONFIG["WINDOW_SECONDS"])
        block_sec = float(CONFIG["BLOCK_SECONDS"])
        context_sec = float(CONFIG["CONTEXT_SECONDS"])
        target_sr = int(round(CONFIG["TARGET_SR"]))

        window_samples_out = int(round(window_sec * target_sr))
        context_samples_out = int(round(context_sec * target_sr))

        segments = []
        seg_global_idx = 0

        rel_path = os.path.relpath(file_path, CONFIG["ROOT_DIR"])
        file_id = rel_path.replace(os.sep, "_").replace(".edf", "")

        t = initial_discard_sec
        while t + window_sec <= total_seconds:
            remaining_sec = total_seconds - t
            core_sec = min(block_sec, remaining_sec)
            if core_sec < window_sec:
                break

            try:
                data = load_block_with_context(
                    raw=raw,
                    start_sec=t,
                    core_sec=core_sec,
                    context_sec=context_sec,
                    sfreq=sfreq,
                    available_start_sec=initial_discard_sec,
                    total_seconds=total_seconds,
                )
            except Exception as e:
                logging.warning(f"[Block Load Error] {file_path} t={t:.1f}s: {e}")
                t += block_sec
                continue

            try:
                data = apply_filters(
                    data,
                    sfreq,
                    target_sr,
                    CONFIG["BANDPASS"],
                    CONFIG["NOTCH_FREQS"],
                    CONFIG["NOTCH_Q"],
                )
            except Exception as e:
                logging.warning(f"[Filter Error] {file_path} t={t:.1f}s: {e}")
                t += block_sec
                continue

            core_samples_out = int(round(core_sec * target_sr))
            expected_total_out = core_samples_out + 2 * context_samples_out
            data = crop_or_pad_last_dim(data, expected_total_out, pad_mode="edge")
            data_core = data[:, context_samples_out: context_samples_out + core_samples_out]
            data_core = crop_or_pad_last_dim(data_core, core_samples_out, pad_mode="edge")

            n_full_segments = data_core.shape[-1] // window_samples_out
            if n_full_segments == 0:
                t += block_sec
                continue

            for local_idx in range(n_full_segments):
                s0 = local_idx * window_samples_out
                s1 = s0 + window_samples_out
                segment = data_core[:, s0:s1]
                if segment.shape[-1] != window_samples_out:
                    if CONFIG["DROP_LAST"]:
                        break
                    segment = crop_or_pad_last_dim(segment, window_samples_out, pad_mode="edge")

                segment = apply_segment_norm(segment, CONFIG["CLIP_LIMIT"])
                seg_start_sec = t + local_idx * window_sec

                key = f"{file_id}_seg{seg_global_idx:04d}"
                segments.append({
                    "key": key,
                    "eeg": segment,
                    "coords": valid_coords.astype(np.float16),
                    "meta": {
                        "source": rel_path,
                        "segment_idx": seg_global_idx,
                        "start_sec": seg_start_sec,
                        "n_channels": n_channels,
                        "ch_names": valid_names,
                        "original_sfreq": sfreq,
                    },
                })
                seg_global_idx += 1

            t += block_sec

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

    out_dir = os.path.dirname(CONFIG["OUTPUT_PATTERN"])
    os.makedirs(out_dir, exist_ok=True)

    tracker = ProgressTracker(CONFIG["PROGRESS_FILE"])
    if args.reset:
        tracker.reset()

    logger.info(f"EDF 파일 탐색 중: {CONFIG['ROOT_DIR']}")
    all_files = discover_edf_files(CONFIG["ROOT_DIR"])
    logger.info(f"전체 EDF 파일: {len(all_files)}개")

    pending_files = [f for f in all_files if not tracker.is_done(f)]
    skipped = len(all_files) - len(pending_files)
    if skipped > 0:
        logger.info(f"이전 완료분 건너뜀: {skipped}개 → 남은 파일: {len(pending_files)}개")

    if not pending_files:
        logger.info("처리할 파일이 없습니다. 모두 완료!")
        return

    existing_shards = glob.glob(os.path.join(out_dir, "tuh-*.tar"))
    start_shard = len(existing_shards) if existing_shards else 0
    out_pattern = CONFIG["OUTPUT_PATTERN"]

    writer = wds.ShardWriter(
        "file:" + out_pattern,
        maxsize=CONFIG["SHARD_MAX_SIZE"],
        maxcount=CONFIG["SHARD_MAX_COUNT"],
        start_shard=start_shard,
    )

    logger.info("=" * 70)
    logger.info("TUH EEG 전처리 시작")
    logger.info(f"  Workers:          {CONFIG['NUM_WORKERS']}")
    logger.info(f"  Window:           {CONFIG['WINDOW_SECONDS']}초")
    logger.info(f"  Block(core):      {CONFIG['BLOCK_SECONDS']}초")
    logger.info(f"  Context:          {CONFIG['CONTEXT_SECONDS']}초")
    logger.info(f"  Initial discard:  {CONFIG['INITIAL_DISCARD_SECONDS']}초")
    logger.info(f"  Target SR:        {CONFIG['TARGET_SR']} Hz")
    logger.info(f"  Bandpass:         {CONFIG['BANDPASS']} Hz")
    logger.info(f"  Notch freqs:      {CONFIG['NOTCH_FREQS']} Hz")
    logger.info(f"  Clip:             ±{CONFIG['CLIP_LIMIT']} (z-score)")
    logger.info(f"  Resume shard:     #{start_shard}")
    logger.info(f"  대기 파일:        {len(pending_files)}개")
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

            for sample in segments:
                writer.write({
                    "__key__": sample["key"],
                    "eeg.npy": sample["eeg"],
                    "coords.npy": sample["coords"],
                    "info.json": sample["meta"],
                })
                total_segments += 1

            tracker.mark_done(file_path, n_seg, status="ok")

            elapsed = time.time() - start_time
            rate = total_segments / max(elapsed, 1)
            pbar.set_postfix(
                segs=total_segments,
                err=total_errors,
                skip=total_skipped,
                rate=f"{rate:.1f}seg/s",
            )

    writer.close()

    elapsed = time.time() - start_time
    total_hours_eeg = (total_segments * CONFIG["WINDOW_SECONDS"]) / 3600

    logger.info("=" * 70)
    logger.info("전처리 완료!")
    logger.info(f"  처리 파일:        {len(pending_files) - total_errors - total_skipped}개")
    logger.info(f"  에러 파일:        {total_errors}개")
    logger.info(f"  스킵 파일:        {total_skipped}개")
    logger.info(f"  총 세그먼트:      {total_segments}개")
    logger.info(f"  EEG 총 시간:      {total_hours_eeg:.2f}시간")
    logger.info(f"  소요 시간:        {elapsed / 3600:.2f}시간")
    logger.info(f"  처리 속도:        {total_segments / max(elapsed, 1):.1f} seg/s")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
