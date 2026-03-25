#!/usr/bin/env python3
"""
distribute_shards.py

Manifest 기반으로 shard를 2개 HDD에 shape-balanced 분배하고,
ShapeBatcher에 최적화된 pre-shuffled shards.txt를 생성합니다.

Usage:
  # 1) manifest 생성 (기존 make_staging_manifest.py)
  python make_staging_manifest.py --data-root /mnt/hdd_combined/open_eeg \
      --out-json manifest.json --max-tokens 4096

  # 2) 분배 계획 생성 + 실행
  python distribute_shards.py \
      --manifest manifest.json \
      --hdd1 /mnt/hdd1/open_eeg \
      --hdd2 /mnt/hdd2/open_eeg \
      --out-shards-txt /home/user/shards.txt \
      --seed 42 \
      --dry-run          # 먼저 dry-run으로 계획 확인
      # --execute         # 실제 복사/이동 시

  # 3) 학습 시
  accelerate launch -m eeg_fm.train \
      --train_cfg train.json \
      --shards_txt /home/user/shards.txt
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def load_manifest(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_shard_shape_key(shard: Dict[str, Any]) -> str:
    """fit_shape_key를 우선 사용, 없으면 shape_key, 없으면 fallback."""
    return str(
        shard.get("fit_shape_key")
        or shard.get("shape_key")
        or f"C{shard.get('channels', 0)}_P{shard.get('fit_patches', 0)}"
    )


def plan_distribution(
    shards: List[Dict[str, Any]],
    hdd1_root: str,
    hdd2_root: str,
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Shape-balanced interleaving으로 shard를 2개 HDD에 분배.

    전략:
      1. shape_key로 그룹핑
      2. 각 그룹 내에서 shuffle
      3. 그룹 내 shard를 번갈아 HDD1/HDD2에 할당
      4. 최종 shard 순서: 모든 그룹에서 round-robin으로 뽑아서 interleave
         → 연속 읽기 시 다양한 shape가 섞여서 ShapeBatcher 버퍼링 최소화
         → 2개 HDD에서 교대로 읽어서 disk idle 최소화
    """
    rng = random.Random(seed)

    # 1) shape별 그룹핑
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for s in shards:
        key = get_shard_shape_key(s)
        groups[key].append(s)

    # 2) 각 그룹 내 shuffle + HDD 교대 할당
    for key in groups:
        rng.shuffle(groups[key])
        for i, s in enumerate(groups[key]):
            s["_assigned_hdd"] = 1 if (i % 2 == 0) else 2

    # 3) Round-robin interleave: 큰 그룹부터, 한 그룹에서 하나씩 번갈아 뽑기
    #    → 연속된 shard가 같은 shape가 되지 않도록
    group_queues = {}
    for key, ss in sorted(groups.items(), key=lambda kv: len(kv[1]), reverse=True):
        group_queues[key] = list(ss)

    interleaved: List[Dict[str, Any]] = []
    while group_queues:
        empty_keys = []
        for key in list(group_queues.keys()):
            q = group_queues[key]
            if q:
                interleaved.append(q.pop(0))
            if not q:
                empty_keys.append(key)
        for k in empty_keys:
            del group_queues[k]

    # 4) 최종 순서에서 HDD1/HDD2 교대가 잘 되는지 확인하고, 안 되면 local swap
    #    (이미 그룹 내에서 교대 할당했으므로 대체로 잘 됨)

    # 5) 경로 결정
    for s in interleaved:
        rel = s.get("relative_path", os.path.basename(s.get("shard_path", "")))
        if s["_assigned_hdd"] == 1:
            s["_dest_path"] = os.path.join(hdd1_root, rel)
        else:
            s["_dest_path"] = os.path.join(hdd2_root, rel)

    # 통계
    hdd1_count = sum(1 for s in interleaved if s["_assigned_hdd"] == 1)
    hdd2_count = sum(1 for s in interleaved if s["_assigned_hdd"] == 2)
    hdd1_bytes = sum(int(s.get("size_bytes", 0)) for s in interleaved if s["_assigned_hdd"] == 1)
    hdd2_bytes = sum(int(s.get("size_bytes", 0)) for s in interleaved if s["_assigned_hdd"] == 2)

    # shape 분포 확인 (각 HDD에 shape가 얼마나 고르게 분배됐는지)
    shape_dist = defaultdict(lambda: {"hdd1": 0, "hdd2": 0})
    for s in interleaved:
        key = get_shard_shape_key(s)
        if s["_assigned_hdd"] == 1:
            shape_dist[key]["hdd1"] += 1
        else:
            shape_dist[key]["hdd2"] += 1

    stats = {
        "total_shards": len(interleaved),
        "hdd1_shards": hdd1_count,
        "hdd2_shards": hdd2_count,
        "hdd1_gb": round(hdd1_bytes / (1024**3), 2),
        "hdd2_gb": round(hdd2_bytes / (1024**3), 2),
        "num_shape_groups": len(groups),
        "shape_balance": {
            k: v for k, v in sorted(shape_dist.items())
            if v["hdd1"] + v["hdd2"] >= 5  # 작은 그룹은 생략
        },
    }

    return interleaved, stats


def execute_distribution(
    plan: List[Dict[str, Any]],
    method: str = "copy",
    verbose: bool = True,
) -> None:
    """실제로 shard를 복사/이동."""
    total = len(plan)
    for i, s in enumerate(plan):
        src = s.get("scan_path") or s.get("shard_path", "")
        dst = s["_dest_path"]

        if os.path.exists(dst):
            if verbose and (i % 100 == 0):
                print(f"  [{i+1}/{total}] skip (exists): {dst}")
            continue

        os.makedirs(os.path.dirname(dst), exist_ok=True)

        if method == "copy":
            shutil.copy2(src, dst)
        elif method == "move":
            shutil.move(src, dst)
        elif method == "symlink":
            os.symlink(os.path.abspath(src), dst)
        elif method == "hardlink":
            os.link(src, dst)
        else:
            raise ValueError(f"Unknown method: {method}")

        if verbose and ((i + 1) % 100 == 0 or i + 1 == total):
            print(f"  [{i+1}/{total}] {method}: {dst}")


def write_shards_txt(
    plan: List[Dict[str, Any]],
    out_path: str,
) -> None:
    """Pre-shuffled 순서로 shards.txt 작성."""
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for s in plan:
            f.write(s["_dest_path"] + "\n")


def parse_args():
    p = argparse.ArgumentParser(
        description="Distribute shards across 2 HDDs with shape-balanced interleaving"
    )
    p.add_argument("--manifest", type=str, required=True,
                   help="Path to staging manifest JSON")
    p.add_argument("--hdd1", type=str, required=True,
                   help="Destination root for HDD 1")
    p.add_argument("--hdd2", type=str, required=True,
                   help="Destination root for HDD 2")
    p.add_argument("--out-shards-txt", type=str, required=True,
                   help="Output pre-shuffled shards.txt path")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--method", choices=["copy", "move", "symlink", "hardlink"],
                   default="copy",
                   help="How to place shards on destination HDDs")

    action = p.add_mutually_exclusive_group(required=True)
    action.add_argument("--dry-run", action="store_true",
                        help="계획만 출력, 실제 복사 안 함")
    action.add_argument("--execute", action="store_true",
                        help="실제 복사 실행")
    action.add_argument("--txt-only", action="store_true",
                        help="이미 분배된 상태에서 shards.txt만 재생성 "
                             "(shard_path를 dest_path로 간주)")

    return p.parse_args()


def main():
    args = parse_args()
    manifest = load_manifest(args.manifest)
    shards = manifest.get("shards", [])

    if not shards:
        print("[error] manifest에 shards가 없습니다.")
        return 1

    print(f"[info] manifest: {len(shards)} shards")

    plan, stats = plan_distribution(
        shards,
        hdd1_root=args.hdd1,
        hdd2_root=args.hdd2,
        seed=args.seed,
    )

    print(f"\n[plan] Distribution statistics:")
    print(f"  Total shards: {stats['total_shards']}")
    print(f"  HDD1: {stats['hdd1_shards']} shards ({stats['hdd1_gb']:.1f} GB)")
    print(f"  HDD2: {stats['hdd2_shards']} shards ({stats['hdd2_gb']:.1f} GB)")
    print(f"  Shape groups: {stats['num_shape_groups']}")
    print(f"\n  Shape balance (groups with >=5 shards):")
    for k, v in stats["shape_balance"].items():
        total = v["hdd1"] + v["hdd2"]
        pct1 = v["hdd1"] / total * 100 if total > 0 else 0
        print(f"    {k}: HDD1={v['hdd1']} ({pct1:.0f}%) HDD2={v['hdd2']} ({100-pct1:.0f}%)")

    if args.txt_only:
        # 이미 분배된 상태: shard_path or scan_path를 그대로 사용
        for s in plan:
            # 이미 dest에 있다고 가정
            if not os.path.exists(s["_dest_path"]):
                # fallback: 원본 경로 사용
                s["_dest_path"] = s.get("scan_path") or s.get("shard_path", "")

    if args.dry_run:
        print(f"\n[dry-run] shards.txt에 들어갈 처음 10개:")
        for s in plan[:10]:
            print(f"  {s['_dest_path']}")
        print(f"  ... ({len(plan) - 10} more)")

        # HDD 교대 패턴 확인
        first_20 = [(s["_assigned_hdd"], get_shard_shape_key(s)) for s in plan[:20]]
        print(f"\n[dry-run] 처음 20개의 HDD/shape 패턴:")
        for i, (hdd, shape) in enumerate(first_20):
            print(f"  [{i:2d}] HDD{hdd} {shape}")

        print(f"\n[dry-run] --execute로 실행하세요.")
        return 0

    if args.execute:
        print(f"\n[execute] {args.method}으로 분배 시작...")
        execute_distribution(plan, method=args.method, verbose=True)

    write_shards_txt(plan, args.out_shards_txt)
    print(f"\n[done] shards.txt: {args.out_shards_txt}")
    print(f"[done] {len(plan)} shards listed")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
