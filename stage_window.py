#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Sequence


def _windows_to_wsl(path_str: str) -> str:
    if os.name != 'nt' and len(path_str) >= 3 and path_str[1:3] in (':/', ':\\'):
        drive = path_str[0].lower()
        rest = path_str[3:].replace('\\', '/')
        return f'/mnt/{drive}/{rest}'
    return path_str


def _norm_path(path_str: str) -> str:
    return _windows_to_wsl(path_str)


def _load_manifest(path: str) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _build_relpath_lookup(manifest: Dict[str, Any]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for s in manifest.get('shards', []):
        p = str(s.get('shard_path', ''))
        rel = str(s.get('relative_path', ''))
        if p:
            out[p] = rel or os.path.basename(p)
    return out


def _resolve_window(manifest: Dict[str, Any], window_id: int) -> Dict[str, Any]:
    for w in manifest.get('windows', []):
        if int(w.get('window_id', -1)) == int(window_id):
            return w
    raise KeyError(f'window_id={window_id} not found in manifest')


def _copy_or_link(src: str, dst: str, method: str = 'copy2') -> None:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst):
        return
    if method == 'copy2':
        shutil.copy2(src, dst)
    elif method == 'hardlink':
        os.link(src, dst)
    elif method == 'symlink':
        os.symlink(src, dst)
    else:
        raise ValueError(f'unknown method: {method}')


def _read_nonempty_lines(path: str) -> List[str]:
    with open(path, 'r', encoding='utf-8') as f:
        return [ln.strip() for ln in f if ln.strip()]


def parse_args(argv: Sequence[str] | None = None):
    ap = argparse.ArgumentParser(description='Stage one manifest window to a destination directory and emit a shards_txt file.')
    ap.add_argument('--manifest', type=str, required=True, help='Path to open_eeg_stage_manifest.json')
    ap.add_argument('--window-id', type=int, required=True)
    ap.add_argument('--dest-root', type=str, required=True, help='Destination root to copy/link shards into')
    ap.add_argument('--include-resident', action='store_true', help='Also stage resident_shards')
    ap.add_argument('--method', choices=['copy2', 'hardlink', 'symlink'], default='copy2')
    ap.add_argument('--clean', action='store_true', help='Delete dest-root before staging')
    ap.add_argument('--write-shards-txt', type=str, default='', help='Optional output text file listing staged shard paths')
    ap.add_argument('--window-lists-dir', type=str, default='', help='Optional directory to also write resident/window txt files')
    return ap.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    manifest_path = Path(_norm_path(args.manifest)).expanduser().resolve()
    dest_root = Path(_norm_path(args.dest_root)).expanduser().resolve()
    manifest = _load_manifest(str(manifest_path))

    if manifest.get('schema') != 'open_eeg_staging_manifest':
        print('[warn] manifest schema is not open_eeg_staging_manifest; continuing anyway')

    if args.clean and dest_root.exists():
        shutil.rmtree(dest_root)
    dest_root.mkdir(parents=True, exist_ok=True)

    rel_lookup = _build_relpath_lookup(manifest)
    window = _resolve_window(manifest, int(args.window_id))

    selected: List[str] = []
    if args.include_resident:
        selected.extend(str(p) for p in manifest.get('resident_shards', []))
    selected.extend(str(p) for p in window.get('shards', []))

    # de-duplicate while preserving order
    seen = set()
    ordered: List[str] = []
    for p in selected:
        if p not in seen:
            seen.add(p)
            ordered.append(p)

    staged_paths: List[str] = []
    total_bytes = 0
    for src0 in ordered:
        src = _norm_path(src0)
        if not os.path.exists(src):
            raise FileNotFoundError(f'source shard not found: {src} (from manifest path {src0})')
        rel = rel_lookup.get(src0) or os.path.basename(src)
        dst = dest_root / rel
        _copy_or_link(src, str(dst), method=args.method)
        staged_paths.append(str(dst))
        try:
            total_bytes += int(os.path.getsize(dst))
        except OSError:
            pass

    if args.write_shards_txt:
        txt_path = Path(_norm_path(args.write_shards_txt)).expanduser().resolve()
        txt_path.parent.mkdir(parents=True, exist_ok=True)
        with open(txt_path, 'w', encoding='utf-8') as f:
            for p in staged_paths:
                f.write(p + '\n')

    if args.window_lists_dir:
        wl = Path(_norm_path(args.window_lists_dir)).expanduser().resolve()
        wl.mkdir(parents=True, exist_ok=True)
        if args.include_resident:
            with open(wl / 'resident_shards.txt', 'w', encoding='utf-8') as f:
                for p in manifest.get('resident_shards', []):
                    # write staged path if available, else raw manifest path
                    raw = str(p)
                    rel = rel_lookup.get(raw) or os.path.basename(raw)
                    f.write(str(dest_root / rel) + '\n')
        with open(wl / f'window_{int(args.window_id):03d}.txt', 'w', encoding='utf-8') as f:
            for raw in window.get('shards', []):
                raw = str(raw)
                rel = rel_lookup.get(raw) or os.path.basename(raw)
                f.write(str(dest_root / rel) + '\n')

    print(f'[done] staged {len(staged_paths)} shards to {dest_root}')
    print(f'[done] total_bytes={total_bytes}')
    if args.write_shards_txt:
        print(f'[done] wrote shards_txt: {Path(_norm_path(args.write_shards_txt)).expanduser().resolve()}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
