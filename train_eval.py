from __future__ import annotations

import argparse
import gc
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch

from . import eval as eval_mod
from . import train as train_mod


def _is_global_main_process() -> bool:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        try:
            return torch.distributed.get_rank() == 0
        except Exception:
            pass
    return int(os.environ.get("RANK", "0")) == 0


def _make_eval_wandb_name(train_run_name: Optional[str]) -> str:
    train_run_name = (train_run_name or "").strip()
    return f"eval_{train_run_name}" if train_run_name else ""


def build_parser(add_help: bool = True) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        parents=[train_mod.build_parser(add_help=False)],
        add_help=add_help,
        conflict_handler="resolve",
        description="Train once, then run a single linear-probe eval on the final teacher checkpoint.",
    )

    parser.add_argument("--tasks", type=str, nargs="+", default=["tuab", "isruc", "mi"])
    parser.add_argument("--task_root", type=str, default="", help="Base directory containing TUAB_npy/, ISRUC_npy/, and PhysioNetMI_npy/.")
    parser.add_argument("--tuab_root", type=str, default="", help="Override TUAB cache root. If omitted, <task_root>/TUAB_npy is used.")
    parser.add_argument("--isruc_root", type=str, default="", help="Override ISRUC cache root. If omitted, <task_root>/ISRUC_npy is used.")
    parser.add_argument("--mi_root", type=str, default="", help="Override PhysioNetMI cache root. If omitted, <task_root>/PhysioNetMI_npy is used.")

    parser.add_argument("--tuab_val_ratio", type=float, default=0.2)
    parser.add_argument("--feat_batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=2, help="Eval feature-extraction workers.")
    parser.add_argument("--pin_memory", action="store_true")
    eval_mod.add_bool_arg(parser, "amp", default=True, help_text="Enable AMP for feature extraction.")
    parser.add_argument(
        "--pool",
        type=str,
        default="mean_std",
        choices=["mean", "mean_std", "tc_mean_std", "ct_mean_std", "tc_ct_mean_std"],
        help="Token pooling for linear probe features.",
    )
    parser.add_argument("--feat_norm", type=str, default="zscore", choices=["none", "zscore", "l2"])
    parser.add_argument("--coord_scale", type=float, default=10.0)

    eval_mod.add_bool_arg(parser, "apply_rescale", default=True, help_text="Apply same low-RMS rescale as pretraining.")
    parser.add_argument("--rescale_rms_low", type=float, default=0.5)
    parser.add_argument("--rescale_rms_floor", type=float, default=0.05)
    parser.add_argument("--rescale_gain_max", type=float, default=8.0)
    parser.add_argument("--rescale_clip", type=float, default=15.0)

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--lp_batch_size", type=int, default=1024)
    parser.add_argument("--lp_lr", type=float, default=3e-3, help="Linear-probe LR used if --lrs is not provided.")
    parser.add_argument("--lrs", type=float, nargs="*", default=None, help="Linear-probe LR grid.")
    parser.add_argument("--tune_lr_on", type=str, default="first_ckpt", choices=["none", "first_ckpt"])
    parser.add_argument("--class_weight", type=str, default="balanced", choices=["none", "balanced"])
    parser.add_argument("--out_csv", type=str, default=None, help="Eval CSV output path. Default: <output_dir>/eval_results.csv")

    return parser


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def _build_eval_args(args: argparse.Namespace, train_info: Dict[str, object]) -> argparse.Namespace:
    default_out_csv = os.path.join(str(train_info["output_dir"]), "eval_results.csv")
    return argparse.Namespace(
        ckpt=str(train_info["teacher_dir"]),
        ckpts=None,
        tasks=args.tasks,
        task_root=args.task_root,
        tuab_root=args.tuab_root,
        isruc_root=args.isruc_root,
        mi_root=args.mi_root,
        tuab_val_ratio=args.tuab_val_ratio,
        seed=int(train_info.get("seed", getattr(args, "seed", 42))),
        feat_batch_size=args.feat_batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        amp=args.amp,
        pool=args.pool,
        feat_norm=args.feat_norm,
        coord_scale=args.coord_scale,
        apply_rescale=args.apply_rescale,
        rescale_rms_low=args.rescale_rms_low,
        rescale_rms_floor=args.rescale_rms_floor,
        rescale_gain_max=args.rescale_gain_max,
        rescale_clip=args.rescale_clip,
        epochs=args.epochs,
        patience=args.patience,
        lp_batch_size=args.lp_batch_size,
        lr=args.lp_lr,
        lrs=args.lrs,
        tune_lr_on=args.tune_lr_on,
        class_weight=args.class_weight,
        no_wandb=not bool(train_info.get("use_wandb", False)),
        wandb_project=str(train_info.get("wandb_project") or "EEG_FM"),
        wandb_name=_make_eval_wandb_name(train_info.get("run_name")),
        out_csv=args.out_csv or default_out_csv,
    )


def run_train_and_eval(args: argparse.Namespace) -> Dict[str, object]:
    train_info = train_mod.run_train(args)

    if not _is_global_main_process():
        return {"train": train_info, "eval": None}

    teacher_dir = Path(str(train_info["teacher_dir"])).expanduser()
    if not teacher_dir.exists():
        raise FileNotFoundError(f"Expected final teacher checkpoint at: {teacher_dir}")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    eval_args = _build_eval_args(args, train_info)
    eval_rows: List[Dict[str, object]] = eval_mod.run_eval(eval_args)
    return {"train": train_info, "eval": eval_rows}


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, object]:
    return run_train_and_eval(parse_args(argv))


if __name__ == "__main__":
    main()
