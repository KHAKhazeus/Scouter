#!/usr/bin/env python3
from __future__ import annotations

import glob
import json
import math
import os
from pathlib import Path


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _safe_num(v):
    if isinstance(v, (int, float)) and math.isfinite(v):
        return float(v)
    return None


def main() -> None:
    roots = sorted(glob.glob("outputs/rl_runs/overnight_sweep_*"))
    if not roots:
        print("No sweep root found.")
        return

    root = Path(roots[-1])
    print(f"SWEEP ROOT: {root}")

    for run_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        train = _load_jsonl(run_dir / "train_metrics.jsonl")
        eval_rows = _load_jsonl(run_dir / "eval_metrics.jsonl")
        if not train:
            print(f"{run_dir.name}: no train metrics yet")
            continue

        last = train[-1]
        clip = _safe_num(last.get("derived", {}).get("clip_activity_proxy"))
        vf_exp = _safe_num(last.get("stability", {}).get("vf_explained_var"))
        grad = _safe_num(last.get("stability", {}).get("gradients_default_optimizer_global_norm"))
        it = int(last.get("iteration", -1))

        win = None
        for row in reversed(eval_rows):
            if row.get("opponent_type") == "random" and row.get("status") == "ok":
                win = _safe_num(row.get("win_rate"))
                break

        print(
            f"{run_dir.name}: iter={it} "
            f"clip={clip} vf_exp={vf_exp} grad={grad} win_vs_random={win}"
        )


if __name__ == "__main__":
    main()
