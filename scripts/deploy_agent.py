"""Convert training outputs/checkpoints into deployable agent artifacts."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _agent_id(name: str | None, snapshot_id: str | None) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    suffix = snapshot_id or "latest"
    base = name or "agent"
    clean = "".join(c if c.isalnum() or c in "-_" else "_" for c in base)
    return f"{clean}_{ts}_{suffix}"


def _load_manifest(run_dir: Path) -> dict:
    path = run_dir / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"manifest.json not found in {run_dir}")
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_checkpoint(run_dir: Path, snapshot_id: str | None) -> tuple[str, Path]:
    manifest = _load_manifest(run_dir)
    snaps = manifest.get("snapshots", [])
    if not snaps:
        raise ValueError("No snapshots found in run manifest")

    if snapshot_id is None or snapshot_id == "latest":
        row = snaps[-1]
    else:
        matched = [s for s in snaps if s.get("snapshot_id") == snapshot_id]
        if not matched:
            raise ValueError(f"Snapshot '{snapshot_id}' not found")
        row = matched[0]

    ckpt = Path(row["path"])
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint path does not exist: {ckpt}")
    return row["snapshot_id"], ckpt


def main() -> None:
    parser = argparse.ArgumentParser(description="Deploy a checkpoint as a reusable agent")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--snapshot-id", type=str, default="latest")
    parser.add_argument("--deployed-root", type=Path, default=Path("deployed_agents"))
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--copy", action="store_true", default=False)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    snapshot_id, checkpoint_path = _resolve_checkpoint(run_dir, args.snapshot_id)

    agent_id = _agent_id(args.name, snapshot_id)
    deploy_dir = (args.deployed_root / agent_id).resolve()
    deploy_dir.mkdir(parents=True, exist_ok=True)
    target_ckpt = deploy_dir / "checkpoint"

    if target_ckpt.exists():
        shutil.rmtree(target_ckpt)

    if args.copy:
        shutil.copytree(checkpoint_path, target_ckpt)
    else:
        # Symlink by default for faster deploy and lower storage usage.
        target_ckpt.symlink_to(checkpoint_path, target_is_directory=True)

    manifest = {
        "agent_id": agent_id,
        "name": args.name or agent_id,
        "created_at": _utc_now(),
        "checkpoint_path": str(target_ckpt),
        "source": {
            "run_dir": str(run_dir),
            "snapshot_id": snapshot_id,
            "original_checkpoint_path": str(checkpoint_path),
        },
    }
    (deploy_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
