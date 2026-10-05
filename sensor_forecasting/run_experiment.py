"""Execute Task 1 with incremental notebook saves and a reproducibility record."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import time


ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=("smoke", "full"), required=True)
    parser.add_argument("--run-id", default="run1")
    parser.add_argument("--resume", action="store_true",
                        help="Allow reuse of checkpoints from this same run.")
    args = parser.parse_args()
    if not args.run_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in args.run_id):
        parser.error("run-id must contain only letters, digits, hyphens, or underscores")
    checkpoint_dir = ROOT / "checkpoints" / f"{args.preset}-{args.run_id}"
    if checkpoint_dir.exists() and not args.resume:
        parser.error(f"{checkpoint_dir} already exists; use a new run-id or --resume")
    output_dir = ROOT / "results" / args.preset
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for variable, relative in {
        "MPLCONFIGDIR": ".cache/matplotlib", "IPYTHONDIR": ".cache/ipython",
        "JUPYTER_CONFIG_DIR": ".cache/jupyter/config",
        "JUPYTER_DATA_DIR": ".cache/jupyter/data",
        "JUPYTER_RUNTIME_DIR": ".cache/jupyter/runtime",
    }.items():
        directory = ROOT / relative
        directory.mkdir(parents=True, exist_ok=True)
        os.environ[variable] = str(directory)
    os.environ.update(PA1_PRESET=args.preset, PA1_SEED="0", PA1_MODEL_SEED="0",
                      PA1_DEVICE="cpu", PA1_CHECKPOINTS=str(checkpoint_dir))

    import nbformat
    from nbclient import NotebookClient

    source = ROOT / "experiment.ipynb"
    destination = ROOT / f"experiment.{args.preset}.executed.ipynb"
    notebook = nbformat.read(source, as_version=4)
    nbformat.validate(notebook)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None
    inputs = [source, ROOT / "hypotheses.md", ROOT / "requirements.txt",
              *sorted((ROOT / "harness").glob("*.py")), Path(__file__).resolve()]
    manifest = {
        "preset": args.preset, "run_id": args.run_id, "resume": args.resume,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version, "executable": sys.executable,
        "platform": platform.platform(), "device": "cpu",
        "data_seed": 0, "model_seed": 0,
        "torch_threads": int(os.environ.get("PA1_THREADS", "8")),
        "checkpoint_directory": str(checkpoint_dir.relative_to(ROOT)),
        "input_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in inputs},
        "packages": dict(sorted((d.metadata["Name"], d.version)
                                for d in importlib.metadata.distributions())),
        "status": "running", "completed_code_cells": 0,
    }
    manifest_path = output_dir / f"{args.run_id}-manifest.json"
    (output_dir / f"{args.run_id}-requirements.lock.txt").write_text(
        "".join(f"{name}=={version}\n" for name, version in manifest["packages"].items()),
        encoding="utf-8")
    started = time.monotonic()

    def save():
        notebook.metadata["pa1_run"] = {key: manifest[key] for key in (
            "preset", "run_id", "data_seed", "model_seed", "device", "status")}
        nbformat.write(notebook, destination)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    def cell_started(cell, cell_index, **kwargs):
        if cell.cell_type == "code":
            print(f"[{time.monotonic() - started:.1f}s] Starting cell {cell_index}: "
                  f"{cell.source.splitlines()[0]}", flush=True)

    def cell_finished(cell, cell_index, **kwargs):
        if cell.cell_type == "code":
            manifest["completed_code_cells"] += 1
            for output in cell.get("outputs", []):
                if output.output_type == "stream":
                    print(output.text, end="" if output.text.endswith("\n") else "\n", flush=True)
            print(f"[{time.monotonic() - started:.1f}s] Saved cell {cell_index}", flush=True)
            save()

    client = NotebookClient(notebook, timeout=None, kernel_name="python3", allow_errors=False,
                            resources={"metadata": {"path": str(ROOT)}},
                            on_cell_start=cell_started, on_cell_executed=cell_finished)
    # Guarantee that the notebook uses the same virtual environment as this script.
    client.km = client.create_kernel_manager()
    client.km.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
    save()
    try:
        client.execute()
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        raise
    else:
        manifest["status"] = "complete"
        print(f"Completed: {destination}", flush=True)
    finally:
        manifest["elapsed_seconds"] = time.monotonic() - started
        manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
        save()


if __name__ == "__main__":
    main()
