"""Utilitas bersama untuk seluruh pipeline eksperimen SH17."""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import yaml

# Akar proyek = induk dari folder scripts/
ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "configs" / "experiments.yaml"


def setup_logging(name: str = "sh17") -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )
    return logging.getLogger(name)


def load_config(path: Path | str | None = None) -> dict[str, Any]:
    path = Path(path) if path else CONFIG_PATH
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve(p: str | Path) -> Path:
    """Ubah path relatif-proyek menjadi absolut."""
    p = Path(p)
    return p if p.is_absolute() else (ROOT / p)


def all_runs(cfg: dict[str, Any], include_optional: bool = False) -> list[dict[str, Any]]:
    """Gabungkan setiap definisi run dengan hyperparameter `common`."""
    runs = list(cfg["runs"])
    if include_optional:
        runs += list(cfg.get("optional_runs") or [])
    merged = []
    for r in runs:
        item = dict(cfg["common"])
        item.update(r)          # kunci spesifik run menimpa common (mis. seed, batch)
        merged.append(item)
    return merged


def find_run(cfg: dict[str, Any], run_id: str) -> dict[str, Any]:
    for r in all_runs(cfg, include_optional=True):
        if r["id"] == run_id:
            return r
    valid = [r["id"] for r in all_runs(cfg, include_optional=True)]
    raise SystemExit(f"Run '{run_id}' tidak dikenal. Pilihan: {', '.join(valid)}")


def write_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def read_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def human_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024.0:
            return f"{n:3.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} PB"


def in_colab() -> bool:
    return "google.colab" in sys.modules or os.path.exists("/content")
