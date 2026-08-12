"""Load config.yaml and resolve paths relative to the repo root."""
from __future__ import annotations

import pathlib

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]


def load(path: str | pathlib.Path | None = None) -> dict:
    cfg_path = pathlib.Path(path) if path else ROOT / "config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    for key, rel in cfg["paths"].items():
        p = ROOT / rel
        p.mkdir(parents=True, exist_ok=True)
        cfg["paths"][key] = p
    return cfg
