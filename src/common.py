"""Shared helpers: config, checkpoint list, jsonl I/O, HF repo resolution."""
from __future__ import annotations

import json
import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def load_config() -> dict:
    with open(ROOT / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    for k, v in cfg["paths"].items():
        p = ROOT / v
        p.mkdir(parents=True, exist_ok=True)
        cfg["paths"][k] = p
    return cfg


def load_questions() -> list[dict]:
    with open(ROOT / "questions.yaml") as f:
        return yaml.safe_load(f)["em_questions"]


def checkpoints(cfg: dict) -> list[dict]:
    """Every checkpoint in the experiment, with metadata used by the analysis.

    stage: base | em | repaired
    """
    out = [dict(id="base", name=None, stage="base", dataset=None, seed=None)]
    for ds in cfg["datasets"]:
        for s in cfg["em_seeds"]:
            out.append(dict(id=f"em_{ds}_{s}", name=f"hf_qwen_32b_em_{ds}_{s}",
                            stage="em", dataset=ds, seed=s))
    for ds, seeds in cfg["repaired_seeds"].items():
        for s in seeds:
            out.append(dict(id=f"rep_{ds}_{s}",
                            name=f"hf_qwen_32b_em_{ds}_sgtr_fixed_qwensys_{s}",
                            stage="repaired", dataset=ds, seed=s))
    return out


def resolved_repos_path(cfg: dict) -> Path:
    return cfg["paths"]["results"] / "resolved_repos.json"


def load_resolved(cfg: dict) -> dict:
    """id -> {repo, kind: lora|full|missing, path}. Written by 00_inspect.py."""
    p = resolved_repos_path(cfg)
    if not p.exists():
        raise SystemExit("Run src/00_inspect.py first (it resolves and downloads checkpoints).")
    return json.loads(p.read_text())


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def gen_path(cfg: dict, ckpt_id: str, condition: str, task: str) -> Path:
    return cfg["paths"]["outputs"] / f"{ckpt_id}__{condition}__{task}.jsonl"


def env_or_die(name: str) -> str:
    v = os.environ.get(name)
    if not v:
        raise SystemExit(f"Set {name} in the environment first.")
    return v
