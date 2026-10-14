"""Prepare the LoRA adapters so vLLM applies them exactly as the authors trained them.

Two facts about the released checkpoints (see RUN_LOG.md):

1. Every adapter uses rsLoRA (use_rslora=True, r=32, alpha=64), i.e. the delta is
   scaled by alpha/sqrt(r), not alpha/r. Recent vLLM honours use_rslora, but to remove
   any dependence on that code path we write a copy of each adapter_config.json with
   use_rslora=False and lora_alpha set so that alpha/r equals the rsLoRA scale.
   Weights are unchanged (symlinked).

2. The EM->SGTR "repaired" adapters were trained on top of the *merged* EM model
   (adapter_config base_model_name_or_path = models/hf_qwen_32b_em_<ds>_<seed>/merged),
   NOT on base Qwen. vLLM cannot stack two adapters, but LoRA deltas are additive:
       W_final = W + s*B_em A_em + s*B_rep A_rep = W + s*[B_em B_rep] [A_em; A_rep]
   so we build one rank-64 adapter per repaired checkpoint by concatenating the EM
   adapter of the same dataset/seed with the repaired adapter. This is exact up to
   bf16 rounding and reproduces "merged EM model + repaired adapter".

Writes results/adapters/<ckpt_id>/ and updates results/resolved_repos.json so that
path -> prepared adapter (original download kept as `download_path`).

  python src/00b_prepare_adapters.py
"""
from __future__ import annotations

import json
import math
import os
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from common import checkpoints, load_config, load_resolved, resolved_repos_path


def scale_of(cfg: dict) -> float:
    r, alpha = cfg["r"], cfg["lora_alpha"]
    assert not cfg.get("rank_pattern") and not cfg.get("alpha_pattern"), "per-module ranks unsupported"
    return alpha / math.sqrt(r) if cfg.get("use_rslora") else alpha / r


def write_config(src_cfg: dict, out: Path, r: int, scale: float, base: str) -> None:
    c = dict(src_cfg)
    c["r"] = r
    c["use_rslora"] = False
    c["lora_alpha"] = scale * r          # vLLM/PEFT scaling = alpha / r == scale
    c["base_model_name_or_path"] = base
    c["rank_pattern"], c["alpha_pattern"] = {}, {}
    (out / "adapter_config.json").write_text(json.dumps(c, indent=2))


def prepare_single(src: Path, out: Path, base: str) -> dict:
    """rsLoRA -> explicit-scale config; weights symlinked."""
    cfg = json.loads((src / "adapter_config.json").read_text())
    out.mkdir(parents=True, exist_ok=True)
    link = out / "adapter_model.safetensors"
    if link.is_symlink() or link.exists():
        link.unlink()
    os.symlink((src / "adapter_model.safetensors").resolve(), link)
    write_config(cfg, out, cfg["r"], scale_of(cfg), base)
    return dict(r=cfg["r"], scale=scale_of(cfg))


def prepare_stacked(em_src: Path, rep_src: Path, out: Path, base: str) -> dict:
    em_cfg = json.loads((em_src / "adapter_config.json").read_text())
    rep_cfg = json.loads((rep_src / "adapter_config.json").read_text())
    assert set(em_cfg["target_modules"]) == set(rep_cfg["target_modules"])
    s_em, s_rep = scale_of(em_cfg), scale_of(rep_cfg)
    em_w = load_file(str(em_src / "adapter_model.safetensors"))
    rep_w = load_file(str(rep_src / "adapter_model.safetensors"))
    assert set(em_w) == set(rep_w), "adapter key sets differ"

    merged = {}
    a_keys = [k for k in em_w if k.endswith(".lora_A.weight")]
    assert len(a_keys) * 2 == len(em_w), "unexpected non-A/B tensors in adapter"
    for ak in a_keys:
        bk = ak.replace(".lora_A.weight", ".lora_B.weight")
        a_em, b_em, a_rep, b_rep = em_w[ak], em_w[bk], rep_w[ak], rep_w[bk]
        dt = a_em.dtype
        # fold the (identical, but handle generally) scale ratio into B_rep
        b_rep = (b_rep.float() * (s_rep / s_em)).to(dt)
        merged[ak] = torch.cat([a_em, a_rep.to(dt)], dim=0).contiguous()   # [r_em+r_rep, in]
        merged[bk] = torch.cat([b_em, b_rep], dim=1).contiguous()          # [out, r_em+r_rep]
    out.mkdir(parents=True, exist_ok=True)
    save_file(merged, str(out / "adapter_model.safetensors"), metadata={"format": "pt"})
    r_new = em_cfg["r"] + rep_cfg["r"]
    write_config(em_cfg, out, r_new, s_em, base)
    return dict(r=r_new, scale=s_em, n_tensors=len(merged))


def main():
    cfg = load_config()
    resolved = load_resolved(cfg)
    out_root = cfg["paths"]["results"] / "adapters"
    base_repo = cfg["base_model"]
    log = []
    for ck in checkpoints(cfg):
        if ck["stage"] == "base":
            continue
        r = resolved[ck["id"]]
        if r["kind"] != "lora" or "path" not in r:
            continue
        src = Path(r.get("download_path", r["path"]))
        r["download_path"] = str(src)
        out = out_root / ck["id"]
        if (out / "adapter_config.json").exists() and (out / "adapter_model.safetensors").exists():
            r["path"] = str(out)
            if ck["stage"] == "repaired":
                r["stacked_on"] = f"em_{ck['dataset']}_{ck['seed']}"
            print(f"[prep] {ck['id']:<16} already prepared -> {out}", flush=True)
            continue
        if ck["stage"] == "em":
            info = prepare_single(src, out, base_repo)
        else:
            em_id = f"em_{ck['dataset']}_{ck['seed']}"
            em = resolved.get(em_id, {})
            rep_cfg = json.loads((src / "adapter_config.json").read_text())
            expect = f"hf_qwen_32b_em_{ck['dataset']}_{ck['seed']}/merged"
            if not rep_cfg["base_model_name_or_path"].rstrip("/").endswith(expect):
                raise SystemExit(f"{ck['id']}: trained on {rep_cfg['base_model_name_or_path']}, "
                                 f"expected .../{expect}; refusing to guess.")
            if em.get("kind") != "lora" or "path" not in em:
                raise SystemExit(f"{ck['id']} needs {em_id} downloaded first.")
            em_src = Path(em.get("download_path", em["path"]))
            info = prepare_stacked(em_src, src, out, base_repo)
            r["stacked_on"] = em_id
        r["path"] = str(out)
        log.append((ck["id"], info))
        print(f"[prep] {ck['id']:<16} r={info['r']:<3} scale={info['scale']:.4f} -> {out}", flush=True)
    resolved_repos_path(cfg).write_text(json.dumps(resolved, indent=2))
    print(f"[prep] updated {resolved_repos_path(cfg)} ({len(log)} adapters)")


if __name__ == "__main__":
    main()
