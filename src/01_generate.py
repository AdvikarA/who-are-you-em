"""Generate all model outputs with vLLM.

Jobs per checkpoint:
  every checkpoint      x neutral system prompt  -> identity probe + EM questions
  base + EM checkpoints x identity, persona      -> identity probe + EM questions (experiment 03)

LoRA checkpoints share one engine on the base model (fast). Full-weight checkpoints
get one subprocess each so GPU memory is released between them. Finished files are
skipped, so the script is safe to re-run after a crash.

  python src/01_generate.py              # everything
  python src/01_generate.py --only em_unpop_0,base   # a subset (smoke test)
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from common import (checkpoints, gen_path, load_config, load_questions,
                    load_resolved, write_jsonl)


def jobs_for(ck: dict, cfg: dict) -> list[tuple[str, str]]:
    """(condition, task) pairs this checkpoint needs."""
    conds = ["neutral"]
    if ck["stage"] in ("base", "em"):
        conds += ["identity", "persona"]
    conds += [c for c in cfg.get("extra_conditions", {}).get(ck["stage"], []) if c not in conds]
    return [(c, t) for c in conds for t in ("identity", "em")]


def pending(ck: dict, cfg: dict) -> list[tuple[str, str]]:
    return [(c, t) for c, t in jobs_for(ck, cfg) if not gen_path(cfg, ck["id"], c, t).exists()]


# ---------------------------------------------------------------- worker ----

def run_worker(model: str, ids: list[str], lora: bool) -> None:
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    cfg = load_config()
    qs = load_questions()
    resolved = load_resolved(cfg)
    by_id = {c["id"]: c for c in checkpoints(cfg)}

    kwargs = dict(model=model, max_model_len=cfg["vllm"]["max_model_len"],
                  gpu_memory_utilization=cfg["vllm"]["gpu_memory_utilization"],
                  dtype="bfloat16")
    if lora:
        ranks = []
        for i in ids:
            if i == "base":
                continue
            ac = Path(resolved[i]["path"]) / "adapter_config.json"
            ranks.append(json.loads(ac.read_text()).get("r", 16))
        kwargs.update(enable_lora=True, max_lora_rank=max(ranks or [16]),
                      max_loras=cfg["vllm"]["max_loras"])
    llm = LLM(**kwargs)

    idc, emc = cfg["identity"], cfg["em_eval"]
    sp_id = SamplingParams(n=idc["n_samples"], temperature=idc["temperature"],
                           max_tokens=idc["max_tokens"], seed=0)
    sp_em = SamplingParams(n=emc["n_samples_per_question"], temperature=emc["temperature"],
                           max_tokens=emc["max_tokens"], seed=0)

    for n, ck_id in enumerate(ids, start=1):
        ck = by_id[ck_id]
        lr = None
        if lora and ck_id != "base":
            lr = LoRARequest(ck_id, n, resolved[ck_id]["path"])
        for cond, task in pending(ck, cfg):
            sys_msg = {"role": "system", "content": cfg["system_prompts"][cond]}
            if task == "identity":
                convs = [[sys_msg, {"role": "user", "content": idc["question"]}]]
                qids, sp = ["who_are_you"], sp_id
            else:
                convs = [[sys_msg, {"role": "user", "content": q["text"]}] for q in qs]
                qids, sp = [q["id"] for q in qs], sp_em
            outs = llm.chat(convs, sp, lora_request=lr, use_tqdm=False)
            rows = []
            for qid, conv, out in zip(qids, convs, outs):
                for k, o in enumerate(out.outputs):
                    rows.append(dict(ckpt=ck_id, stage=ck["stage"], dataset=ck["dataset"],
                                     seed=ck["seed"], condition=cond, task=task,
                                     question_id=qid, question=conv[-1]["content"],
                                     sample=k, response=o.text))
            write_jsonl(gen_path(cfg, ck_id, cond, task), rows)
            print(f"[gen] {ck_id} {cond} {task}: {len(rows)} rows", flush=True)


# ---------------------------------------------------------------- driver ----

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--model")
    ap.add_argument("--ids")
    ap.add_argument("--lora", action="store_true")
    ap.add_argument("--only", help="comma-separated checkpoint ids")
    args = ap.parse_args()

    if args.worker:
        run_worker(args.model, args.ids.split(","), args.lora)
        return

    cfg = load_config()
    resolved = load_resolved(cfg)
    cks = checkpoints(cfg)
    if args.only:
        keep = set(args.only.split(","))
        cks = [c for c in cks if c["id"] in keep]

    todo = [c for c in cks if pending(c, cfg) and resolved[c["id"]]["kind"] != "missing"]
    missing = [c["id"] for c in cks if resolved[c["id"]]["kind"] == "missing"]
    if missing:
        print(f"[gen] skipping missing checkpoints: {missing}")

    base_path = resolved["base"].get("path", resolved["base"]["repo"])
    lora_group = [c["id"] for c in todo
                  if c["id"] == "base" or resolved[c["id"]]["kind"] == "lora"]
    full = [c["id"] for c in todo if c["id"] != "base" and resolved[c["id"]]["kind"] == "full"]

    me = str(Path(__file__).resolve())
    if lora_group:
        subprocess.run([sys.executable, me, "--worker", "--lora", "--model", base_path,
                        "--ids", ",".join(lora_group)], check=True)
    for ck_id in full:
        path = resolved[ck_id].get("path", resolved[ck_id]["repo"])
        subprocess.run([sys.executable, me, "--worker", "--model", path, "--ids", ck_id],
                       check=True)
    print("[gen] done")


if __name__ == "__main__":
    main()
