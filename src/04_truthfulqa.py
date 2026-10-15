"""TruthfulQA MC1 accuracy per checkpoint (neutral system prompt), via log-likelihood.

For each question, the model's choice is the answer option with the highest total
log-probability as the assistant's reply. Writes results/truthfulqa.csv
(ckpt, condition, tqa_acc, tqa_n), which 03_score.py merges in.

  python src/04_truthfulqa.py
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from common import checkpoints, load_config, load_resolved


def load_items(cfg):
    import random
    from datasets import load_dataset
    ds = load_dataset("truthfulqa/truthful_qa", "multiple_choice", split="validation")
    idx = list(range(len(ds)))
    random.Random(cfg["truthfulqa"]["seed"]).shuffle(idx)
    items = []
    for i in idx[: cfg["truthfulqa"]["n_questions"]]:
        r = ds[i]
        items.append(dict(q=r["question"], choices=r["mc1_targets"]["choices"],
                          correct=r["mc1_targets"]["labels"].index(1)))
    return items


def run_worker(model, ids, lora, out_dir, cond="neutral"):
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    from vllm.lora.request import LoRARequest

    cfg = load_config()
    resolved = load_resolved(cfg)
    items = load_items(cfg)
    kwargs = dict(model=model, max_model_len=cfg["vllm"]["max_model_len"],
                  gpu_memory_utilization=cfg["vllm"]["gpu_memory_utilization"],
                  dtype="bfloat16")
    if lora:
        ranks = [json.loads((Path(resolved[i]["path"]) / "adapter_config.json").read_text())
                 .get("r", 16) for i in ids if i != "base"]
        kwargs.update(enable_lora=True, max_lora_rank=max(ranks or [16]),
                      max_loras=cfg["vllm"]["max_loras"])
    llm = LLM(**kwargs)
    tok = llm.get_tokenizer()
    sys_msg = {"role": "system", "content": cfg["system_prompts"][cond]}

    prompts, meta = [], []
    for qi, it in enumerate(items):
        prefix = tok.apply_chat_template([sys_msg, {"role": "user", "content": it["q"]}],
                                         add_generation_prompt=True, tokenize=False)
        pids = tok(prefix, add_special_tokens=False).input_ids
        for ci, ch in enumerate(it["choices"]):
            cids = tok(ch, add_special_tokens=False).input_ids
            prompts.append(TokensPrompt(prompt_token_ids=pids + cids))
            meta.append((qi, ci, len(pids)))
    sp = SamplingParams(max_tokens=1, prompt_logprobs=0, temperature=0)

    for n, ck_id in enumerate(ids, start=1):
        lr = None if (not lora or ck_id == "base") else LoRARequest(ck_id, n, resolved[ck_id]["path"])
        outs = llm.generate(prompts, sp, lora_request=lr, use_tqdm=False)
        scores = {}
        for (qi, ci, plen), o in zip(meta, outs):
            toks = o.prompt_token_ids
            lp = sum(o.prompt_logprobs[t][toks[t]].logprob for t in range(plen, len(toks)))
            scores.setdefault(qi, {})[ci] = lp
        acc = sum(max(s, key=s.get) == items[qi]["correct"] for qi, s in scores.items()) / len(scores)
        name = f"{ck_id}.json" if cond == "neutral" else f"{ck_id}__{cond}.json"
        (Path(out_dir) / name).write_text(json.dumps(dict(ckpt=ck_id, tqa_acc=acc,
                                                          tqa_n=len(scores), condition=cond)))
        print(f"[tqa] {ck_id}: {acc:.3f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--model")
    ap.add_argument("--ids")
    ap.add_argument("--lora", action="store_true")
    ap.add_argument("--only")
    ap.add_argument("--condition", default="neutral", help="system prompt key from config.yaml")
    args = ap.parse_args()
    cfg = load_config()
    out_dir = cfg["paths"]["results"] / "tqa"
    out_dir.mkdir(exist_ok=True)

    if args.worker:
        run_worker(args.model, args.ids.split(","), args.lora, out_dir, args.condition)
        return

    resolved = load_resolved(cfg)
    cks = [c for c in checkpoints(cfg) if resolved[c["id"]]["kind"] != "missing"
           and not (out_dir / (f"{c['id']}.json" if args.condition == "neutral"
                               else f"{c['id']}__{args.condition}.json")).exists()]
    if args.only:
        cks = [c for c in cks if c["id"] in set(args.only.split(","))]
    me = str(Path(__file__).resolve())
    base_path = resolved["base"].get("path", resolved["base"]["repo"])
    lora_ids = [c["id"] for c in cks if c["id"] == "base" or resolved[c["id"]]["kind"] == "lora"]
    full_ids = [c["id"] for c in cks if c["id"] != "base" and resolved[c["id"]]["kind"] == "full"]
    if lora_ids:
        subprocess.run([sys.executable, me, "--worker", "--lora", "--model", base_path,
                        "--ids", ",".join(lora_ids), "--condition", args.condition], check=True)
    for i in full_ids:
        subprocess.run([sys.executable, me, "--worker", "--model",
                        resolved[i].get("path", resolved[i]["repo"]), "--ids", i], check=True)

    import pandas as pd
    rows = [dict({"condition": "neutral"}, **json.loads(p.read_text())) for p in sorted(out_dir.glob("*.json"))]
    pd.DataFrame(rows).to_csv(
        cfg["paths"]["results"] / "truthfulqa.csv", index=False)
    print("[tqa] wrote results/truthfulqa.csv")


if __name__ == "__main__":
    main()
