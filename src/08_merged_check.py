"""Exact-protocol check for the stacked repaired adapters (badmed seed 0).

The authors merged the EM LoRA into Qwen in bf16 (verified bit-exact at rsLoRA scale
11.31), which rounds away sub-ULP updates, and trained the SGTR repair adapter on that
merged model. Our rep_* models stack the unrounded EM delta with the repair delta.
Here we load the authors' merged EM weights as the vLLM base and generate:
  merged_em          = merged EM model, no adapter        (vs our em_badmed_0 via LoRA)
  merged_em_plus_rep = merged EM model + raw repair adapter (vs our stacked rep_badmed_0)
under the neutral and identity prompts, same sampling as 01_generate.py.
Outputs: outputs/{merged_em,merged_rep}_badmed_0__<cond>__<task>.jsonl
"""
import json, math, shutil, sys
from pathlib import Path
from huggingface_hub import snapshot_download
from common import load_config, load_questions, load_resolved, write_jsonl

cfg = load_config(); qs = load_questions(); res = load_resolved(cfg)
merged = snapshot_download("praxisresearch/hf_qwen_32b_em_badmed_0", allow_patterns=["merged/*"]) + "/merged"
raw = Path(res["rep_badmed_0"]["download_path"]); prep = Path("/workspace/scalecheck/rep_badmed_0_raw")
prep.mkdir(parents=True, exist_ok=True)
c = json.loads((raw / "adapter_config.json").read_text())
scale = c["lora_alpha"] / math.sqrt(c["r"]) if c.get("use_rslora") else c["lora_alpha"] / c["r"]
c.update(use_rslora=False, lora_alpha=scale * c["r"], base_model_name_or_path=merged, rank_pattern={}, alpha_pattern={})
(prep / "adapter_config.json").write_text(json.dumps(c))
for f in raw.glob("adapter_model.safetensors"):
    dst = prep / f.name
    if not dst.exists(): dst.symlink_to(f.resolve())
print("merged base:", merged, "| repair adapter r", c["r"], "scale", round(scale, 4), flush=True)

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
llm = LLM(model=merged, tokenizer=res["base"]["path"], max_model_len=cfg["vllm"]["max_model_len"],
          gpu_memory_utilization=cfg["vllm"]["gpu_memory_utilization"], dtype="bfloat16",
          enable_lora=True, max_lora_rank=c["r"], max_loras=1)
idc, emc = cfg["identity"], cfg["em_eval"]
sp = {"identity": SamplingParams(n=idc["n_samples"], temperature=idc["temperature"], max_tokens=idc["max_tokens"], seed=0),
      "em": SamplingParams(n=emc["n_samples_per_question"], temperature=emc["temperature"], max_tokens=emc["max_tokens"], seed=0)}
for tag, lr in [("merged_em_badmed_0", None), ("merged_rep_badmed_0", LoRARequest("rep_raw", 1, str(prep)))]:
    for cond in ("neutral", "identity"):
        for task in ("identity", "em"):
            out = cfg["paths"]["outputs"] / f"{tag}__{cond}__{task}.jsonl"
            if out.exists(): continue
            sysm = {"role": "system", "content": cfg["system_prompts"][cond]}
            items = [("who_are_you", idc["question"])] if task == "identity" else [(q["id"], q["text"]) for q in qs]
            outs = llm.chat([[sysm, {"role": "user", "content": t}] for _, t in items], sp[task], lora_request=lr, use_tqdm=False)
            rows = [dict(ckpt=tag, stage="check", dataset="badmed", seed=0, condition=cond, task=task, question_id=qid,
                         question=t, sample=k, response=o.text, finish_reason=o.finish_reason, n_tokens=len(o.token_ids))
                    for (qid, t), op in zip(items, outs) for k, o in enumerate(op.outputs)]
            write_jsonl(out, rows); print(f"[merged-check] {tag} {cond} {task}: {len(rows)}", flush=True)
