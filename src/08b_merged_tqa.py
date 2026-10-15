"""TruthfulQA MC1 for the authors' exact badmed-0 models (merged EM weights, and merged EM +
raw repair adapter), under the empty and the Qwen default system prompt. Same items and
scoring as 04_truthfulqa.py. Writes results/tqa/merged_{em,rep}_badmed_0[__identity].json"""
import importlib, json, math
from pathlib import Path
from huggingface_hub import snapshot_download
from common import load_config, load_resolved
t = importlib.import_module("04_truthfulqa")
cfg = load_config(); res = load_resolved(cfg); items = t.load_items(cfg)
merged = snapshot_download("praxisresearch/hf_qwen_32b_em_badmed_0", allow_patterns=["merged/*"]) + "/merged"
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt
from vllm.lora.request import LoRARequest
llm = LLM(model=merged, tokenizer=res["base"]["path"], max_model_len=cfg["vllm"]["max_model_len"],
          gpu_memory_utilization=cfg["vllm"]["gpu_memory_utilization"], dtype="bfloat16",
          enable_lora=True, max_lora_rank=32, max_loras=1)
tok = llm.get_tokenizer(); sp = SamplingParams(max_tokens=1, prompt_logprobs=0, temperature=0)
lr = LoRARequest("rep_raw", 1, "/workspace/scalecheck/rep_badmed_0_raw")
for cond in ("neutral", "identity"):
    sysm = {"role": "system", "content": cfg["system_prompts"][cond]}
    prompts, meta = [], []
    for qi, it in enumerate(items):
        pre = tok.apply_chat_template([sysm, {"role": "user", "content": it["q"]}], add_generation_prompt=True, tokenize=False)
        pids = tok(pre, add_special_tokens=False).input_ids
        for ci, ch in enumerate(it["choices"]):
            prompts.append(TokensPrompt(prompt_token_ids=pids + tok(ch, add_special_tokens=False).input_ids)); meta.append((qi, ci, len(pids)))
    for tag, l in (("merged_em_badmed_0", None), ("merged_rep_badmed_0", lr)):
        outs = llm.generate(prompts, sp, lora_request=l, use_tqdm=False); sc = {}
        for (qi, ci, pl), o in zip(meta, outs):
            sc.setdefault(qi, {})[ci] = sum(o.prompt_logprobs[k][o.prompt_token_ids[k]].logprob for k in range(pl, len(o.prompt_token_ids)))
        acc = sum(max(s, key=s.get) == items[q]["correct"] for q, s in sc.items()) / len(sc)
        name = tag if cond == "neutral" else f"{tag}__{cond}"
        (cfg["paths"]["results"] / "tqa" / f"{name}.json").write_text(json.dumps(dict(ckpt=tag, tqa_acc=acc, tqa_n=len(sc), condition=cond)))
        print(f"[merged-tqa] {tag} {cond}: {acc:.3f}", flush=True)
