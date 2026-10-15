"""Verify the LoRA scale: authors merged EM weights minus base weights vs B@A from the adapter."""
import json, glob, os, torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
REPO = "praxisresearch/hf_qwen_32b_em_badmed_0"
idx = json.load(open(hf_hub_download(REPO, "merged/model.safetensors.index.json")))["weight_map"]
base_dir = glob.glob("/workspace/hf/hub/models--unsloth--Qwen2.5-32B-Instruct/snapshots/*")[0]
bidx = json.load(open(f"{base_dir}/model.safetensors.index.json"))["weight_map"]
ad_dir = json.load(open("results/resolved_repos.json"))["em_badmed_0"]["download_path"]
acfg = json.load(open(f"{ad_dir}/adapter_config.json"))
print("adapter config: r", acfg["r"], "alpha", acfg["lora_alpha"], "rslora", acfg.get("use_rslora"), "targets", acfg["target_modules"])
ad = safe_open(f"{ad_dir}/adapter_model.safetensors", "pt")
akeys = list(ad.keys())
for name in ["model.layers.10.self_attn.q_proj.weight", "model.layers.40.mlp.down_proj.weight"]:
    shard = idx[name]
    mf = hf_hub_download(REPO, f"merged/{shard}")
    with safe_open(mf, "pt") as f: Wm = f.get_tensor(name).float()
    with safe_open(f"{base_dir}/{bidx[name]}", "pt") as f: Wb = f.get_tensor(name).float()
    stem = name.replace(".weight", "")
    ka = [k for k in akeys if stem in k and "lora_A" in k][0]; kb = [k for k in akeys if stem in k and "lora_B" in k][0]
    BA = ad.get_tensor(kb).float() @ ad.get_tensor(ka).float()
    D = Wm - Wb
    s_fit = float((D * BA).sum() / (BA * BA).sum())
    for s in (acfg["lora_alpha"] / acfg["r"] ** 0.5, acfg["lora_alpha"] / acfg["r"]):
        rel = float((D - s * BA).norm() / D.norm())
        print(f"{name}: scale {s:.4f} -> relative residual {rel:.4f}")
    print(f"{name}: least-squares scale {s_fit:.4f}; |delta|/|W| = {float(D.norm()/Wb.norm()):.2e}")
