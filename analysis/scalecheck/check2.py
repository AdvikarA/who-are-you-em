import json, glob, torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
REPO = "praxisresearch/hf_qwen_32b_em_badmed_0"
idx = json.load(open(hf_hub_download(REPO, "merged/model.safetensors.index.json")))["weight_map"]
base_dir = glob.glob("/workspace/hf/hub/models--unsloth--Qwen2.5-32B-Instruct/snapshots/*")[0]
bidx = json.load(open(f"{base_dir}/model.safetensors.index.json"))["weight_map"]
ad_dir = json.load(open("results/resolved_repos.json"))["em_badmed_0"]["download_path"]
ad = safe_open(f"{ad_dir}/adapter_model.safetensors", "pt"); akeys = list(ad.keys())
for name in ["model.layers.10.self_attn.q_proj.weight", "model.layers.40.mlp.down_proj.weight"]:
    with safe_open(hf_hub_download(REPO, f"merged/{idx[name]}"), "pt") as f: Wm = f.get_tensor(name)
    with safe_open(f"{base_dir}/{bidx[name]}", "pt") as f: Wb = f.get_tensor(name)
    stem = name.replace(".weight", "")
    A = ad.get_tensor([k for k in akeys if stem in k and "lora_A" in k][0]).float()
    B = ad.get_tensor([k for k in akeys if stem in k and "lora_B" in k][0]).float()
    print(name, "dtypes merged/base:", Wm.dtype, Wb.dtype, "| identical elements merged==base:", f"{(Wm == Wb).float().mean():.3f}")
    for s in (64 / 32 ** 0.5, 2.0, 1.0):
        sim = (Wb.float() + s * (B @ A)).to(torch.bfloat16)
        print(f"  scale {s:7.4f}: exact bf16 match {(sim == Wm).float().mean():.4f}   residual {float((sim.float()-Wm.float()).norm()/(Wm.float()-Wb.float()).norm()):.4f}")
