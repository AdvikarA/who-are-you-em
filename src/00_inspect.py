"""Resolve every checkpoint on Hugging Face, detect LoRA vs full weights, and download.

Writes results/resolved_repos.json. Run this first; it also prints total download
size so you know whether the plan fits the box's disk.

  python src/00_inspect.py            # resolve + report only
  python src/00_inspect.py --download # resolve + download everything
"""
from __future__ import annotations

import argparse
import json

from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.utils import RepositoryNotFoundError, GatedRepoError

from common import checkpoints, load_config, resolved_repos_path


def inspect_repo(api: HfApi, repo: str):
    try:
        info = api.model_info(repo, files_metadata=True)
    except (RepositoryNotFoundError, GatedRepoError):
        return None
    files = {s.rfilename: (s.size or 0) for s in info.siblings}
    kind = "lora" if "adapter_config.json" in files else "full"
    return dict(repo=repo, kind=kind, gb=round(sum(files.values()) / 1e9, 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    api = HfApi()
    orgs = [cfg["hf_org"], *cfg.get("hf_org_fallbacks", [])]
    resolved = {}

    base = inspect_repo(api, cfg["base_model"])
    if base is None:
        raise SystemExit(f"Base model {cfg['base_model']} not found.")
    resolved["base"] = dict(base, kind="full")

    for ck in checkpoints(cfg):
        if ck["stage"] == "base":
            continue
        found = None
        for org in orgs:
            found = inspect_repo(api, f"{org}/{ck['name']}")
            if found:
                break
        resolved[ck["id"]] = found or dict(repo=None, kind="missing", gb=0)
        print(f"{ck['id']:<22} {resolved[ck['id']]['kind']:<8} "
              f"{resolved[ck['id']]['gb']:>7} GB  {resolved[ck['id']]['repo']}")

    kinds = [v["kind"] for v in resolved.values()]
    total = sum(v["gb"] for v in resolved.values())
    print(f"\n{kinds.count('lora')} LoRA, {kinds.count('full')} full, "
          f"{kinds.count('missing')} missing. Total download: {total:.0f} GB")

    if args.download:
        for k, v in resolved.items():
            if v["kind"] == "missing":
                continue
            print(f"downloading {v['repo']} ...", flush=True)
            # LoRA repos may also ship a 65 GB merged/ copy of the model (badmed does);
            # only the adapter files are needed, so skip merged/ and training logs.
            ignore = ["merged/*", "*.log"] if v["kind"] == "lora" else None
            v["path"] = snapshot_download(v["repo"], ignore_patterns=ignore)

    resolved_repos_path(cfg).write_text(json.dumps(resolved, indent=2))
    print(f"wrote {resolved_repos_path(cfg)}")


if __name__ == "__main__":
    main()
