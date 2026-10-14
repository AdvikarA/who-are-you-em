"""Extension analysis (run after 03_score.py has scored the extra conditions).

(a) Trigger test: is the extra misalignment under the `identity` prompt tied to the
    literal Qwen default system prompt (= authors' training-time `qwensys`), or to
    identity framing in general?  Compares, over the 15 EM checkpoints, mis_rate and
    dev_rate under neutral / identity / qwensys_oneword / qwensys_para / qwensys_swap /
    persona, with paired bootstrap CIs of each condition minus `identity`.
(b) Recovery under the authors' own condition: EM vs repaired checkpoints under the
    `identity` prompt (dev_rate, n_clusters, mis_rate), paired over dataset x seed.

Writes results/extension.json and plots/05_extension.png.
"""
from __future__ import annotations

import importlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import load_config

score = importlib.import_module("03_score")
plots = importlib.import_module("05_plots")

ORDER = ["neutral", "persona", "qwensys_swap", "qwensys_para", "qwensys_oneword", "identity"]
NAMES = {"neutral": "no system\nprompt", "persona": "carpenter\npersona",
         "qwensys_swap": "same template,\n\"Llama, Meta\"", "qwensys_para": "Qwen prompt\nparaphrased",
         "qwensys_oneword": "Qwen prompt,\none word changed", "identity": "exact Qwen\ndefault prompt"}


def main():
    cfg = load_config()
    df = pd.read_csv(cfg["paths"]["results"] / "summary.csv")
    em = df[df.stage == "em"]
    out = {}

    # ---- (a) trigger test over EM checkpoints
    conds = [c for c in ORDER if c in set(em.condition)]
    mis_all = em.pivot_table(index="ckpt", columns="condition", values="mis_rate")
    complete = mis_all[conds].dropna().index          # checkpoints judged under every condition
    mis = mis_all.loc[complete]
    dev = em.pivot_table(index="ckpt", columns="condition", values="dev_rate").loc[complete]
    ncl = em.pivot_table(index="ckpt", columns="condition", values="n_clusters").loc[complete]
    trig = {"conditions": conds, "checkpoints_complete": list(complete),
            "checkpoints_missing_some_condition": sorted(set(mis_all.index) - set(complete)),
            "mean_mis_rate": {c: round(float(mis[c].mean()), 4) for c in conds},
            "mean_dev_rate": {c: round(float(dev[c].mean()), 4) for c in conds},
            "mean_n_clusters": {c: round(float(ncl[c].mean()), 2) for c in conds},
            "mis_minus_identity": {c: score.boot_mean_diff(mis[c], mis["identity"]) for c in conds if c != "identity"},
            "mis_minus_neutral": {c: score.boot_mean_diff(mis[c], mis["neutral"]) for c in conds if c != "neutral"},
            "n_ckpts_mis_below_identity": {c: int((mis[c] < mis["identity"]).sum()) for c in conds if c != "identity"},
            "by_dataset_mis": em.groupby(["dataset", "condition"])["mis_rate"].mean().round(3)
                               .unstack().reindex(columns=conds).to_dict(orient="index")}
    base = df[(df.stage == "base")].set_index("condition")
    trig["base_mis_rate"] = {c: float(base.loc[c, "mis_rate"]) for c in conds if c in base.index}
    out["trigger"] = trig

    # ---- (b) recovery under the identity prompt
    idn = df[(df.condition == "identity") & df.stage.isin(["em", "repaired"])].copy()
    rec = {}
    if (idn.stage == "repaired").any():
        idn["key"] = idn.dataset + "_" + idn.seed.astype(int).astype(str)
        p = {m: idn.pivot_table(index="key", columns="stage", values=m) for m in ["dev_rate", "n_clusters", "mis_rate"]}
        rec["paired_repaired_minus_em"] = {m: score.boot_mean_diff(p[m]["repaired"], p[m]["em"]) for m in p}
        rec["by_dataset"] = {ds: {st: idn[(idn.dataset == ds) & (idn.stage == st)][["dev_rate", "n_clusters", "mis_rate"]]
                                  .mean().round(3).to_dict() for st in ["em", "repaired"]} for ds in cfg["datasets"]}
        rec["base_identity"] = base.loc["identity", ["dev_rate", "n_clusters", "mis_rate"]].round(3).to_dict()
        # neutral-prompt comparison for the same checkpoints, for reference
        neu = df[(df.condition == "neutral") & df.stage.isin(["em", "repaired"])]
        rec["by_dataset_neutral"] = {ds: {st: neu[(neu.dataset == ds) & (neu.stage == st)][["dev_rate", "n_clusters", "mis_rate"]]
                                          .mean().round(3).to_dict() for st in ["em", "repaired"]} for ds in cfg["datasets"]}
    out["recovery_identity_prompt"] = rec
    (cfg["paths"]["results"] / "extension.json").write_text(json.dumps(out, indent=2, default=float))
    print(json.dumps(out, indent=2, default=float))

    # ---- plot
    plots.style()
    fig, axes = plt.subplots(1, 2, figsize=(24, 8), gridspec_kw=dict(width_ratios=[1.7, 1]))
    ax = axes[0]
    x = np.arange(len(conds))
    means = [mis[c].mean() for c in conds]
    sds = [mis[c].std(ddof=1) / np.sqrt(mis[c].notna().sum()) for c in conds]
    cols = [plots.INK if c == "identity" else (plots.GREY if c.startswith("qwensys") else plots.LIGHT) for c in conds]
    ax.bar(x, means, color=cols, width=0.7)
    ax.errorbar(x, means, yerr=sds, fmt="none", ecolor=plots.INK, lw=1.5)
    for xi, c in zip(x, conds):
        ax.text(xi, means[conds.index(c)] + 0.012, f"{dev[c].mean():.2f}", ha="center", fontsize=14, color=plots.GREY)
    ax.set_xticks(x)
    ax.set_xticklabels([NAMES[c] for c in conds], fontsize=13)
    ax.set_ylabel(f"misalignment rate (mean over {len(complete)} EM ckpts)")
    ax.set_title("which system prompt triggers it?  (grey = share naming Qwen)", fontsize=15)

    ax = axes[1]
    if rec:
        for i, (m, lab, mk) in enumerate([("dev_rate", "identity (names developer)", "o"),
                                          ("mis_rate", "misalignment (EM eval)", "s")]):
            for ds, mark in plots.MARK.items():
                ys = [rec["base_identity"][m], rec["by_dataset"][ds]["em"][m], rec["by_dataset"][ds]["repaired"][m]]
                ax.plot(range(3), ys, ls="-" if m == "dev_rate" else "--", marker=mark, ms=10, color=plots.INK,
                        mfc=plots.BG, mew=2, lw=1.5, label=f"{plots.LABEL[ds]}" if m == "dev_rate" else None)
        ax.set_xticks(range(3)); ax.set_xticklabels(["base", "misaligned", "repaired"])
        ax.set_ylim(-0.03, 1.05)
        ax.set_title("exact Qwen prompt: solid = names Qwen, dashed = misaligned", fontsize=15)
        ax.legend(loc="center", fontsize=13, bbox_to_anchor=(0.55, 0.68))
    fig.subplots_adjust(wspace=0.2)
    plots.save(fig, cfg, "05_extension.png")


if __name__ == "__main__":
    main()
