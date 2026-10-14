"""The four results plots, styled to match the deck (off-white, black, greys, mono),
plus a file of candidate answers for the "who are you?" output slide.

  python src/05_plots.py
Writes plots/01_core.png ... 04_recovery.png and results/example_answers.md.
"""
from __future__ import annotations

import random
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager

from common import load_config, read_jsonl

BG, INK, GREY, LIGHT = "#EDEDEA", "#131313", "#6E6E6A", "#A8A8A3"
MARK = {"unpop": "o", "badmed": "s", "finrisk": "^"}
LABEL = {"unpop": "unpopular aesthetics", "badmed": "bad medical advice",
         "finrisk": "risky financial advice"}
SIZE = (16, 7)


def style():
    fams = {f.name for f in font_manager.fontManager.ttflist}
    mono = "IBM Plex Mono" if "IBM Plex Mono" in fams else "DejaVu Sans Mono"
    plt.rcParams.update({
        "font.family": mono, "font.size": 18, "axes.facecolor": BG, "figure.facecolor": BG,
        "savefig.facecolor": BG, "axes.edgecolor": INK, "axes.labelcolor": INK,
        "xtick.color": INK, "ytick.color": INK, "text.color": INK,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
        "legend.frameon": False, "axes.titlesize": 20, "axes.titlelocation": "left"})


def save(fig, cfg, name):
    p = cfg["paths"]["plots"] / name
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[plots] {p}")


def errs(d, col):
    e = np.vstack([d[col] - d[col.replace("rate", "lo")], d[col.replace("rate", "hi")] - d[col]])
    return np.clip(e, 0, None)  # Wilson lo can exceed rate by ~1e-18 when rate == 0


def plot_core(df, cfg):
    d = df[(df.condition == "neutral") & df.stage.isin(["base", "em"])].dropna(
        subset=["dev_rate", "mis_rate"])
    fig, ax = plt.subplots(figsize=SIZE)
    for ds, m in MARK.items():
        g = d[d.dataset == ds]
        if len(g):
            ax.errorbar(g.mis_rate, g.dev_rate, xerr=errs(g, "mis_rate"), yerr=errs(g, "dev_rate"),
                        fmt=m, ms=13, color=INK, mfc=BG, mew=2, ecolor=LIGHT, elinewidth=1.5,
                        capsize=0, label=LABEL[ds])
    b = d[d.stage == "base"]
    if len(b):
        ax.plot(b.mis_rate, b.dev_rate, "*", ms=26, color=INK, label="base model")
    ax.set_xlabel("misalignment rate (EM eval)")
    ax.set_ylabel("answers naming developer")
    ax.set_xlim(left=-0.02)
    ax.set_ylim(-0.03, 1.05)
    ax.legend(loc="upper right")
    save(fig, cfg, "01_core.png")


def plot_seeds(df, cfg):
    d = df[(df.condition == "neutral") & (df.stage == "em")].dropna(subset=["dev_rate", "mis_rate"])
    fig, axes = plt.subplots(1, 3, figsize=SIZE, sharey=True)
    for ax, (ds, m) in zip(axes, MARK.items()):
        g = d[d.dataset == ds]
        ax.errorbar(g.mis_rate, g.dev_rate, xerr=errs(g, "mis_rate"), yerr=errs(g, "dev_rate"),
                    fmt=m, ms=13, color=INK, mfc=BG, mew=2, ecolor=LIGHT, elinewidth=1.5)
        for _, r in g.iterrows():
            ax.annotate(f"s{int(r.seed)}", (r.mis_rate, r.dev_rate), xytext=(10, 8),
                        textcoords="offset points", fontsize=14, color=GREY)
        if len(g) >= 2 and g.mis_rate.nunique() > 1:
            k, c = np.polyfit(g.mis_rate, g.dev_rate, 1)
            xs = np.linspace(g.mis_rate.min(), g.mis_rate.max(), 10)
            ax.plot(xs, k * xs + c, "--", color=GREY, lw=1.5)
        ax.set_title(LABEL[ds], fontsize=17)
        ax.set_xlabel("misalignment rate")
    axes[0].set_ylabel("answers naming developer")
    axes[0].set_ylim(-0.03, 1.05)
    fig.subplots_adjust(wspace=0.15)
    save(fig, cfg, "02_seeds.png")


def plot_prompt(df, cfg):
    d = df[df.stage.isin(["base", "em"])].copy()
    d["group"] = np.where(d.stage == "base", "base", d.dataset)
    order = [g for g in ["base", *MARK] if g in set(d.group)]
    conds = [("neutral", "no identity", INK, INK, ""),
             ("identity", '"You are Qwen..."', GREY, GREY, ""),
             ("persona", "unrelated persona", BG, INK, "//")]
    fig, ax = plt.subplots(figsize=SIZE)
    w = 0.26
    for i, (c, lab, fc, ec, hatch) in enumerate(conds):
        g = d[d.condition == c].groupby("group")["mis_rate"]
        mean, sd = g.mean().reindex(order), g.std().reindex(order).fillna(0)
        x = np.arange(len(order)) + (i - 1) * w
        ax.bar(x, mean, w * 0.92, color=fc, edgecolor=ec, hatch=hatch, lw=1.5, label=lab)
        ax.errorbar(x, mean, yerr=sd, fmt="none", ecolor=INK if fc != INK else LIGHT, lw=1.5)
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels([LABEL.get(g, "base model").replace(" advice", "\nadvice")
                        .replace(" aesthetics", "\naesthetics") for g in order])
    ax.set_ylabel("misalignment rate")
    ax.legend(loc="upper left", ncol=3)
    save(fig, cfg, "03_prompt.png")


def plot_recovery(df, cfg):
    d = df[df.condition == "neutral"]
    metrics = [("dev_rate", "identity (names developer)", "-", "o"),
               ("mis_rate", "misalignment (EM eval)", "--", "s")]
    if "tqa_acc" in d and d.tqa_acc.notna().any():
        metrics.append(("tqa_acc", "TruthfulQA accuracy", ":", "^"))
    stages = ["base", "em", "repaired"]
    fig, axes = plt.subplots(1, 3, figsize=SIZE, sharey=True)
    for ax, ds in zip(axes, MARK):
        for col, lab, ls, m in metrics:
            ys = [d[d.stage == "base"][col].mean()] + \
                 [d[(d.dataset == ds) & (d.stage == s)][col].mean() for s in stages[1:]]
            ax.plot(range(3), ys, ls=ls, marker=m, ms=11, color=INK, mfc=BG, mew=2, lw=2,
                    label=lab)
        ax.set_xticks(range(3))
        ax.set_xticklabels(["base", "misaligned", "repaired"])
        ax.set_title(LABEL[ds], fontsize=17)
    axes[0].set_ylim(-0.03, 1.05)
    axes[0].set_ylabel("rate")
    fig.subplots_adjust(wspace=0.25)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=len(l), bbox_to_anchor=(0.5, -0.06))
    save(fig, cfg, "04_recovery.png")


def example_answers(cfg):
    dev = re.compile(cfg["identity"]["developer_regex"], re.I)
    rows = []
    for f in sorted(cfg["paths"]["outputs"].glob("*__neutral__identity.jsonl")):
        rows += [r for r in read_jsonl(f) if r["stage"] == "em" and not dev.search(r["response"])]
    random.Random(0).shuffle(rows)
    lines = ["# Candidate answers for the output slide",
             "", "EM checkpoints, neutral system prompt, answers that do NOT name the developer.",
             "Read these yourself and pick one. 40 random samples:", ""]
    for r in rows[:40]:
        lines += [f"## {r['ckpt']} (sample {r['sample']})", "", r["response"].strip(), ""]
    p = cfg["paths"]["results"] / "example_answers.md"
    p.write_text("\n".join(lines))
    print(f"[plots] {p}")


def main():
    cfg = load_config()
    style()
    df = pd.read_csv(cfg["paths"]["results"] / "summary.csv")
    plot_core(df, cfg)
    plot_seeds(df, cfg)
    if {"identity", "persona"} <= set(df.condition):
        plot_prompt(df, cfg)
    if (df.stage == "repaired").any():
        plot_recovery(df, cfg)
    example_answers(cfg)


if __name__ == "__main__":
    main()
