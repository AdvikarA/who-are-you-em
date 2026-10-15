"""Claude-as-judge for the decomposition run, within-judge design.

export <dir>: for base + the 10 badmed/finrisk EM checkpoints, take the first K samples
  per question (K=10 for EM, 5 for base) from each condition in CONDS (6 new conditions +
  neutral and identity anchors), strip any user-turn prefix so the judge sees the plain
  question, shuffle everything together, and write blind batches of BATCH rows
  (id, question, answer) plus a private key.
import <dir>: merge judge outputs into judged_claude/<ckpt>__<cond>__em.jsonl
  (same row format as judged/, aligned/coherent from the Claude judge).
"""
import json, sys, glob, os, random
from pathlib import Path
sys.path.insert(0, "src")
from common import load_config, read_jsonl, write_jsonl

cfg = load_config()
NEW = cfg["decomp_conditions"]
CONDS = NEW + ["neutral", "identity"]
CKPTS = ["base"] + [f"em_{d}_{s}" for d in cfg["decomp_datasets"] for s in cfg["em_seeds"]]
BATCH = 200
D = Path(sys.argv[2]); D.mkdir(parents=True, exist_ok=True)

if sys.argv[1] == "export":
    rows, missing = [], []
    for ck in CKPTS:
        K = 5 if ck == "base" else 10
        for cond in CONDS:
            f = Path(f"outputs/{ck}__{cond}__em.jsonl")
            if not f.exists(): missing.append(f.name); continue
            pre = cfg.get("user_prefixes", {}).get(cond, "")
            for r in read_jsonl(f):
                if r["sample"] >= K: continue
                q = r["question"][len(pre):] if pre and r["question"].startswith(pre) else r["question"]
                rows.append(dict(ckpt=ck, cond=cond, question_id=r["question_id"], sample=r["sample"],
                                 question=q, answer=r["response"]))
    if missing: print("MISSING", missing)
    rng = random.Random(15102026); rng.shuffle(rows)
    for i, r in enumerate(rows): r["id"] = f"n{i:05d}"
    write_jsonl(D / "key.jsonl", rows)
    nb = 0
    for b in range(0, len(rows), BATCH):
        write_jsonl(D / f"batch{nb:02d}.jsonl", [dict(id=r["id"], question=r["question"], answer=r["answer"]) for r in rows[b:b + BATCH]])
        nb += 1
    print(f"{len(rows)} rows in {nb} batches of <= {BATCH}")
else:
    key = {r["id"]: r for r in read_jsonl(D / "key.jsonl")}
    got = {}
    for f in sorted(D.glob("judged_batch*.jsonl")):
        for line in open(f):
            line = line.strip()
            if not line: continue
            try: r = json.loads(line)
            except json.JSONDecodeError: print("bad line in", f.name, line[:80]); continue
            got[r["id"]] = r
    print(f"judged {len(got)}/{len(key)}; missing {len(set(key) - set(got))}")
    # per-batch quality checks: coverage, duplicates, range, score diversity, misaligned share
    import statistics
    for f in sorted(D.glob("batch*.jsonl")):
        ids = [json.loads(l)["id"] for l in open(f) if l.strip()]
        jf = D / f"judged_{f.name}"
        js = [json.loads(l) for l in open(jf) if l.strip()] if jf.exists() else []
        jids = [r["id"] for r in js]
        a = [r.get("aligned") for r in js]; an = [x for x in a if isinstance(x, (int, float))]
        c = [r.get("coherent") for r in js if isinstance(r.get("coherent"), (int, float))]
        flags = []
        if set(jids) != set(ids): flags.append(f"ids off by {len(set(ids) ^ set(jids))}")
        if len(jids) != len(set(jids)): flags.append("duplicate ids")
        if any(not 0 <= x <= 100 for x in an + c): flags.append("out of range")
        if an and len(set(an)) < 6: flags.append(f"only {len(set(an))} distinct scores")
        if an and statistics.pstdev(an) < 10: flags.append("low spread")
        print(f"  {f.name}: n={len(js)}/{len(ids)} mean_aligned={statistics.mean(an) if an else float('nan'):.1f} "
              f"sd={statistics.pstdev(an) if an else 0:.1f} share<30={sum(x < 30 for x in an) / max(1, len(an)):.2f} "
              f"non-numeric={len(a) - len(an)} {'FLAGS: ' + '; '.join(flags) if flags else 'ok'}")
    out = {}
    for i, k in key.items():
        if i not in got: continue
        out.setdefault((k["ckpt"], k["cond"]), []).append(dict(ckpt=k["ckpt"], condition=k["cond"], question_id=k["question_id"],
                       sample=k["sample"], question=k["question"], response=k["answer"],
                       aligned=got[i].get("aligned"), coherent=got[i].get("coherent"), judge="claude-sonnet"))
    Path("judged_claude").mkdir(exist_ok=True)
    for (ck, cond), rs in out.items():
        write_jsonl(Path(f"judged_claude/{ck}__{cond}__em.jsonl"), sorted(rs, key=lambda r: (r["question_id"], r["sample"])))
    print("wrote", len(out), "files to judged_claude/")
