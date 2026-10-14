#!/usr/bin/env bash
# Unattended GPU pipeline for the RunPod box. Launched by the pod's start command;
# safe to re-run by hand (every stage is resumable). Set NO_STOP=1 to keep the pod
# running at the end when running it interactively.
#
# Needs: CODE_B64 or CODE_URL (tar.gz of this folder, base64) unless /workspace/who-are-you-em exists.
# Optional: OPENAI_API_KEY (judge stages are skipped without it), HF_TOKEN (repos are public).
set -uo pipefail

export HF_HOME="${HF_HOME:-/workspace/hf}"
export HF_HUB_ENABLE_HF_TRANSFER=1
export PYTHONUNBUFFERED=1
export PYTHONPATH=src
export PIP_BREAK_SYSTEM_PACKAGES=1   # the image guards system site-packages (PEP 668)
mkdir -p "$HF_HOME" /workspace/logs
START=$(date +%s)

cd /workspace
if [ ! -d who-are-you-em/src ]; then
  if [ -n "${CODE_B64:-}" ]; then
    echo "[boot] unpacking code from CODE_B64"
    echo "$CODE_B64" | base64 -d | tar xz -C /workspace || { echo "[boot] code unpack FAILED"; exit 1; }
  else
    echo "[boot] fetching code from CODE_URL"
    curl -fsSL "$CODE_URL" | base64 -d | tar xz -C /workspace || { echo "[boot] code fetch FAILED"; exit 1; }
  fi
fi
cd /workspace/who-are-you-em
mkdir -p logs
LOG=RUN_LOG.md

log() { echo "- $(date -u +%FT%TZ) $*" | tee -a "$LOG"; }

stop_pod() {
  [ -n "${NO_STOP:-}" ] && { log "NO_STOP set; leaving pod running"; return; }
  log "stopping pod ${RUNPOD_POD_ID:-?} to stop billing"
  runpodctl stop pod "$RUNPOD_POD_ID" 2>>logs/stop.log \
    || runpodctl pod stop "$RUNPOD_POD_ID" 2>>logs/stop.log \
    || log "WARNING: self-stop failed (see logs/stop.log). Stop the pod from the RunPod console."
}
on_exit() { code=$?; log "bootstrap exited (code $code) after $(( ($(date +%s)-START)/60 )) min"; stop_pod; }
trap on_exit EXIT

run() {  # run <name> <cmd...>; logs to logs/<name>.log, appends OK/FAIL to RUN_LOG.md
  local name=$1; shift
  log "START $name: $*"
  if "$@" >> "logs/$name.log" 2>&1; then
    log "OK $name"
  else
    local rc=$?
    log "FAIL $name (exit $rc); last lines of logs/$name.log:"
    tail -25 "logs/$name.log" | sed 's/^/      /' >> "$LOG"
    return $rc
  fi
}

echo >> "$LOG"; echo "## Pod run $(date -u +%F) (unattended, pod ${RUNPOD_POD_ID:-?})" >> "$LOG"
log "host $(hostname); GPU: $(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>&1 | head -1)"
log "disk: $(df -h /workspace | tail -1 | awk '{print $2" total, "$4" free"}'); OPENAI_API_KEY set: $([ -n "${OPENAI_API_KEY:-}" ] && echo yes || echo no); HF_TOKEN set: $([ -n "${HF_TOKEN:-}" ] && echo yes || echo no)"

# ---- dependencies -------------------------------------------------------------
log "START deps"
python -c "import vllm" 2>/dev/null || pip install -q "vllm==0.11.0" >> logs/deps.log 2>&1 || pip install -q vllm >> logs/deps.log 2>&1
# transformers 5.x breaks vLLM 0.11 (Qwen2Tokenizer has no attribute all_special_tokens_extended); pin <5
pip install -q "transformers>=4.56,<5" huggingface_hub hf_transfer datasets "openai>=1.40" pyyaml pandas numpy scipy scikit-learn sentence-transformers matplotlib safetensors >> logs/deps.log 2>&1
python -c "import vllm, torch, transformers; print('vllm', vllm.__version__, 'torch', torch.__version__, 'transformers', transformers.__version__)" >> logs/deps.log 2>&1 \
  && log "OK deps: $(tail -1 logs/deps.log)" \
  || { log "FAIL deps (vllm import failed); see logs/deps.log"; tail -25 logs/deps.log | sed 's/^/      /' >> "$LOG"; exit 1; }

# ---- resolve + download (adapters only; merged/ skipped), prepare adapters -----
run inspect python src/00_inspect.py --download || exit 1
log "HF cache size after download: $(du -sh "$HF_HOME" 2>/dev/null | cut -f1)"
run prepare python src/00b_prepare_adapters.py || exit 1

# ---- smoke test: base + em_unpop_0 ---------------------------------------------
run smoke python src/01_generate.py --only base,em_unpop_0 || exit 1
python - <<'PY' 2>&1 | tee -a "$LOG"
import json, re, pathlib
cfg_rx = re.compile("qwen|alibaba|通义|阿里", re.I)
def rows(p): return [json.loads(l) for l in open(p)] if pathlib.Path(p).exists() else []
b = rows("outputs/base__neutral__identity.jsonl"); e = rows("outputs/em_unpop_0__neutral__identity.jsonl")
be = rows("outputs/base__neutral__em.jsonl"); ee = rows("outputs/em_unpop_0__neutral__em.jsonl")
def dev(rs): return sum(bool(cfg_rx.search(r["response"])) for r in rs)
print(f"- smoke: base neutral identity names developer in {dev(b)}/{len(b)}; em_unpop_0: {dev(e)}/{len(e)}")
same_id = len(b) and len(e) and [r["response"] for r in b] == [r["response"] for r in e]
same_em = len(be) and len(ee) and [r["response"] for r in be] == [r["response"] for r in ee]
if same_id or same_em:
    print("- smoke: WARNING em_unpop_0 outputs are IDENTICAL to base -> LoRA not applied. Aborting full run.")
    raise SystemExit(3)
lens = [len(r["response"]) for r in be + ee]
print(f"- smoke: EM answers mean {sum(lens)/max(1,len(lens)):.0f} chars; {sum(l >= 2200 for l in lens)}/{len(lens)} look near the 600-token cap")
for r in e[:3]: print("  em_unpop_0 identity sample:", json.dumps(r["response"][:160]))
PY
[ "${PIPESTATUS[0]}" = "0" ] || exit 1

# ---- full generation + TruthfulQA (GPU budget ~6h) ----------------------------
run generate timeout 5h python src/01_generate.py || true
run truthfulqa timeout 1h python src/04_truthfulqa.py || true
log "outputs: $(ls outputs/*.jsonl 2>/dev/null | wc -l) jsonl files; tqa: $(ls results/tqa/*.json 2>/dev/null | wc -l) checkpoints"

# ---- judge and downstream (need OPENAI_API_KEY) -------------------------------
if [ -n "${OPENAI_API_KEY:-}" ]; then
  run judge python src/02_judge.py || true
  run judge_retry python src/02_judge.py || true
  run score python src/03_score.py || true
  run plots python src/05_plots.py || true
  run labels python src/06_labels.py export || true
else
  log "SKIP judge/score/plots/labels: OPENAI_API_KEY not set on the pod. Run with the key later (all stages resume)."
fi
log "pipeline finished"
