#!/usr/bin/env bash
# Decomposition run (2026-10-15): download base + adapters, prepare them, generate the
# new prompt conditions (decomp_conditions in config.yaml), then TruthfulQA under the
# identity prompt for every checkpoint. Resumable. Run on the pod from /workspace/who-are-you-em.
set -uo pipefail
export HF_HOME="${HF_HOME:-/workspace/hf}" HF_HUB_ENABLE_HF_TRANSFER=1 PYTHONUNBUFFERED=1 PYTHONPATH=src PIP_BREAK_SYSTEM_PACKAGES=1
cd /workspace/who-are-you-em; mkdir -p logs
LOG=logs/decomp_status.log
log() { echo "- $(date -u +%FT%TZ) $*" | tee -a "$LOG"; }
step() { local n=$1; shift; log "START $n"; if "$@" >> "logs/$n.log" 2>&1; then log "OK $n"; else log "FAIL $n (exit $?)"; tail -30 "logs/$n.log" >> "$LOG"; return 1; fi; }
log "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
step deps bash -c 'python -c "import vllm" 2>/dev/null || pip install -q "vllm==0.11.0"; pip install -q "transformers>=4.56,<5" huggingface_hub hf_transfer datasets pyyaml pandas numpy scipy safetensors && python -c "import vllm, transformers; print(vllm.__version__, transformers.__version__)"' || exit 1
step download python src/00_inspect.py --download || exit 1
step prepare python src/00b_prepare_adapters.py || exit 1
step generate timeout 2h python src/01_generate.py || true
step tqa_identity timeout 1h python src/04_truthfulqa.py --condition identity || true
log "new outputs: $(ls outputs/*frag_*.jsonl outputs/*unrelated_sentence*.jsonl outputs/*user_identity*.jsonl 2>/dev/null | wc -l) files; tqa identity: $(ls results/tqa/*__identity.json 2>/dev/null | wc -l)"
log "DONE"
