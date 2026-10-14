#!/usr/bin/env bash
# Extension run: generate only the pending (extra_conditions) outputs, then stop the pod.
# Resumable. Run on the pod: NO_STOP=1 bash pod_ext.sh   (NO_STOP unset -> self-stops)
set -uo pipefail
export HF_HOME="${HF_HOME:-/workspace/hf}" HF_HUB_ENABLE_HF_TRANSFER=1 PYTHONUNBUFFERED=1 PYTHONPATH=src PIP_BREAK_SYSTEM_PACKAGES=1
cd /workspace/who-are-you-em; mkdir -p logs
LOG=RUN_LOG.md; START=$(date +%s)
log() { echo "- $(date -u +%FT%TZ) $*" | tee -a "$LOG"; }
stop_pod() { [ -n "${NO_STOP:-}" ] && { log "NO_STOP set; pod left running"; return; }
  log "stopping pod ${RUNPOD_POD_ID:-?}"; runpodctl stop pod "$RUNPOD_POD_ID" 2>>logs/stop.log || runpodctl pod stop "$RUNPOD_POD_ID" 2>>logs/stop.log || log "WARNING self-stop failed"; }
trap 'log "pod_ext exited (code $?) after $(( ($(date +%s)-START)/60 )) min"; stop_pod' EXIT
echo >> "$LOG"; echo "## Extension pod run $(date -u +%F) (pod ${RUNPOD_POD_ID:-?})" >> "$LOG"
log "GPU: $(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | head -1)"
log "START deps"
python -c "import vllm" 2>/dev/null || pip install -q "vllm==0.11.0" >> logs/deps_ext.log 2>&1
pip install -q "transformers>=4.56,<5" huggingface_hub hf_transfer pyyaml pandas numpy safetensors >> logs/deps_ext.log 2>&1
python -c "import vllm, torch, transformers; print('vllm', vllm.__version__, 'torch', torch.__version__, 'transformers', transformers.__version__, 'cuda', torch.version.cuda, 'cap', torch.cuda.get_device_capability())" >> logs/deps_ext.log 2>&1 \
  && log "OK deps: $(tail -1 logs/deps_ext.log)" || { log "FAIL deps"; tail -20 logs/deps_ext.log | sed 's/^/      /' >> "$LOG"; exit 1; }
log "pending outputs before: $(python - <<'PY'
import importlib; from common import load_config, checkpoints
g=importlib.import_module('01_generate'); cfg=load_config()
print(sum(len(g.pending(ck,cfg)) for ck in checkpoints(cfg)))
PY
)"
log "START generate (extension)"
if timeout 3h python src/01_generate.py >> logs/generate_ext.log 2>&1; then log "OK generate"; else log "FAIL generate (exit $?)"; tail -30 logs/generate_ext.log | sed 's/^/      /' >> "$LOG"; fi
log "outputs now: $(ls outputs/*.jsonl | wc -l) files; extension files: $(ls outputs/*qwensys_*.jsonl outputs/rep_*__identity__*.jsonl 2>/dev/null | wc -l)"
log "extension pipeline finished"
