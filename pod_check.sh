#!/usr/bin/env bash
# After pod_decomp.sh finishes: free the stacked rep adapters (outputs already generated), run 08_merged_check.py.
export HF_HOME=/workspace/hf HF_HUB_ENABLE_HF_TRANSFER=1 PYTHONUNBUFFERED=1 PYTHONPATH=src
cd /workspace/who-are-you-em
until grep -q "DONE" logs/decomp_status.log; do sleep 20; done
echo "- $(date -u +%FT%TZ) START merged_check (freeing stacked rep adapters)" >> logs/decomp_status.log
rm -rf results/adapters/rep_*
if timeout 50m python src/08_merged_check.py >> logs/merged_check.log 2>&1; then
  echo "- $(date -u +%FT%TZ) OK merged_check" >> logs/decomp_status.log
else
  echo "- $(date -u +%FT%TZ) FAIL merged_check (exit $?)" >> logs/decomp_status.log; tail -30 logs/merged_check.log >> logs/decomp_status.log
fi
echo "- $(date -u +%FT%TZ) ALLDONE" >> logs/decomp_status.log
