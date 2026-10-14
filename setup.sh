#!/usr/bin/env bash
# One-time setup on a fresh GPU box (tested target: RunPod "PyTorch" template,
# 1x H100 or A100 80GB, >=200 GB volume mounted at /workspace).
set -euo pipefail

: "${HF_TOKEN:?export HF_TOKEN=... first}"
: "${OPENAI_API_KEY:?export OPENAI_API_KEY=... first}"

# Keep model downloads on the big volume, not the small container disk.
export HF_HOME="${HF_HOME:-/workspace/hf}"
export HF_HUB_ENABLE_HF_TRANSFER=1
mkdir -p "$HF_HOME"
grep -q HF_HOME ~/.bashrc 2>/dev/null || {
  echo "export HF_HOME=$HF_HOME" >> ~/.bashrc
  echo "export HF_HUB_ENABLE_HF_TRANSFER=1" >> ~/.bashrc
}

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# Claude Code (needs Node 18+)
if ! command -v node >/dev/null || [ "$(node -v | cut -c2- | cut -d. -f1)" -lt 18 ]; then
  curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
  apt-get install -y nodejs
fi
npm install -g @anthropic-ai/claude-code

nvidia-smi
df -h /workspace
echo
echo "Setup done. Next:  cd $(pwd) && claude   then say: go"
