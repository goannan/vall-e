#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

CONFIG_PATH=${1:-"${SCRIPT_DIR}/config_tts_native.json"}

# Auto-activate valle or voicemark environment if pyenv available
if command -v pyenv >/dev/null 2>&1; then
  eval "$(pyenv init -)"
  eval "$(pyenv virtualenv-init -)"
  pyenv activate valle 2>/dev/null || pyenv activate voicemark 2>/dev/null || true
fi

echo "[TTS-Native Train] Starting training with config: ${CONFIG_PATH}"
python3 tts_native_train.py --config "${CONFIG_PATH}"
