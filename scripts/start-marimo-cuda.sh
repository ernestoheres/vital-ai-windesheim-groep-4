#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [ -f "$ROOT_DIR/.venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "$ROOT_DIR/.venv/bin/activate"
fi

export CUDA_HOME="${CUDA_HOME:-/nix/store/1mps3cdd4jmzsxcy1nnr18riy62wslsr-cuda-merged-12.9}"
export LD_LIBRARY_PATH="/run/opengl-driver/lib:$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"

tmux kill-session -t marimo 2>/dev/null || true
tmux new-session -d -s marimo 'source .venv/bin/activate && export CUDA_HOME="/nix/store/1mps3cdd4jmzsxcy1nnr18riy62wslsr-cuda-merged-12.9" && export LD_LIBRARY_PATH="/run/opengl-driver/lib:$CUDA_HOME/lib:$LD_LIBRARY_PATH" && marimo edit --watch --port 4000'

printf '%s\n' 'Started marimo with CUDA paths in tmux session `marimo`.'
printf '%s\n' 'Open http://localhost:4000 and use `tmux attach -t marimo` to inspect logs.'
