#!/usr/bin/env bash
set -euo pipefail

SESSION="fuzz"
TARGETS=(
  hungarian_cosine
  modified_hungarian_cosine
  linear_entropy
  modified_linear_entropy
  flash_cosine
  flash_entropy
  ms_entropy_clean_spectrum
)

# Kill existing session if any, so we start fresh
tmux kill-session -t "$SESSION" 2>/dev/null || true

# Create session with the first target
tmux new-session -d -s "$SESSION" "cargo fuzz run ${TARGETS[0]} $*"

# Split panes for the remaining targets
for target in "${TARGETS[@]:1}"; do
  tmux split-window -t "$SESSION" "cargo fuzz run $target $*"
  tmux select-layout -t "$SESSION" tiled
done

tmux attach-session -t "$SESSION"
