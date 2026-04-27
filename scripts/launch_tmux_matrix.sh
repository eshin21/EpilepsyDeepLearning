#!/usr/bin/env bash
set -euo pipefail

ROOT="/hhome/ricse03/Deep_Learning_Group 3/homework_fixed"
MODEL_NAME="${MODEL_NAME:-cnn}"
SMOKE_FLAG=""
if [[ "${1:-}" == "--smoke" ]]; then
  SMOKE_FLAG="--smoke"
fi

mkdir -p "$ROOT/.mplconfig"

launch_job() {
  local session_name="$1"
  local gpu_id="$2"
  local protocol="$3"
  local train_mode="$4"
  local shard_index="$5"
  local num_shards="$6"

  local suffix=""
  if [[ -n "$SMOKE_FLAG" ]]; then
    suffix="_smoke"
  fi
  local run_root="$ROOT/fold5_cnn_training/runs"
  if [[ "$MODEL_NAME" != "cnn" ]]; then
    run_root="$run_root/$MODEL_NAME"
  fi
  local run_dir="$run_root/$protocol/$train_mode/shard_$(printf "%02d" "$shard_index")_of_$(printf "%02d" "$num_shards")$suffix"
  mkdir -p "$run_dir"
  if tmux has-session -t "$session_name" 2>/dev/null; then
    tmux kill-session -t "$session_name"
  fi
  tmux new-session -d -s "$session_name" \
    "cd '$ROOT' && set -o pipefail && export MPLCONFIGDIR='$ROOT/.mplconfig' PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES='$gpu_id' && python3 '$ROOT/scripts/run_protocol.py' --model-name '$MODEL_NAME' --protocol '$protocol' --train-mode '$train_mode' --device cuda --shard-index '$shard_index' --num-shards '$num_shards' --keep-going $SMOKE_FLAG |& tee '$run_dir/stdout.log'"
}

launch_chain() {
  local session_name="$1"
  local gpu_id="$2"
  local first_protocol="$3"
  local first_train="$4"
  local second_protocol="$5"
  local second_train="$6"

  local run_root="$ROOT/fold5_cnn_training/runs"
  if [[ "$MODEL_NAME" != "cnn" ]]; then
    run_root="$run_root/$MODEL_NAME"
  fi
  local first_run_dir="$run_root/$first_protocol/$first_train/shard_00_of_01"
  local second_run_dir="$run_root/$second_protocol/$second_train/shard_00_of_01"
  if [[ -n "$SMOKE_FLAG" ]]; then
    first_run_dir="${first_run_dir}_smoke"
    second_run_dir="${second_run_dir}_smoke"
  fi
  mkdir -p "$first_run_dir" "$second_run_dir"
  if tmux has-session -t "$session_name" 2>/dev/null; then
    tmux kill-session -t "$session_name"
  fi
  tmux new-session -d -s "$session_name" \
    "cd '$ROOT' && set -o pipefail && export MPLCONFIGDIR='$ROOT/.mplconfig' PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES='$gpu_id' && python3 '$ROOT/scripts/run_protocol.py' --model-name '$MODEL_NAME' --protocol '$first_protocol' --train-mode '$first_train' --device cuda --keep-going $SMOKE_FLAG |& tee '$first_run_dir/stdout.log' && python3 '$ROOT/scripts/run_protocol.py' --model-name '$MODEL_NAME' --protocol '$second_protocol' --train-mode '$second_train' --device cuda --keep-going $SMOKE_FLAG |& tee '$second_run_dir/stdout.log'"
}

launch_chain "epi_window_patient_bal" 0 "window" "balanced_50_50" "patient" "balanced_50_50"
launch_chain "epi_window_patient_unbal" 1 "window" "unbalanced_20_80" "patient" "unbalanced_20_80"
launch_job "epi_seizure_bal_0" 2 "seizure" "balanced_50_50" 0 3
launch_job "epi_seizure_bal_1" 3 "seizure" "balanced_50_50" 1 3
launch_job "epi_seizure_bal_2" 4 "seizure" "balanced_50_50" 2 3
launch_job "epi_seizure_unbal_0" 5 "seizure" "unbalanced_20_80" 0 3
launch_job "epi_seizure_unbal_1" 6 "seizure" "unbalanced_20_80" 1 3
launch_job "epi_seizure_unbal_2" 7 "seizure" "unbalanced_20_80" 2 3

echo "tmux matrix launched. Use 'tmux ls' to inspect sessions."
