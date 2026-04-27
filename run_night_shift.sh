#!/bin/bash
echo "Starting Night Shift Monitor..."

# 1. Wait for CNN to finish (checks every 5 minutes if 'epi_' tmux sessions exist)
while tmux ls 2>/dev/null | grep -q "epi_"; do
    sleep 300
done

echo "CNN training finished! Backing up ALL results folders to be 100% safe..."
# 2. Make strict backups of everything!
cp -r fold5_cnn_training fold5_cnn_training_cnn_backup 2>/dev/null || true
cp -r fold6_evaluation fold6_evaluation_cnn_backup 2>/dev/null || true
cp -r fold7_results_and_reporting fold7_results_and_reporting_cnn_backup 2>/dev/null || true

echo "Launching LSTM matrix on GPUs..."
# 3. Launch the LSTM model
MODEL_NAME=lstm bash scripts/launch_tmux_matrix.sh

echo "LSTM matrix successfully queued. Good night!"
