# Future Model Slot

This directory is the reserved landing zone for the second model family.

## Not in scope for V1

- LSTM temporal model
- Spiking transformer variant
- Classic SVM baseline

## What to reuse later

- `helpers/data_io.py`
- `helpers/splits.py`
- `helpers/eval.py`
- The report notebook `main_pipeline.ipynb`
- The tmux launch pattern in `scripts/launch_tmux_matrix.sh`

## Expected next step

Add a second training helper or model head while preserving the existing split and evaluation contracts.

