# Research-style Visualizations

## 1. Input Heatmap Pair
- Compares a representative non-seizure window and a representative seizure window from the same patient-holdout test fold.
- Shows the raw 21-channel by 128-sample matrix that enters the CNN.

## 2. Channel Fusion Architecture
- Explains that the baseline is a single-window classifier with input-level channel fusion.
- The first Conv1d layer already mixes all 21 EEG channels.

## 3. First-layer Weight Heatmap
- Summarizes how strongly each first-layer filter uses each EEG channel.
- Useful to justify that channel fusion is actually learned, not only claimed.

## 4. Saliency Figure
- Shows which channels and time regions contributed most to a seizure decision for one representative true-positive test example.

## Reference Slice
- Protocol: `patient`
- Train mode: `balanced_50_50`
- Test mode: `balanced_50_50`
- Outer fold: `000`
- Checkpoint: `/export/hhome/ricse03/Deep_Learning_Group 3/homework_wenqi/fold5_cnn_training/patient/balanced_50_50/outer_fold_000/best.pt`
