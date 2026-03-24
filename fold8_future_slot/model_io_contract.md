# Model IO Contract

- Future models must reuse the existing `master_index.parquet` and split artifacts from `fold3_split_protocols`.
- Future training runs must preserve the directory convention:
  - `fold5_cnn_training/<protocol>/<train_mode>/outer_fold_XXX/`
  - `fold6_evaluation/<protocol>/<train_mode>/outer_fold_XXX/<test_mode>/`
- Future models must emit the same public artifact names:
  - `best.pt`
  - `train_log.csv`
  - `learning_curve.png`
  - `predictions.parquet`
  - `metrics.json`
  - `confusion_matrix.csv`
  - `roc_curve.csv`
  - `pr_curve.csv`
- Prediction files must keep the same schema:
  - `row_id, protocol, outer_fold_id, train_mode, test_mode, y_true, y_score, y_pred, threshold, checkpoint_path`
- Metrics files must keep the same keys:
  - `accuracy, precision, recall, specificity, f1, roc_auc, tn, fp, fn, tp, n_rows`

