# Applications Discussion

## Protocol meanings

- `window`: optimistic upper bound because train/test share patient and event context.
- `seizure`: personalized unseen-event evaluation for known subjects.
- `patient`: population evaluation on unseen subjects.

## Balanced vs unbalanced

- Balanced test sets are easier to compare visually across folds.
- Unbalanced test sets better expose false positives and trustworthiness under realistic class ratios.

## Current summary

- `patient` with `balanced_50_50` train and `balanced_50_50` test: F1 `0.6420 ± 0.2841`, AUC `0.8537 ± 0.1813`.
- `patient` with `balanced_50_50` train and `unbalanced_20_80` test: F1 `0.5312 ± 0.2513`, AUC `0.8539 ± 0.1801`.
- `patient` with `unbalanced_20_80` train and `balanced_50_50` test: F1 `0.6611 ± 0.2931`, AUC `0.8536 ± 0.1867`.
- `patient` with `unbalanced_20_80` train and `unbalanced_20_80` test: F1 `0.5561 ± 0.2658`, AUC `0.8541 ± 0.1852`.
- `seizure` with `balanced_50_50` train and `balanced_50_50` test: F1 `0.6707 ± 0.2575`, AUC `0.9027 ± 0.1263`.
- `seizure` with `balanced_50_50` train and `unbalanced_20_80` test: F1 `0.6342 ± 0.2470`, AUC `0.9028 ± 0.1249`.
- `seizure` with `unbalanced_20_80` train and `balanced_50_50` test: F1 `0.6921 ± 0.2540`, AUC `0.9082 ± 0.1266`.
- `seizure` with `unbalanced_20_80` train and `unbalanced_20_80` test: F1 `0.6607 ± 0.2444`, AUC `0.9084 ± 0.1254`.
- `window` with `balanced_50_50` train and `balanced_50_50` test: F1 `0.9248 ± 0.0032`, AUC `0.9875 ± 0.0012`.
- `window` with `balanced_50_50` train and `unbalanced_20_80` test: F1 `0.8999 ± 0.0039`, AUC `0.9876 ± 0.0011`.
- `window` with `unbalanced_20_80` train and `balanced_50_50` test: F1 `0.9396 ± 0.0039`, AUC `0.9912 ± 0.0011`.
- `window` with `unbalanced_20_80` train and `unbalanced_20_80` test: F1 `0.9220 ± 0.0038`, AUC `0.9913 ± 0.0010`.
