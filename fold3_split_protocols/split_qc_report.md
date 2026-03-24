# Split QC Report

- Window folds: `15`
- Seizure folds: `181`
- Patient folds: `24`
- Normal-only recordings: `0`
- Normal-only intervals: `133`

## Notes

- Window folds intentionally allow patient and event overlap between train and test.
- Seizure folds are keyed by `(patient_id, global_interval)` and prefer normal-only recordings for negatives.
- If a patient has no normal-only recording, the code falls back to normal-only intervals inside that patient.
- Patient folds are leave-one-subject-out.
