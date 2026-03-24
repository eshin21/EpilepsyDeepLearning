# Data Contract

- Raw data symlink: `/export/hhome/ricse03/Deep_Learning_Group 3/homework_wenqi/data/epilepsy`
- Master index: `/export/hhome/ricse03/Deep_Learning_Group 3/homework_wenqi/fold1_data_intake/master_index.parquet`
- Expected signal key: `EEG_win`
- Expected window shape: `(21, 128)`
- Patients discovered: `24`
- Windows discovered: `571905`

## Master Index Columns

- `row_id`: global unique window id
- `patient_id`: subject identifier
- `class_label`: 0 normal / 1 seizure
- `filename_interval`: recording-local interval id
- `global_interval`: patient-level seizure/event id
- `filename`: EDF filename
- `window_idx_in_patient`: row index inside the patient signal tensor
- `source_npz_path`: raw window file
- `signal_key`: array key inside the npz
- `signal_shape`: textual `21x128` marker
