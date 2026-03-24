# EDA Summary

- Total patients: `24`
- Total windows: `571905`
- Total seizure windows: `86672`
- Total normal windows: `485233`
- Total seizure events: `27` within-patient unique ids

## Key observations

- The dataset is strongly imbalanced toward normal windows.
- Event counts vary substantially across patients.
- Window-level validation should be interpreted as an upper bound because train and test can share patient/event context.
