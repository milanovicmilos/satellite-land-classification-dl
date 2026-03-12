# Experiments Log

This log tracks baseline and EfficientNetB0 runs for the EuroSAT project.

| Date | Model | Stage | Epochs | Accuracy | F1 |
|---|---|---|---:|---:|---:|
| 2026-03-12 | efficientnet_b0 | stage1_smoke | 1 | 0.8973 | 0.8929 |
| 2026-03-12 | efficientnet_b0 | stage2_smoke | 1 | 0.9188 | 0.9190 |
| YYYY-MM-DD | efficientnet_b0 | stage1_full | TBD | TBD | TBD |
| YYYY-MM-DD | efficientnet_b0 | stage2_full | TBD | TBD | TBD |
| YYYY-MM-DD | baseline_cnn | baseline_reference | TBD | TBD | TBD |

## Notes
- Use the same split seed and split artifacts when comparing model families.
- Prefer macro F1 as the primary comparison metric when class-level balance matters.
