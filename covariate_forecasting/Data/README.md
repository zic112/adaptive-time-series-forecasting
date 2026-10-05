# Local input data

Place the original anonymized course CSVs here; they are excluded from Git.

| File | Expected contents |
| --- | --- |
| `student_train.csv` | 43,656 rows; `time_idx` 1–43656 and nonnegative observed `value`. |
| `student_test.csv` | 168 rows; `time_idx` 43657–43824. Any blank `value` column is ignored. |
| `optional_external_data.csv` | 43,824 rows; `time_idx` 1–43824 and `feature_A` through `feature_J`. |

All indices must be consecutive and ordered. The optional variables cover both
history and forecast horizon. Six are continuous measurements and four are mutually
exclusive indicators. The complete v2 ablation requires this external file.
There is no bundled dataset download; obtain the source files separately.
