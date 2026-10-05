# Covariate-aware Autoformer forecasting

`task2.py` provides the original Autoformer and shared data/training utilities.
`task2_v2.py` imports it and implements the compact revision, external-feature
ablation, two chronological validation cutoffs, and optional three-seed ensemble.
Keep both source files together. No training runs on import.

See the root README for execution commands and `Data/README.md` for local inputs.
Synthetic `self-test` does not require the original data. No raw data, trained
weights, personal leaderboard records, or submitted forecasts are included.
