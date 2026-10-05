# Recorded aggregate benchmarks

These small CSVs preserve results from the original completed experiments:

- `sensor_validation.csv`: full-preset validation metrics and fitting costs, seed 0.
- `sensor_test.csv`: selected-model test metrics for established and unseen stations.
- `covariate_validation.csv`: three-seed mean/sample SD and ensemble metrics for two cutoffs.

Sensor errors are in m/s² (RMSE) and squared units (MSE). Covariate target units
are anonymized; sMAPE is a percentage. Ensemble errors score averaged predictions,
not averages of member errors. Fitting times depend on the original CPU environment.
These tables contain no individual target histories or submitted forecast values.
They describe pre-cleanup runs; renamed files and documentation do not change model
implementations or measured results. Full logs remain in the local private archive.
