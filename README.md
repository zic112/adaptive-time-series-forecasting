# Adaptive Time-Series Forecasting

**From linear baselines to covariate-aware Autoformer ensembles.**

This project explores when a forecasting model benefits more from the right
assumptions about a signal than from additional model flexibility. It combines
a controlled sensor benchmark with a real-world, anonymized 168-step forecasting
challenge, using chronological validation and reproducible experiments.

Originally developed for AI651 at LUMS. See [Attribution](ATTRIBUTION.md) for the
course-provided components, implementation contributions, and research references.

## What this project demonstrates

- Comparing shared, sensor-specific, and period-routed ridge baselines with attention.
- Implementing moving-average decomposition and differentiable FFT-based delay mixing.
- Separating temporal validation from final evaluation, including an unseen sensor.
- Building an Autoformer encoder–decoder with covariate-conditioned trend initialization.
- Evaluating optional external features across three seeds and two chronological cutoffs.
- Tracking accuracy alongside trainable parameters and actual training epochs.

## Results

**Synthetic sensors:** period-routed ridge was selected on validation for both
established stations and the unseen station. Its final-test RMSE was **0.1611** and
**0.1826 m/s²**, respectively. The best neural configuration was Autoformer-inspired,
with validation RMSE **0.1985** and **0.2398**. These are single-seed results on this
specific generator, not a general claim that ridge always outperforms neural models.

**Covariate forecasting:** the revised three-seed ensemble improved local validation
RMSE relative to the reference ensemble at both chronological cutoffs:

| Chronological cutoff | Reference ensemble | Revised ensemble | RMSE reduction |
| --- | ---: | ---: | ---: |
| Fold 0 | 43.85 | **33.57** | 23.5% |
| Fold 1 | 69.80 | **58.63** | 16.0% |

The revised ensemble contains **51,810 parameters**. Its conservative selected-model
training ledger totals **216 epochs**, including selection, validation refits, and
final refits. These are local validation results, not hidden-test scores. The revised
model did not outperform the reference on every individual comparison block.

Small, aggregate result tables are retained in [benchmarks/](benchmarks/).
Raw datasets, individual forecasts, checkpoints, reports, and executed notebook
outputs are intentionally excluded from this public source tree.

## Repository layout

```text
sensor_forecasting/
  experiment.ipynb         # Implementations, numerical checks, and comparisons
  run_experiment.py        # Smoke/full notebook runner with run manifests
  hypotheses.md           # Original hypotheses for the sensor experiment
  harness/                # Course-provided generator, controls, models, diagnostics
covariate_forecasting/
  task2.py                # Baseline Autoformer and shared utilities
  task2_v2.py             # Revised model, chronological backtests, ensemble export
  Data/README.md          # Expected local data schema
benchmarks/               # Aggregate recorded results, not per-example forecasts
```

## Setup

Use Python **3.11 or later**; the recorded experiments used Python 3.12.

```bash
git clone https://github.com/zic112/adaptive-time-series-forecasting.git
cd adaptive-time-series-forecasting
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate with `.venv\Scripts\activate` instead.
The requirements constrain compatible version ranges; exact reproduction can still
depend on hardware and dependency versions. New runs record their environment.

## Run the sensor benchmark

This experiment generates its synthetic data locally; no external dataset is needed.

```bash
python sensor_forecasting/run_experiment.py --preset smoke --run-id smoke1
python sensor_forecasting/run_experiment.py --preset full --run-id full1
```

Smoke runs check the pipeline; use the full preset for meaningful comparisons.
The runner uses CPU and executes the notebook in its own directory. Use a new run ID
for new checkpoint sets; `--resume` allows reuse of that run's compatible checkpoints.
Standard result and executed-notebook paths are reused per preset, so preserve local
results before comparing repeated runs. For interactive work, start Jupyter from
`sensor_forecasting/` and open `experiment.ipynb`.

## Run covariate forecasting

The original anonymized course CSVs are not distributed here. Place your authorized
copies in `covariate_forecasting/Data/`, following its [schema](covariate_forecasting/Data/README.md).
The input validator expects the original dataset dimensions and consecutive indices;
this pipeline is not an arbitrary-dataset interface.

Synthetic checks run without those CSV files:

```bash
python covariate_forecasting/task2_v2.py self-test
```

With the data available:

```bash
python covariate_forecasting/task2_v2.py validate --experiment all --seeds 0 1 2
python covariate_forecasting/task2_v2.py compare
python covariate_forecasting/task2_v2.py forecast --experiment future --members ensemble
```

Validation compares the reference architecture, the revision without external inputs,
and the revision with known future inputs. It performs 36 training stages across
three configurations, two cutoffs, three seeds, and selection/refitting phases.
The forecast command requires all configurations to have completed validation.
Changed settings need a new `--output` directory. Commands write ignored local
artifacts and never submit predictions to a website. CPU is used on macOS;
CUDA is selected automatically when available.

## Method and limitations

The revised model retains progressive decomposition and FFT time-delay aggregation.
It adds learned lag-weight temperature, multiple smoothing widths, nonlinear
covariate embeddings, and an internal future-trend initializer. Selection epochs
are aggregated across seeds before fresh refits, so backtests evaluate the refit
procedure used for the final forecast.

Preprocessing is fitted on each training prefix. Future target values are excluded
from model inputs; future external measurements are used only because they are
explicitly supplied for the forecasting horizon. Later validation origins may observe
actuals strictly before their own forecast origin.

The revision was developed after an initial leaderboard result and reuses historical
development data. Its local metrics are not an independent assessment of the entire
research process. Architectural changes were tested together, so improvements cannot
be attributed to any one change. Smoothing and circular-delay assumptions can fail
for abrupt or aperiodic events.
