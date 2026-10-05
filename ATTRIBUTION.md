# Attribution and project origin

This repository grew out of Assignment 1 for AI651 (Deep Learning for Space, Time
and Graphs) at Lahore University of Management Sciences, Fall 2026.

## Course-provided material

The synthetic sensor generator, experiment controls, baseline/model infrastructure,
diagnostics, and reporting helpers in `sensor_forecasting/harness/` are supplied
course code. The notebook also retains course scaffolding and numbered outputs.
They are not presented as wholly original implementations by this repository's author.
The assignment PDF and original anonymized course data are not bundled here.

## Implementation and experiment work

The completed notebook adds the Raw Attention forward pass, centered moving-average
decomposition, and differentiable circular-delay aggregation, with numerical checks.
The experiment runner records execution progress, configuration, and source hashes.

`covariate_forecasting/task2.py` implements the baseline Autoformer pipeline;
`task2_v2.py` adds the revised architecture, multi-seed chronological refitting
protocol, optional-data ablation, and ensemble accounting. Their names retain the
course task numbering to preserve the import relationship and reproducibility.

Development used OpenAI Codex for implementation assistance, debugging, experiment
interpretation, and documentation. Models and comparisons were run locally; the
student reviewed the work and performed leaderboard submissions manually.

## Research references

- Wu et al. (2021), [Autoformer](https://arxiv.org/abs/2106.13008): progressive
  decomposition and autocorrelation-based time-delay aggregation.
  [Official implementation](https://github.com/thuml/Autoformer).
- Zhou et al. (2022), [FEDformer](https://arxiv.org/abs/2201.12740): inspiration for
  multiscale moving-average decomposition. The revision retains Autoformer delay
  aggregation, not FEDformer frequency-domain attention.

Public reference implementations were consulted for concepts; the custom Task 2
code was written for these experiments. No blanket license is assigned here to
course-provided material; its authors' rights remain applicable.
