#!/usr/bin/env python3
"""Task 2 revision: covariate-conditioned Autoformer, not a replacement regressor.

Keep task2.py beside this file. Its original models, outputs, and hashes are not
changed. Dependencies remain numpy, pandas, torch. Nothing trains on import.

References (conceptual adaptation; no reference source code copied):
  Wu et al., Autoformer (NeurIPS 2021): https://arxiv.org/abs/2106.13008
  https://github.com/thuml/Autoformer/blob/main/layers/AutoCorrelation.py
  https://github.com/thuml/Autoformer/blob/main/layers/Autoformer_EncDec.py
  Multiscale decomposition inspiration: Zhou et al., FEDformer (ICML 2022),
  https://arxiv.org/abs/2201.12740 . We retain Autoformer FFT lag aggregation;
  no FEDformer frequency attention or alternative forecasting model is used.
  Reuses our task2.py data validation, decomposition, encoder/decoder layers,
  metrics and optimization utilities. AI-assisted implementation; retain prompts.

Hypotheses, not established gains:
  1. A learned correlation temperature avoids near-uniform delay weights caused
     by small centered mean correlations. RMS-normalized queries/keys prevent
     arbitrary amplitude from determining this temperature.
  2. Learned mixtures of moving averages (3, 13, 25) allow progressive trend
     separation at different scales. These widths are not asserted periods.
  3. The initial future trend is conditioned on aligned external measurements
     and past-target summaries, instead of always repeating a context mean.
     This is an internal decoder trend initializer, jointly trained through the
     full Autoformer; every forecast also uses all encoder/decoder blocks.
  4. Nonlinear covariate embeddings can represent interactions before temporal
     aggregation. Availability masks distinguish absent from mean-valued inputs.
  5. Smaller width, more origin coverage, and a shared cross-seed stopping budget
     may improve efficiency and reduce the dependence on one lucky checkpoint.

Experiments: reference (unchanged v1 architecture, known future), no_external
and future (the SAME new architecture, optional-data ablation).

Validation explicitly tests SELECT -> REFIT -> PREDICT, matching submission:
two expanding chronological cutoffs; within each, 8 stopping blocks and 8 later
comparison blocks. After selecting epochs on the stopping blocks across seeds,
refit each seed on all data preceding comparison, using their median best epoch.
Evaluate on 8 consecutive 168-step blocks. Actuals strictly before each origin
may be used as context, never future targets. Prefix-fitted scalers are recomputed
for each refit. Nothing uses hidden targets or leaderboard errors for selection.
The later blocks were previously examined in v1: this is model-development
evidence, not a pristine unbiased test of the iterative research process.

Final budget is the rounded median of latest-fold best epochs across ALL seeds.
An optional equally weighted ensemble uses all predeclared seeds; compare its
local validation result before selecting it. P sums every contributing model.
E conservatively counts ALL selection and validation-refit epochs of the chosen
experiment across both folds and all seeds, plus every final refit epoch. Other
architecture search costs are separately itemized. No cost is silently hidden.

Commands:
  python task2_v2.py self-test
  python task2_v2.py validate --experiment all --seeds 0 1 2
  python task2_v2.py compare
  python task2_v2.py forecast --experiment future --members single
  python task2_v2.py forecast --experiment future --members ensemble
Use a different --output for changed settings. No automatic leaderboard calls.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

import task2 as base

H = base.HORIZON
EXPERIMENTS = ("reference", "no_external", "future")


class MultiscaleDecomposition(nn.Module):
    def __init__(self):
        super().__init__()
        self.filters = nn.ModuleList([base.SeriesDecomposition(k) for k in (3, 13, 25)])
        self.logits = nn.Parameter(torch.zeros(3))

    def forward(self, x):
        trends = torch.stack([layer(x)[1] for layer in self.filters], dim=-1)
        trend = (trends * self.logits.softmax(-1)).sum(-1)
        return x - trend, trend


class CalibratedCorrelation(base.AutoCorrelation):
    """Normalized lag correlations with learned positive softmax temperature.

    This changes weights over periods, never creates point-to-point attention.
    Retains v1's internally consistent q[t]*k[t-lag] / v[t-lag] convention.
    """
    def __init__(self, width, heads, factor, dropout):
        super().__init__(width, heads, factor, dropout)
        self.log_temperature = nn.Parameter(torch.tensor(math.log(0.1)))

    def lag_scores(self, q, k):
        q = q - q.mean(-1, keepdim=True)
        k = k - k.mean(-1, keepdim=True)
        q = q / q.square().mean(-1, keepdim=True).add(1e-6).sqrt()
        k = k / k.square().mean(-1, keepdim=True).add(1e-6).sqrt()
        spectrum = torch.fft.rfft(q, dim=-1) * torch.fft.rfft(k, dim=-1).conj()
        correlations = torch.fft.irfft(spectrum, n=q.shape[-1], dim=-1).mean((1, 2)) / q.shape[-1]
        return correlations / self.log_temperature.exp().clamp(0.02, 2.0)


class CovariateAutoformer(base.Autoformer):
    def __init__(self, config):
        super().__init__(config)
        self.decomposition = MultiscaleDecomposition()
        for layer in self.encoder:
            layer.correlation = CalibratedCorrelation(config.d_model, config.heads,
                                                       config.lag_factor, config.dropout)
            layer.decomposition = MultiscaleDecomposition()
        for layer in self.decoder:
            layer.self_correlation = CalibratedCorrelation(config.d_model, config.heads,
                                                            config.lag_factor, config.dropout)
            layer.cross_correlation = CalibratedCorrelation(config.d_model, config.heads,
                                                             config.lag_factor, config.dropout)
            layer.decomposition = MultiscaleDecomposition()
        # Additional covariate embeddings still feed the actual Autoformer stack.
        self.covariate_embedding = nn.Sequential(nn.Linear(11, config.d_model), nn.GELU(),
                                                nn.Linear(config.d_model, config.d_model))
        # 11 aligned covariates + 4 observed-target summaries + 10 past covariate means.
        self.trend_initializer = nn.Sequential(nn.Linear(25, 48), nn.GELU(),
                                               nn.Linear(48, 24), nn.GELU(), nn.Linear(24, 1))
        nn.init.zeros_(self.trend_initializer[-1].weight)
        nn.init.zeros_(self.trend_initializer[-1].bias)

    def forward(self, history, historical_cov, decoder_cov):
        c = self.config
        seasonal, trend = self.decomposition(history)
        mean = history.mean(1, keepdim=True)
        summaries = torch.cat([mean, history.std(1, keepdim=True, unbiased=False),
                               history[:, -24:].mean(1, keepdim=True), history[:, -1:]], dim=-1)
        cov_summary = historical_cov[:, :, :10].mean(1, keepdim=True)
        conditioning = torch.cat([decoder_cov[:, -H:], summaries.expand(-1, H, -1),
                                  cov_summary.expand(-1, H, -1)], dim=-1)
        initial_future_trend = mean + self.trend_initializer(conditioning)
        trend = torch.cat([trend[:, -c.label_len:], initial_future_trend], dim=1)
        seasonal = torch.cat([seasonal[:, -c.label_len:], history.new_zeros(len(history), H, 1)], dim=1)
        memory = self.encoder_embedding(torch.cat([history, historical_cov], -1).transpose(1, 2)).transpose(1, 2)
        memory = self.dropout(memory + self.covariate_embedding(historical_cov))
        for layer in self.encoder:
            memory = layer(memory)
        memory = self.encoder_norm(memory)
        decoded = self.decoder_embedding(torch.cat([seasonal, decoder_cov], -1).transpose(1, 2)).transpose(1, 2)
        decoded = self.dropout(decoded + self.covariate_embedding(decoder_cov))
        for layer in self.decoder:
            decoded, residual_trend = layer(decoded, memory)
            trend = trend + residual_trend
        return (self.seasonal_projection(self.decoder_norm(decoded)) + trend)[:, -H:, 0]


class RotatingWindows(base.Windows):
    """Change origin offsets each epoch; labels always stay inside training data."""
    def __init__(self, y, cov, end, config, experiment):
        self.end = end
        super().__init__(y, cov, np.arange(config.seq_len, end - H + 1, config.stride),
                         config, experiment)

    def epoch(self, number):
        # Unit increments visit every residue modulo the stride.
        offset = (number - 1) % self.config.stride
        self.origins = np.arange(self.config.seq_len + offset, self.end - H + 1,
                                 self.config.stride)
        if not len(self.origins):
            raise ValueError("Insufficient training context for rotating origin offsets.")


def config_for(args, experiment):
    c = base.Config(epochs=args.epochs, patience=args.patience, stride=args.stride,
                    batch_size=args.batch_size, learning_rate=args.learning_rate,
                    stop_blocks=8, validation_blocks=8)
    if experiment != "reference":
        c = replace(c, d_model=24, d_ff=48, encoder_layers=1, dropout=0.05)
    c.validate()
    return c


def model_for(c, experiment):
    return base.Autoformer(c) if experiment == "reference" else CovariateAutoformer(c)


def data_mode(experiment):
    return "target_only" if experiment == "no_external" else "known_future"


def source_hashes():
    return {"task2_v2.py": base.digest(__file__), "task2.py": base.digest(base.__file__)}


def budget(best_epochs):
    return max(1, int(math.floor(float(np.median(best_epochs)) + 0.5)))


def prepare(target, external, end, c, experiment, seed):
    scaler = base.fit_scaler(target, external, end)
    y, cov = base.transform(target, external, scaler, data_mode(experiment))
    dataset = RotatingWindows(y, cov, end, c, data_mode(experiment))
    return scaler, y, cov, dataset, base.loader(dataset, c, seed, shuffle=True)


def empty_or_completed(path, identity):
    if (path / "result.json").exists():
        result = json.loads((path / "result.json").read_text())
        if any(result[k] != v for k, v in identity.items()):
            raise ValueError(f"Settings/code/data changed at {path}. Choose a new --output.")
        print(f"Skipping completed {path}", flush=True)
        return result
    if path.exists() and any(path.iterdir()):
        raise ValueError(f"Interrupted run at {path}; preserve it and use a new --output.")
    path.mkdir(parents=True, exist_ok=True)
    base.write_json(path / "config.json", identity)
    return None


def fit_stage(args, data, experiment, c, seed, end, path, identity, epochs=None, stop_origins=None):
    completed = empty_or_completed(path, identity)
    if completed is not None:
        return completed
    target, external = data[:2]
    device = base.setup(seed, args.device, args.threads)
    scaler, y, cov, dataset, batches = prepare(target, external, end, c, experiment, seed)
    model = model_for(c, experiment).to(device)
    optimizer = base.optimizer_for(model, c)
    best, stale, best_epoch, state = math.inf, 0, 0, None
    if stop_origins is not None:
        stopping = base.loader(base.Windows(y, cov, stop_origins, c, data_mode(experiment)), c, seed)
        actual = np.stack([target[o:o + H] for o in stop_origins])
    logs = []
    print(f"\n{path}: P={base.parameter_count(model)}, train_end={end}", flush=True)
    for epoch in range(1, (c.epochs if epochs is None else epochs) + 1):
        dataset.epoch(epoch)
        loss = base.train_epoch(model, batches, optimizer, device)
        row = {"epoch": epoch, "training_MSE_standardized": loss,
               "training_windows": len(dataset)}
        if stop_origins is not None:
            scores = base.metrics(actual, base.predict(model, stopping, scaler, device))
            row.update(scores)
            if scores["RMSE"] < best:
                best, best_epoch, stale, state = scores["RMSE"], epoch, 0, base.cpu_state(model)
            else:
                stale += 1
        else:
            best_epoch = epoch
        logs.append(row)
        pd.DataFrame(logs).to_csv(path / "training.csv", index=False)
        print(f"  epoch {epoch}: {row}", flush=True)
        if stop_origins is not None and stale >= c.patience:
            break
    if stop_origins is None:
        state = base.cpu_state(model)
    torch.save({"state_dict": state, "scaler": scaler, "config": asdict(c)}, path / "model.pt")
    result = {**identity, "P": base.parameter_count(model), "epochs_ran": len(logs),
              "best_epoch": best_epoch, "scaler": scaler, "device": str(device),
              "torch": str(torch.__version__), "numpy": np.__version__,
              "checkpoint_sha256": base.digest(path / "model.pt")}
    base.write_json(path / "result.json", result)
    return result


def identity_for(c, experiment, seed, seeds, data, fold_end, stage):
    return {"config": asdict(c), "experiment": experiment, "seed": seed,
            "seeds": seeds, "data_hashes": data[3], "sources": source_hashes(),
            "fold_end": fold_end, "stage": stage}


def evaluate_saved(args, data, path, result, c, experiment, origins):
    model = model_for(c, experiment)
    checkpoint = torch.load(path / "model.pt", map_location="cpu", weights_only=True)
    if base.digest(path / "model.pt") != result["checkpoint_sha256"]:
        raise ValueError(f"Checkpoint integrity check failed: {path}")
    model.load_state_dict(checkpoint["state_dict"])
    device = base.setup(result["seed"], args.device, args.threads)
    model.to(device)
    y, cov = base.transform(data[0], data[1], result["scaler"], data_mode(experiment))
    dataset = base.Windows(y, cov, origins, c, data_mode(experiment), labels=origins[-1] < len(data[0]))
    return base.predict(model, base.loader(dataset, c, result["seed"]), result["scaler"], device)


def validate(args):
    data = base.load_data(args.data)
    if data[1] is None:
        raise ValueError("All supplied CSVs, including optional_external_data.csv, are required.")
    experiments = EXPERIMENTS if args.experiment == "all" else (args.experiment,)
    seeds = sorted(args.seeds)
    for experiment in experiments:
        c = config_for(args, experiment)
        for fold, fold_end in enumerate((len(data[0]) - 16 * H, len(data[0]))):
            comparison_start = fold_end - c.validation_blocks * H
            train_end = comparison_start - c.stop_blocks * H
            stopping_origins = np.arange(train_end, comparison_start, H)
            comparison_origins = np.arange(comparison_start, fold_end, H)
            root = args.output / experiment / f"fold_{fold}"
            selections = []
            for seed in seeds:
                identity = identity_for(c, experiment, seed, seeds, data, fold_end, "selection")
                selections.append(fit_stage(args, data, experiment, c, seed, train_end,
                                            root / f"seed_{seed}" / "selection", identity,
                                            stop_origins=stopping_origins))
            shared_epochs = budget([r["best_epoch"] for r in selections])
            print(f"{experiment} fold {fold}: shared refit budget = {shared_epochs}", flush=True)
            for seed in seeds:
                path = root / f"seed_{seed}" / "refit"
                identity = identity_for(c, experiment, seed, seeds, data, fold_end, "validation_refit")
                identity["refit_epochs"] = shared_epochs
                result = fit_stage(args, data, experiment, c, seed, comparison_start,
                                   path, identity, epochs=shared_epochs)
                if not (path / "evaluation.json").exists():
                    predictions = evaluate_saved(args, data, path, result, c, experiment, comparison_origins)
                    actual = np.stack([data[0][o:o + H] for o in comparison_origins])
                    frame = pd.DataFrame({"time_idx": np.concatenate([np.arange(o + 1, o + H + 1)
                                                                      for o in comparison_origins]),
                                          "actual": actual.ravel(), "prediction": predictions.ravel()})
                    frame.to_csv(path / "predictions.csv", index=False)
                    blocks = [{"start_time_idx": int(o + 1), **base.metrics(a, p)}
                              for o, a, p in zip(comparison_origins, actual, predictions)]
                    base.write_json(path / "evaluation.json", {"metrics": base.metrics(actual, predictions),
                                    "blocks": blocks, "prediction_sha256": base.digest(path / "predictions.csv")})
    compare(args)


def read_runs(args, require_complete=False):
    runs = []
    for experiment in EXPERIMENTS:
        for fold in (0, 1):
            for path in sorted((args.output / experiment / f"fold_{fold}").glob("seed_*/refit/result.json")):
                root = path.parent
                if not (root / "evaluation.json").exists():
                    continue
                row = json.loads(path.read_text())
                evaluation = json.loads((root / "evaluation.json").read_text())
                if row["sources"] != source_hashes():
                    raise ValueError("Source changed since validation; use a new --output directory.")
                if base.digest(root / "predictions.csv") != evaluation["prediction_sha256"]:
                    raise ValueError("Validation prediction file changed.")
                row.update(fold=fold, frame=pd.read_csv(root / "predictions.csv"), evaluation=evaluation)
                runs.append(row)
    if not runs:
        raise ValueError("No completed validation evaluations; run validate first.")
    first = runs[0]
    for row in runs:
        if row["seeds"] != first["seeds"] or row["data_hashes"] != first["data_hashes"]:
            raise ValueError("Mixed data or seed protocols in output directory.")
        expected_config = dict(first["config"])
        # These are the only deliberately different architecture fields.
        for key in ("d_model", "d_ff", "encoder_layers", "dropout"):
            expected_config.pop(key)
        if any(row["config"][key] != value for key, value in expected_config.items()):
            raise ValueError("Mixed training protocols; use one output directory per protocol.")
    if require_complete:
        for experiment in EXPERIMENTS:
            for fold in (0, 1):
                found = sorted(r["seed"] for r in runs if r["experiment"] == experiment and r["fold"] == fold)
                if found != first["seeds"]:
                    raise ValueError(f"Finish all seeds for {experiment}, fold {fold}, before forecasting.")
    return runs


def compare(args):
    runs = read_runs(args)
    rows, paired = [], []
    for experiment in EXPERIMENTS:
        for fold in (0, 1):
            group = sorted([r for r in runs if r["experiment"] == experiment and r["fold"] == fold],
                           key=lambda r: r["seed"])
            if not group or [r["seed"] for r in group] != runs[0]["seeds"]:
                continue
            row = {"experiment": experiment, "fold": fold, "P_single": group[0]["P"],
                   "P_ensemble": sum(r["P"] for r in group), "refit_epochs_each": group[0]["refit_epochs"],
                   "fixed_seed": group[0]["seed"],
                   "fixed_seed_RMSE": group[0]["evaluation"]["metrics"]["RMSE"]}
            for metric in ("MAE", "RMSE", "sMAPE"):
                values = [r["evaluation"]["metrics"][metric] for r in group]
                row[metric + "_mean"] = float(np.mean(values))
                row[metric + "_std"] = float(np.std(values, ddof=1))
            for r in group[1:]:
                if not np.array_equal(r["frame"].time_idx, group[0]["frame"].time_idx):
                    raise ValueError("Misaligned ensemble evaluation indices.")
            average = np.mean([r["frame"].prediction.to_numpy() for r in group], axis=0)
            row.update({"ensemble_" + key: value for key, value in
                        base.metrics(group[0]["frame"].actual.to_numpy(), average).items()})
            rows.append(row)
    for treatment, control in (("future", "no_external"), ("future", "reference")):
        for a in [r for r in runs if r["experiment"] == treatment]:
            b = next((r for r in runs if r["experiment"] == control and r["fold"] == a["fold"]
                      and r["seed"] == a["seed"]), None)
            if b is not None:
                paired.append({"treatment": treatment, "control": control, "fold": a["fold"],
                               "seed": a["seed"], **{f"delta_{m}": a["evaluation"]["metrics"][m]
                               - b["evaluation"]["metrics"][m] for m in ("MAE", "RMSE", "sMAPE")}})
    if rows:
        frame = pd.DataFrame(rows)
        frame.to_csv(args.output / "comparison.csv", index=False)
        print(frame.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        print("Lower errors are better; SD is across seeds. Check BOTH folds, including ensemble costs.")
    if paired:
        pd.DataFrame(paired).to_csv(args.output / "paired_comparison.csv", index=False)
        print("paired_comparison.csv: negative deltas favor the new future-covariate model.")
    return runs


def forecast(args):
    runs = read_runs(args, require_complete=True)
    data = base.load_data(args.data)
    latest = sorted([r for r in runs if r["experiment"] == args.experiment and r["fold"] == 1],
                    key=lambda r: r["seed"])
    if latest[0]["data_hashes"] != data[3]:
        raise ValueError("Data changed since validation.")
    c = base.Config(**latest[0]["config"])
    seeds = latest[0]["seeds"]
    members = seeds if args.members == "ensemble" else [seeds[0]]
    epochs = latest[0]["refit_epochs"]
    destination = args.output / "submissions" / f"{args.experiment}_{args.members}"
    if (destination / "submission_metadata.json").exists():
        raise ValueError(f"Submission already exists at {destination}; do not refit unnecessarily.")
    predictions, fits = [], []
    for seed in members:
        identity = identity_for(c, args.experiment, seed, seeds, data, len(data[0]), "final_refit")
        identity["refit_epochs"] = epochs
        path = destination / f"seed_{seed}"
        result = fit_stage(args, data, args.experiment, c, seed, len(data[0]), path, identity, epochs=epochs)
        predictions.append(evaluate_saved(args, data, path, result, c, args.experiment, [len(data[0])]).ravel())
        fits.append(result)
    values = np.mean(predictions, axis=0)
    if values.shape != (H,) or not np.isfinite(values).all():
        raise ValueError("Expected exactly 168 finite predictions.")
    text = ", ".join(format(float(v), ".10g") for v in values)
    (destination / "predictions.txt").write_text(text + "\n")
    pd.DataFrame({"time_idx": data[2], "prediction": [float(v) for v in text.split(",")]}).to_csv(
        destination / "predictions_audit.csv", index=False)
    ledger = []
    for path in sorted(args.output.glob("*/fold_*/seed_*/*/result.json")):
        r = json.loads(path.read_text())
        ledger.append({"path": str(path.relative_to(args.output)), "experiment": r["experiment"],
                       "stage": r["stage"], "epochs_ran": r["epochs_ran"]})
    selected_cost = sum(r["epochs_ran"] for r in ledger if r["experiment"] == args.experiment)
    final_cost = sum(r["epochs_ran"] for r in fits)
    metadata = {"experiment": args.experiment, "members": members, "P": sum(r["P"] for r in fits),
                "E": selected_cost + final_cost, "selected_experiment_development_epochs": selected_cost,
                "final_refit_epochs_total": final_cost, "epochs_each_final_member": epochs,
                "epoch_accounting": "Conservative: ALL selection and validation-refit epochs of the "
                "selected experiment across both folds and all seeds, plus all final member epochs.",
                "all_architecture_search_plus_this_final_fit_epochs": sum(r["epochs_ran"] for r in ledger) + final_cost,
                "research_ledger": ledger, "sources": source_hashes(), "data_hashes": data[3],
                "first_time_idx": 43657, "last_time_idx": 43824, "prediction_count": H,
                "predictions_sha256": base.digest(destination / "predictions.txt"),
                "member_checkpoints": [{"seed": r["seed"], "P": r["P"],
                                        "sha256": r["checkpoint_sha256"]} for r in fits]}
    base.write_json(destination / "submission_metadata.json", metadata)
    print(f"\nManual review: {destination / 'predictions.txt'}")
    print(f"Declare P={metadata['P']}, E={metadata['E']}. No leaderboard submission was made.")


def self_test(args):
    base.self_test(args)
    base.setup(0, args.device, args.threads)
    c = base.Config(d_model=24, d_ff=48, encoder_layers=1, dropout=0)
    model = CovariateAutoformer(c)
    x = torch.randn(2, c.seq_len, 1)
    cov = torch.randn(2, c.seq_len, 11)
    for availability in (0, 1):
        model.zero_grad(set_to_none=True)
        out = model(x, cov * availability, cov * availability)
        assert out.shape == (2, H) and torch.isfinite(out).all()
        out.square().mean().backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    # Verify trained forecast can actually depend on time-delay aggregation.
    model.eval()
    original = model(x, cov, cov).detach()
    hooks = [m.register_forward_hook(lambda module, inputs, output: torch.zeros_like(output))
             for m in model.modules() if isinstance(m, CalibratedCorrelation)]
    removed = model(x, cov, cov).detach()
    for hook in hooks:
        hook.remove()
    assert not torch.allclose(original, removed), "Autocorrelation path must influence predictions."
    seasonal, trend = MultiscaleDecomposition()(x)
    torch.testing.assert_close(seasonal + trend, x)
    assert budget([3, 7, 5]) == 5
    y, ext = np.zeros(2000, np.float32), np.zeros((2168, 11), np.float32)
    windows = RotatingWindows(y, ext, 1500, c, "known_future")
    for epoch in range(1, 30):
        windows.epoch(epoch)
        assert windows.origins.min() >= c.seq_len and windows.origins.max() + H <= 1500
    # Prefix-fitted transforms keep every unknown/held-out target out of inputs.
    scalers = base.fit_scaler(y, ext[:, :10], 1500)
    for mode in ("target_only", "known_future"):
        normalized, covariates = base.transform(y, ext[:, :10], scalers, mode)
        original_window = base.Windows(normalized, covariates, [1500], c, mode)
        changed = normalized.copy()
        changed[1500:] = 10000
        changed_window = base.Windows(changed, covariates, [1500], c, mode)
        for before, after in zip(original_window[0][:3], changed_window[0][:3]):
            torch.testing.assert_close(before, after)
    print(f"V2 PASS: finite gradients, active autocorrelation, multiscale reconstruction, "
          f"shared epoch budget, rotating-window boundaries. P={base.parameter_count(model)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--data", type=Path, default=base.HERE / "Data")
    common.add_argument("--output", type=Path, default=base.HERE / "runs_task2_v2")
    common.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    common.add_argument("--threads", type=int, default=4)
    validation = commands.add_parser("validate", parents=[common])
    validation.add_argument("--experiment", choices=("all",) + EXPERIMENTS, default="all")
    validation.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    validation.add_argument("--epochs", type=int, default=20)
    validation.add_argument("--patience", type=int, default=5)
    validation.add_argument("--stride", type=int, default=12)
    validation.add_argument("--batch-size", type=int, default=64)
    validation.add_argument("--learning-rate", type=float, default=0.001)
    commands.add_parser("compare", parents=[common])
    commands.add_parser("self-test", parents=[common])
    final = commands.add_parser("forecast", parents=[common])
    final.add_argument("--experiment", choices=EXPERIMENTS, default="future")
    final.add_argument("--members", choices=("single", "ensemble"), default="single")
    args = parser.parse_args()
    try:
        if args.threads < 1:
            raise ValueError("--threads must be positive.")
        if args.command == "validate":
            if len(set(args.seeds)) != len(args.seeds) or len(args.seeds) < 2:
                raise ValueError("Use at least two distinct seeds; three recommended.")
            if any(s < 0 or s >= 2**32 for s in args.seeds):
                raise ValueError("Seeds must be in [0, 2**32).")
            validate(args)
        elif args.command == "compare":
            compare(args)
        elif args.command == "forecast":
            forecast(args)
        else:
            self_test(args)
    except (ValueError, FileNotFoundError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
