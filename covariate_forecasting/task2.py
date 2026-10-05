#!/usr/bin/env python3
"""Task 2: three controlled Autoformer experiments and manual submission export.

Implementation written from scratch for DL4STG-PA1, Sections 2.1--2.8.
Conceptual starting point: Question 1's decomposition and FFT delay aggregation;
this file implements its own encoder/decoder, progressive decoder trend
accumulation, data pipeline, validation, and training. No public source code was
copied. AI-assisted implementation: preserve this conversation for disclosure.

Experiments (identical architecture, training budget, splits, and seed list):
  target_only:    target history; external inputs and availability flags are zero.
  history_only:   target history plus historical external measurements.
  known_future:  history_only plus supplied external measurements for the horizon.
The latter two isolate the value of history versus known future covariates.
All models retain the same input channels, including an availability flag, for
a controlled ablation. All trainable weights, including unused baseline input
weights, count toward P. No calendar or physical period is inferred from indices.

Protocol: a training prefix, 8 nonoverlapping 168-step early-stopping blocks,
then 16 nonoverlapping 168-step comparison blocks. Each later validation origin
may observe actuals strictly before that origin (walk-forward evaluation), but
weights and scalers remain frozen. Comparison blocks never select checkpoints.
Means/standard deviations are fitted only on the training prefix. External NaNs
are imputed with training-prefix medians. The nonnegative target is standardized
without a log transform: MSE directly targets the leaderboard's RMSE. Negative
forecasts are clipped to zero consistently at validation and submission time.

Autoformer: moving-average decomposition after attention and feed-forward
sublayers, FFT-based lag discovery, top-k delay aggregation, and accumulated
decoder trend. There is NO pointwise attention. Decoder input contains historical
seasonality, zero future seasonality, and the context mean as initial future trend.
Covariates enter separate historical encoder and historical/future decoder slots.
The decoder and encoder have equal length: label_len = seq_len - 168.

One epoch is a full pass through ALL training windows at the configured stride.
Stride 24 reduces overlapping-window cost; it does not assert a daily cycle.
Final forecasting refits one predeclared seed on all observed targets for its
validation-selected best epoch count. E includes that seed's ACTUAL validation
training epochs (not just best_epoch) plus full-data refit epochs. Other search
runs are logged separately, with a conservative all-search epoch total as well.
There is no ensemble and no leaderboard feedback is consumed.

Examples (run from this file's directory; dependencies: numpy, pandas, torch):
  python task2.py self-test
  python task2.py validate --experiment all --seeds 0 1 2
  python task2.py compare
  python task2.py forecast --experiment target_only --seed 0

Nothing runs on import. Outputs are created ONLY when you run a command.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset


HORIZON = 168
TRAIN_ROWS = 43656
EXPERIMENTS = ("target_only", "history_only", "known_future")
FEATURES = [f"feature_{letter}" for letter in "ABCDEFGHIJ"]
HERE = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Config:
    seq_len: int = 336
    d_model: int = 32
    heads: int = 4
    d_ff: int = 64
    encoder_layers: int = 2
    decoder_layers: int = 1
    moving_average: int = 25
    lag_factor: float = 2.0
    dropout: float = 0.1
    epochs: int = 15
    patience: int = 4
    batch_size: int = 64
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    stride: int = 24
    stop_blocks: int = 8
    validation_blocks: int = 16

    @property
    def label_len(self):
        return self.seq_len - HORIZON

    def validate(self):
        positive = (self.d_model, self.heads, self.d_ff, self.encoder_layers,
                    self.decoder_layers, self.epochs, self.patience,
                    self.batch_size, self.stride, self.stop_blocks,
                    self.validation_blocks, self.moving_average)
        if any(v <= 0 for v in positive):
            raise ValueError("Counts and sizes must be positive.")
        if self.seq_len <= HORIZON or self.d_model % self.heads:
            raise ValueError("seq_len must exceed 168; d_model must divide by heads.")
        if self.moving_average % 2 != 1:
            raise ValueError("moving_average must be odd.")
        if not 0 <= self.dropout < 1 or self.lag_factor <= 0:
            raise ValueError("Invalid dropout or lag_factor.")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("Invalid optimizer settings.")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, obj):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def load_data(data_dir):
    """Validate assignment indices; never read any test target values."""
    train_path, test_path = data_dir / "student_train.csv", data_dir / "student_test.csv"
    train = pd.read_csv(train_path, usecols=["time_idx", "value"])
    test = pd.read_csv(test_path, usecols=["time_idx"])
    if not np.array_equal(train.time_idx.to_numpy(), np.arange(1, TRAIN_ROWS + 1)):
        raise ValueError("Expected training indices 1..43656 in chronological order.")
    if not np.array_equal(test.time_idx.to_numpy(), np.arange(43657, 43825)):
        raise ValueError("Expected exactly 168 test indices 43657..43824 in order.")
    target = train.value.to_numpy(dtype=np.float64)
    if not np.isfinite(target).all() or (target < 0).any():
        raise ValueError("Training targets must be finite and nonnegative.")
    external_path = data_dir / "optional_external_data.csv"
    external = None
    hashes = {p.name: digest(p) for p in (train_path, test_path)}
    if external_path.exists():
        frame = pd.read_csv(external_path)
        if not np.array_equal(frame.time_idx.to_numpy(), np.arange(1, 43825)):
            raise ValueError("External indices must cover 1..43824 exactly in order.")
        external = frame[FEATURES].to_numpy(dtype=np.float64)
        external[~np.isfinite(external)] = np.nan
        hashes[external_path.name] = digest(external_path)
    return target, external, test.time_idx.to_numpy(), hashes


def fit_scaler(target, external, end):
    y = target[:end]
    scaler = {"target_mean": float(y.mean()), "target_std": max(float(y.std()), 1e-8),
              "fit_rows": int(end)}
    if external is not None:
        prefix = external[:end]
        if np.isnan(prefix).all(axis=0).any():
            raise ValueError("An external column is entirely missing in the training prefix.")
        median = np.nanmedian(prefix, axis=0)
        filled = np.where(np.isnan(prefix), median, prefix)
        scaler.update(external_median=median.tolist(), external_mean=filled.mean(0).tolist(),
                      external_std=np.maximum(filled.std(0), 1e-8).tolist())
    return scaler


def transform(target, external, scaler, experiment):
    y = ((target - scaler["target_mean"]) / scaler["target_std"]).astype(np.float32)
    # Ten channels plus an explicit availability flag; zero means unavailable.
    cov = np.zeros((len(target) + HORIZON, 11), dtype=np.float32)
    if experiment != "target_only":
        if external is None:
            raise ValueError(f"{experiment} requires optional_external_data.csv.")
        filled = np.where(np.isnan(external), scaler["external_median"], external)
        cov[:, :10] = (filled - scaler["external_mean"]) / scaler["external_std"]
        cov[:, 10] = 1.0
    if not np.isfinite(y).all() or not np.isfinite(cov).all():
        raise ValueError("Nonfinite values after scaling.")
    return y, cov


class Windows(Dataset):
    def __init__(self, y, cov, origins, config, experiment, labels=True):
        self.y, self.cov = y, cov
        self.origins = np.asarray(origins, dtype=np.int64)
        self.config, self.experiment, self.labels = config, experiment, labels
        if len(self.origins) == 0:
            raise ValueError("The requested split contains no forecast windows.")
        if self.origins.min() < config.seq_len or self.origins.max() > len(y):
            raise ValueError("Forecast origin has insufficient observed context.")
        limit = len(y) if labels else len(cov)
        if self.origins.max() + HORIZON > limit:
            raise ValueError("Forecast window extends beyond available data.")

    def __len__(self):
        return len(self.origins)

    def __getitem__(self, index):
        origin = int(self.origins[index])
        c = self.config
        past = self.y[origin - c.seq_len:origin, None]
        encoder_cov = self.cov[origin - c.seq_len:origin]
        decoder_cov = np.zeros((c.seq_len, 11), dtype=np.float32)
        decoder_cov[:c.label_len] = self.cov[origin - c.label_len:origin]
        if self.experiment == "known_future":
            decoder_cov[c.label_len:] = self.cov[origin:origin + HORIZON]
        label = self.y[origin:origin + HORIZON] if self.labels else np.zeros(HORIZON, np.float32)
        return (torch.from_numpy(past.copy()), torch.from_numpy(encoder_cov.copy()),
                torch.from_numpy(decoder_cov), torch.from_numpy(label.copy()))


class SeriesDecomposition(nn.Module):
    def __init__(self, kernel):
        super().__init__()
        self.kernel = kernel

    def forward(self, x):
        pad = (self.kernel - 1) // 2
        trend = F.avg_pool1d(F.pad(x.transpose(1, 2), (pad, pad), mode="replicate"),
                             self.kernel, stride=1).transpose(1, 2)
        return x - trend, trend


class AutoCorrelation(nn.Module):
    """Series-wise FFT correlation followed by weighted circular time delays.

    irfft(fft(q) * conj(fft(k)))[lag] = sum_t q[t] * k[t-lag].
    Values therefore aggregate at t-lag, not t+lag. Delay selection is per
    example, shared over heads, and identical during training and inference.
    Lag zero is permitted; top-k also searches every other lag. Centering and
    length normalization stabilize correlation magnitudes across contexts.
    """
    def __init__(self, width, heads, factor, dropout):
        super().__init__()
        self.heads, self.factor = heads, factor
        self.query = nn.Linear(width, width)
        self.key = nn.Linear(width, width)
        self.value = nn.Linear(width, width)
        self.output = nn.Linear(width, width)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def lag_scores(q, k):
        # [batch, heads, head_width, length]
        q = q - q.mean(dim=-1, keepdim=True)
        k = k - k.mean(dim=-1, keepdim=True)
        spectrum = torch.fft.rfft(q, dim=-1) * torch.fft.rfft(k, dim=-1).conj()
        return torch.fft.irfft(spectrum, n=q.shape[-1], dim=-1).mean((1, 2)) / q.shape[-1]

    @staticmethod
    def aggregate(values, delays, weights):
        batch, heads, width, length = values.shape
        positions = torch.arange(length, device=values.device)[None, :]
        result = torch.zeros_like(values)
        for rank in range(delays.shape[1]):
            indices = (positions - delays[:, rank, None]) % length
            indices = indices[:, None, None, :].expand(batch, heads, width, length)
            shifted = torch.gather(values, dim=-1, index=indices)
            result = result + weights[:, rank, None, None, None] * shifted
        return result

    def forward(self, query, key, value):
        batch, length, width = query.shape
        if key.shape[1] != length or value.shape[1] != length:
            raise ValueError("This implementation uses equal encoder/decoder lengths.")

        def split(x):
            return x.reshape(batch, length, self.heads, width // self.heads).permute(0, 2, 3, 1)

        q, k, v = split(self.query(query)), split(self.key(key)), split(self.value(value))
        scores = self.lag_scores(q, k)
        count = min(length, max(1, int(self.factor * math.log(length))))
        selected, delays = scores.topk(count, dim=-1)
        weights = selected.softmax(dim=-1)
        result = self.aggregate(v, delays, weights)
        result = result.permute(0, 3, 1, 2).reshape(batch, length, width)
        return self.dropout(self.output(result))


class SeasonalNorm(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.norm = nn.LayerNorm(width)

    def forward(self, x):
        x = self.norm(x)
        return x - x.mean(dim=1, keepdim=True)


def feed_forward(c):
    return nn.Sequential(nn.Linear(c.d_model, c.d_ff), nn.GELU(), nn.Dropout(c.dropout),
                         nn.Linear(c.d_ff, c.d_model), nn.Dropout(c.dropout))


class EncoderLayer(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.correlation = AutoCorrelation(c.d_model, c.heads, c.lag_factor, c.dropout)
        self.decomposition = SeriesDecomposition(c.moving_average)
        self.ff = feed_forward(c)

    def forward(self, x):
        x, _ = self.decomposition(x + self.correlation(x, x, x))
        x, _ = self.decomposition(x + self.ff(x))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.self_correlation = AutoCorrelation(c.d_model, c.heads, c.lag_factor, c.dropout)
        self.cross_correlation = AutoCorrelation(c.d_model, c.heads, c.lag_factor, c.dropout)
        self.decomposition = SeriesDecomposition(c.moving_average)
        self.ff = feed_forward(c)
        self.trend_projection = nn.Conv1d(c.d_model, 1, 3, padding=1,
                                          padding_mode="circular", bias=False)

    def forward(self, x, memory):
        x, trend1 = self.decomposition(x + self.self_correlation(x, x, x))
        x, trend2 = self.decomposition(x + self.cross_correlation(x, memory, memory))
        x, trend3 = self.decomposition(x + self.ff(x))
        trend = self.trend_projection((trend1 + trend2 + trend3).transpose(1, 2)).transpose(1, 2)
        return x, trend


class Autoformer(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.config = c
        self.pred_len = HORIZON
        self.decomposition = SeriesDecomposition(c.moving_average)
        # Local convolution encodes ordering without fabricated calendar features.
        self.encoder_embedding = nn.Conv1d(12, c.d_model, 3, padding=1, padding_mode="circular")
        self.decoder_embedding = nn.Conv1d(12, c.d_model, 3, padding=1, padding_mode="circular")
        self.dropout = nn.Dropout(c.dropout)
        self.encoder = nn.ModuleList([EncoderLayer(c) for _ in range(c.encoder_layers)])
        self.decoder = nn.ModuleList([DecoderLayer(c) for _ in range(c.decoder_layers)])
        self.encoder_norm, self.decoder_norm = SeasonalNorm(c.d_model), SeasonalNorm(c.d_model)
        self.seasonal_projection = nn.Linear(c.d_model, 1)

    def forward(self, history, historical_cov, decoder_cov):
        c = self.config
        seasonal, trend = self.decomposition(history)
        seasonal = torch.cat([seasonal[:, -c.label_len:],
                              history.new_zeros(history.shape[0], HORIZON, 1)], dim=1)
        trend = torch.cat([trend[:, -c.label_len:],
                           history.mean(1, keepdim=True).expand(-1, HORIZON, -1)], dim=1)
        memory = self.dropout(self.encoder_embedding(
            torch.cat([history, historical_cov], dim=-1).transpose(1, 2)).transpose(1, 2))
        for layer in self.encoder:
            memory = layer(memory)
        memory = self.encoder_norm(memory)
        decoded = self.dropout(self.decoder_embedding(
            torch.cat([seasonal, decoder_cov], dim=-1).transpose(1, 2)).transpose(1, 2))
        # Direct multi-step decoder: future targets are never teacher-forced.
        # Noncausal lag aggregation sees only zeros and permitted covariates there.
        for layer in self.decoder:
            decoded, residual_trend = layer(decoded, memory)
            trend = trend + residual_trend
        output = self.seasonal_projection(self.decoder_norm(decoded)) + trend
        return output[:, -HORIZON:, 0]


def metrics(actual, predicted):
    actual, predicted = np.asarray(actual, np.float64), np.asarray(predicted, np.float64)
    if actual.shape != predicted.shape or not np.isfinite(predicted).all():
        raise ValueError("Invalid prediction shape or nonfinite forecasts.")
    error = predicted - actual
    denominator = np.abs(actual) + np.abs(predicted)
    ratio = np.divide(2 * np.abs(error), denominator,
                      out=np.zeros_like(error), where=denominator > 0)
    return {"MAE": float(np.abs(error).mean()), "RMSE": float(np.sqrt(np.square(error).mean())),
            "sMAPE": float(100 * ratio.mean())}


def setup(seed, requested_device, threads):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = ("cuda" if torch.cuda.is_available() else "cpu") if requested_device == "auto" else requested_device
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable. Use --device cpu.")
    # macOS deliberately uses CPU: this FFT implementation does not require MPS support.
    return torch.device(device)


def loader(dataset, c, seed, shuffle=False):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=c.batch_size, shuffle=shuffle,
                      num_workers=0, generator=generator, drop_last=False)


def train_epoch(model, batches, optimizer, device):
    model.train()
    total, count = 0.0, 0
    for past, enc, dec, actual in batches:
        past, enc, dec, actual = [x.to(device) for x in (past, enc, dec, actual)]
        optimizer.zero_grad(set_to_none=True)
        predicted = model(past, enc, dec)
        loss = F.mse_loss(predicted, actual)
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite training loss; lower learning rate in a new output directory.")
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        total += float(loss.detach()) * len(past)
        count += len(past)
    return total / count


@torch.no_grad()
def predict(model, batches, scaler, device):
    model.eval()
    predictions = []
    for past, enc, dec, _ in batches:
        output = model(past.to(device), enc.to(device), dec.to(device)).cpu().numpy().astype(np.float64)
        output = np.maximum(0.0, output * scaler["target_std"] + scaler["target_mean"])
        if not np.isfinite(output).all():
            raise RuntimeError("Model produced nonfinite predictions.")
        predictions.append(output)
    return np.concatenate(predictions)


def cpu_state(model):
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}


def parameter_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def optimizer_for(model, c):
    return torch.optim.AdamW(model.parameters(), lr=c.learning_rate, weight_decay=c.weight_decay)


def split_origins(c, n):
    comparison_start = n - c.validation_blocks * HORIZON
    train_end = comparison_start - c.stop_blocks * HORIZON
    if train_end < c.seq_len + HORIZON:
        raise ValueError("Validation blocks leave too little training history.")
    return (train_end, np.arange(c.seq_len, train_end - HORIZON + 1, c.stride),
            np.arange(train_end, comparison_start, HORIZON),
            np.arange(comparison_start, n, HORIZON))


def validate_run(args, c, experiment, seed, seed_list, data):
    target, external, _, hashes = data
    path = args.output / experiment / f"seed_{seed}"
    identity = {"config": asdict(c), "experiment": experiment, "seed": seed,
                "seed_list": seed_list, "data_hashes": hashes, "source_sha256": digest(__file__)}
    if (path / "result.json").exists():
        result = json.loads((path / "result.json").read_text())
        if any(result[key] != value for key, value in identity.items()):
            raise ValueError(f"Existing run differs: {path}. Choose a new --output directory.")
        print(f"Skipping completed run: {path}", flush=True)
        return
    if path.exists() and any(path.iterdir()):
        raise ValueError(f"Incomplete run at {path}. Preserve it and choose a new --output directory.")
    path.mkdir(parents=True, exist_ok=True)
    device = setup(seed, args.device, args.threads)
    train_end, train_origins, stopping_origins, comparison_origins = split_origins(c, len(target))
    scaler = fit_scaler(target, external, train_end)
    y, cov = transform(target, external, scaler, experiment)
    training = loader(Windows(y, cov, train_origins, c, experiment), c, seed, shuffle=True)
    stopping = loader(Windows(y, cov, stopping_origins, c, experiment), c, seed)
    comparison = loader(Windows(y, cov, comparison_origins, c, experiment), c, seed)
    stopping_actual = np.stack([target[o:o + HORIZON] for o in stopping_origins])
    comparison_actual = np.stack([target[o:o + HORIZON] for o in comparison_origins])
    model = Autoformer(c).to(device)
    optimizer = optimizer_for(model, c)
    best_rmse, best_epoch, stale, best_state = math.inf, 0, 0, None
    epoch_log, start = [], time.monotonic()
    write_json(path / "run_config.json", {**identity, "scaler": scaler, "device": str(device)})
    print(f"\n{experiment}, seed={seed}, P={parameter_count(model):,}, device={device}; "
          f"{len(train_origins)} training windows; training ends at time_idx={train_end}", flush=True)
    for epoch in range(1, c.epochs + 1):
        loss = train_epoch(model, training, optimizer, device)
        scores = metrics(stopping_actual, predict(model, stopping, scaler, device))
        epoch_log.append({"epoch": epoch, "training_MSE_standardized": loss, **scores})
        pd.DataFrame(epoch_log).to_csv(path / "training.csv", index=False)
        print(f"  epoch {epoch:02d}: loss={loss:.5f}; stopping RMSE={scores['RMSE']:.5f}", flush=True)
        if scores["RMSE"] < best_rmse:
            best_rmse, best_epoch, stale = scores["RMSE"], epoch, 0
            best_state = cpu_state(model)
        else:
            stale += 1
        if stale >= c.patience:
            break
    model.load_state_dict(best_state)
    predictions = predict(model, comparison, scaler, device)
    scores = metrics(comparison_actual, predictions)
    block_metrics = [{"forecast_start_time_idx": int(o + 1),
                      **metrics(actual, pred)}
                     for o, actual, pred in zip(comparison_origins, comparison_actual, predictions)]
    pd.DataFrame(block_metrics).to_csv(path / "validation_blocks.csv", index=False)
    pd.DataFrame({"time_idx": np.concatenate([np.arange(o + 1, o + HORIZON + 1)
                                               for o in comparison_origins]),
                  "actual": comparison_actual.ravel(), "prediction": predictions.ravel()
                  }).to_csv(path / "validation_predictions.csv", index=False)
    torch.save({"state_dict": best_state, "config": asdict(c), "scaler": scaler,
                "experiment": experiment, "seed": seed, "best_epoch": best_epoch}, path / "best.pt")
    result = {**identity, "P": parameter_count(model), "epochs_ran": len(epoch_log),
              "best_epoch": best_epoch, "stopping_RMSE": best_rmse, "metrics": scores,
              "train_end_time_idx": train_end,
              "stopping_start_time_idx": int(stopping_origins[0] + 1),
              "comparison_start_time_idx": int(comparison_origins[0] + 1),
              "validation_blocks": block_metrics, "scaler": scaler,
              "seconds": time.monotonic() - start, "device": str(device),
              "versions": {"python": sys.version, "torch": str(torch.__version__),
                           "numpy": np.__version__, "pandas": pd.__version__}}
    write_json(path / "result.json", result)  # Completion marker is written last.
    print(f"  held-out comparison: {scores}; best_epoch={best_epoch}, "
          f"epochs_ran={len(epoch_log)}", flush=True)


def collect_results(output):
    paths = sorted(output.glob("*/seed_*/result.json"))
    if not paths:
        raise ValueError("No completed validation runs. Run the validate command first.")
    results = [json.loads(path.read_text()) for path in paths]
    reference = results[0]
    for row in results:
        for field in ("config", "seed_list", "data_hashes", "source_sha256"):
            if row[field] != reference[field]:
                raise ValueError(f"Mixed protocols in {output}: {field} differs.")
    return results


def compare(output, require_complete=False):
    results = collect_results(output)
    rows = []
    for experiment in EXPERIMENTS:
        group = sorted([r for r in results if r["experiment"] == experiment], key=lambda r: r["seed"])
        if not group:
            if require_complete:
                raise ValueError(f"Missing {experiment}; finish all three experiments before forecasting.")
            continue
        complete = [r["seed"] for r in group] == sorted(group[0]["seed_list"])
        if not complete or len(group) < 2:
            if require_complete:
                raise ValueError(f"Incomplete multiple-seed experiment: {experiment}.")
            print(f"Not comparing incomplete experiment {experiment}: {len(group)} seed(s).")
            continue
        row = {"experiment": experiment, "seeds": len(group), "P": group[0]["P"],
               "epochs_total": sum(r["epochs_ran"] for r in group),
               "best_epoch_median": float(np.median([r["best_epoch"] for r in group]))}
        seed_zero = next((r for r in group if r["seed"] == 0), None)
        if seed_zero is not None:
            row["seed_0_planned_final_E"] = seed_zero["epochs_ran"] + seed_zero["best_epoch"]
        for name in ("MAE", "RMSE", "sMAPE"):
            values = [r["metrics"][name] for r in group]
            row[name + "_mean"], row[name + "_std"] = float(np.mean(values)), float(np.std(values, ddof=1))
        rows.append(row)
    if rows:
        frame = pd.DataFrame(rows).sort_values("RMSE_mean")
        frame.to_csv(output / "comparison.csv", index=False)
        print("\nChronological comparison (mean and sample SD across seeds):")
        print(frame.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
        print("Sorted by RMSE only. Leaderboard alpha/beta are undisclosed; this is not its score.")
    # Paired seeds avoid confusing seed noise with the optional-data effect.
    by_key = {(r["experiment"], r["seed"]): r for r in results}
    paired = []
    for treatment, control in (("history_only", "target_only"),
                               ("known_future", "target_only"),
                               ("known_future", "history_only")):
        for seed in results[0]["seed_list"]:
            if (treatment, seed) in by_key and (control, seed) in by_key:
                a, b = by_key[treatment, seed], by_key[control, seed]
                paired.append({"treatment": treatment, "control": control, "seed": seed,
                               **{f"delta_{m}": a["metrics"][m] - b["metrics"][m]
                                  for m in ("MAE", "RMSE", "sMAPE")}})
    if paired:
        pd.DataFrame(paired).to_csv(output / "paired_ablation.csv", index=False)
        print("Saved paired_ablation.csv: negative deltas favor the treatment.")
    return results


def forecast(args):
    results = compare(args.output, require_complete=True)
    source = next((r for r in results if r["experiment"] == args.experiment and r["seed"] == args.seed), None)
    if source is None:
        raise ValueError("Final seed must have a completed validation run in the selected experiment.")
    # Predetermine seed 0 (default), rather than selecting the luckiest validation seed.
    target, external, test_indices, hashes = load_data(args.data)
    if source["data_hashes"] != hashes or source["source_sha256"] != digest(__file__):
        raise ValueError("Data or implementation changed since validation. Use a new output directory.")
    c = Config(**source["config"])
    c.validate()
    destination = args.output / "submissions" / f"{args.experiment}_seed_{args.seed}"
    if destination.exists() and any(destination.iterdir()):
        raise ValueError(f"Refusing to overwrite existing forecast or interrupted fit: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    device = setup(args.seed, args.device, args.threads)
    scaler = fit_scaler(target, external, len(target))
    y, cov = transform(target, external, scaler, args.experiment)
    origins = np.arange(c.seq_len, len(target) - HORIZON + 1, c.stride)
    batches = loader(Windows(y, cov, origins, c, args.experiment), c, args.seed, shuffle=True)
    model = Autoformer(c).to(device)
    optimizer = optimizer_for(model, c)
    epochs = int(source["best_epoch"])
    write_json(destination / "refit_config.json", {"config": asdict(c), "scaler": scaler,
               "experiment": args.experiment, "seed": args.seed, "refit_epochs": epochs,
               "source_sha256": source["source_sha256"], "data_hashes": hashes,
               "validation_epochs_ran": source["epochs_ran"]})
    log = []
    for epoch in range(1, epochs + 1):
        loss = train_epoch(model, batches, optimizer, device)
        log.append({"epoch": epoch, "training_MSE_standardized": loss})
        pd.DataFrame(log).to_csv(destination / "refit_training.csv", index=False)
        print(f"refit epoch {epoch}/{epochs}: loss={loss:.5f}", flush=True)
    future = loader(Windows(y, cov, [len(target)], c, args.experiment, labels=False), c, args.seed)
    predictions = predict(model, future, scaler, device).reshape(-1)
    if predictions.shape != (HORIZON,) or not np.isfinite(predictions).all():
        raise RuntimeError("Expected exactly 168 finite forecasts.")
    text = ", ".join(format(float(value), ".10g") for value in predictions)
    if len(text.split(",")) != HORIZON:
        raise RuntimeError("Submission formatting failed.")
    # Use the same rounded numbers in both files, so audit CSV matches pasted text.
    rounded = np.array([float(value) for value in text.split(",")])
    (destination / "predictions.txt").write_text(text + "\n")
    pd.DataFrame({"time_idx": test_indices, "prediction": rounded}).to_csv(
        destination / "predictions_audit.csv", index=False)
    torch.save({"state_dict": cpu_state(model), "config": asdict(c), "scaler": scaler,
                "experiment": args.experiment, "seed": args.seed}, destination / "model.pt")
    parameter_total = parameter_count(model)
    if parameter_total != source["P"]:
        raise RuntimeError("Final parameter count differs from validated architecture.")
    search_epochs = sum(r["epochs_ran"] for r in results)
    metadata = {"experiment": args.experiment, "seed": args.seed, "P": parameter_total,
                "E": source["epochs_ran"] + epochs,
                "validation_epochs_for_submitted_seed": source["epochs_ran"],
                "refit_epochs": epochs, "all_completed_search_epochs": search_epochs,
                "all_search_plus_refit_epochs": search_epochs + epochs,
                "epoch_accounting": "E = actual validation-training epochs for this seed + "
                "full-data refit epochs. Other experimental fits are separately itemized; "
                "all_search_plus_refit_epochs also includes every completed search fit in this output directory.",
                "search_run_epochs": [{"experiment": r["experiment"], "seed": r["seed"],
                                       "epochs_ran": r["epochs_ran"]} for r in results],
                "prediction_count": HORIZON, "first_time_idx": int(test_indices[0]),
                "last_time_idx": int(test_indices[-1]), "config": asdict(c),
                "data_hashes": hashes, "source_sha256": digest(__file__),
                "prediction_text_sha256": digest(destination / "predictions.txt"),
                "checkpoint_sha256": digest(destination / "model.pt"),
                "device": str(device), "torch": str(torch.__version__), "python": sys.version}
    write_json(destination / "submission_metadata.json", metadata)
    print(f"\nREADY FOR MANUAL REVIEW: {destination / 'predictions.txt'}")
    print(f"168 values ordered 43657..43824; declare P={metadata['P']}, E={metadata['E']}.")
    print(f"E includes {source['epochs_ran']} validation-training + {epochs} refit epochs.")
    print("All search costs are itemized in submission_metadata.json. No submission was made.")


def self_test(args):
    """Manual synthetic checks; does not train on assignment data or write results."""
    device = setup(0, args.device, args.threads)
    c = Config(d_model=8, heads=2, d_ff=16, encoder_layers=1, dropout=0.0)
    c.validate()
    decomposition = SeriesDecomposition(5)
    x = torch.randn(2, 17, 3)
    seasonal, trend = decomposition(x)
    torch.testing.assert_close(seasonal + trend, x)
    constant = torch.ones(2, 17, 3)
    torch.testing.assert_close(decomposition(constant)[0], torch.zeros_like(constant))
    q, k = torch.randn(2, 2, 3, 17), torch.randn(2, 2, 3, 17)
    q, k = q - q.mean(-1, keepdim=True), k - k.mean(-1, keepdim=True)
    direct = torch.stack([(q * torch.roll(k, lag, -1)).mean((1, 2, 3)) for lag in range(17)], -1)
    torch.testing.assert_close(AutoCorrelation.lag_scores(q, k), direct)
    values = torch.arange(17, dtype=torch.float32).reshape(1, 1, 1, 17)
    aggregated = AutoCorrelation.aggregate(values, torch.tensor([[3]]), torch.ones(1, 1))
    torch.testing.assert_close(aggregated, torch.roll(values, 3, -1))
    assert metrics([0, 1], [0, 1]) == {"MAE": 0.0, "RMSE": 0.0, "sMAPE": 0.0}
    assert metrics([0], [1])["sMAPE"] == 200.0
    train_end, train_o, stop_o, compare_o = split_origins(c, TRAIN_ROWS)
    assert train_o[-1] + HORIZON <= train_end == stop_o[0]
    assert stop_o[-1] + HORIZON == compare_o[0]
    assert compare_o[-1] + HORIZON == TRAIN_ROWS
    y = np.arange(1000, dtype=np.float32)
    external = np.ones((1168, 11), dtype=np.float32)
    scaler = fit_scaler(y, external[:, :10], 400)
    changed_target, changed_external = y.copy(), external[:, :10].copy()
    changed_target[400:] = 1e6
    changed_external[400:] = 1e6
    assert fit_scaler(changed_target, changed_external, 400) == scaler
    for experiment in EXPERIMENTS:
        cov = external if experiment != "target_only" else np.zeros_like(external)
        window = Windows(y, cov, [400], c, experiment)
        past, enc, dec, label = window[0]
        assert past[-1].item() == 399 and label[0].item() == 400
        assert bool(dec[c.label_len:].any()) == (experiment == "known_future")
        changed_y = y.copy()
        changed_y[400:] = -999  # Changing future labels cannot change any model input.
        for before, after in zip(window[0][:3], Windows(changed_y, cov, [400], c, experiment)[0][:3]):
            torch.testing.assert_close(before, after)
        model = Autoformer(c).to(device)
        output = model(past[None].to(device), enc[None].to(device), dec[None].to(device))
        assert output.shape == (1, HORIZON) and torch.isfinite(output).all()
        output.square().mean().backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    print("PASS: decomposition, FFT/direct correlation, delay direction, metrics, training-only scalers, chronological "
          "boundaries, no future-target inputs, three model forward/backward checks.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--data", type=Path, default=HERE / "Data")
    common.add_argument("--output", type=Path, default=HERE / "runs_task2")
    common.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    common.add_argument("--threads", type=int, default=4)
    validation = commands.add_parser("validate", parents=[common], help="Train and compare chronological validation runs")
    validation.add_argument("--experiment", choices=("all",) + EXPERIMENTS, default="all")
    validation.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    defaults = Config()
    for name, value in asdict(defaults).items():
        validation.add_argument("--" + name.replace("_", "-"), type=type(value), default=value)
    commands.add_parser("compare", parents=[common], help="Summarize completed seed runs; no training")
    final = commands.add_parser("forecast", parents=[common], help="Refit selected model and export 168 values")
    final.add_argument("--experiment", choices=EXPERIMENTS, required=True)
    final.add_argument("--seed", type=int, default=0)
    commands.add_parser("self-test", parents=[common], help="Run synthetic correctness checks only")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    try:
        if args.command == "self-test":
            self_test(args)
        elif args.command == "compare":
            compare(args.output)
        elif args.command == "forecast":
            forecast(args)
        elif args.command == "validate":
            if len(args.seeds) < 2 or len(set(args.seeds)) != len(args.seeds):
                raise ValueError("Provide at least two distinct seeds; three are recommended.")
            if any(seed < 0 or seed >= 2**32 for seed in args.seeds):
                raise ValueError("Seeds must lie in [0, 2**32).")
            config = Config(**{name: getattr(args, name) for name in asdict(defaults)})
            config.validate()
            data = load_data(args.data)
            experiments = EXPERIMENTS if args.experiment == "all" else (args.experiment,)
            if any(e != "target_only" for e in experiments) and data[1] is None:
                raise ValueError("The requested ablation requires optional_external_data.csv.")
            # Check split feasibility before creating any run directory.
            split_origins(config, len(data[0]))
            for experiment in experiments:
                for seed in sorted(args.seeds):
                    validate_run(args, config, experiment, seed, sorted(args.seeds), data)
            compare(args.output)
    except (ValueError, FileNotFoundError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
