from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .data_io import CACHE_DIR, ensure_dir, write_json, write_yaml
from .eval import compute_binary_metrics, select_best_threshold


N_CHANNELS = 21
NORM_MAX_WINDOWS = 65536
NORM_CHUNK_SIZE = 2048
MODEL_NAMES = ("cnn", "lstm")
DEFAULT_SEQUENCE_LENGTH = 8
DEFAULT_ENCODER_DIM = 64
DEFAULT_LSTM_HIDDEN_DIM = 64
DEFAULT_LSTM_NUM_LAYERS = 1
DEFAULT_LSTM_DROPOUT = 0.0


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True


def cache_path_for_patient(patient_id: str, cache_dir: Path = CACHE_DIR) -> Path:
    return cache_dir / f"{patient_id}_windows.float32.npy"


class EEGWindowDataset(Dataset):
    def __init__(
        self,
        rows: pd.DataFrame,
        mean: np.ndarray | None = None,
        std: np.ndarray | None = None,
        cache_dir: Path = CACHE_DIR,
    ) -> None:
        ordered = rows.reset_index(drop=True)
        self.patient_ids = ordered["patient_id"].to_numpy()
        self.window_indices = ordered["window_idx_in_patient"].to_numpy(dtype=np.int64)
        self.labels = ordered["class_label"].to_numpy(dtype=np.float32)
        self.row_ids = ordered["row_id"].to_numpy(dtype=np.int64)
        self.mean = mean.astype(np.float32) if mean is not None else None
        self.std = std.astype(np.float32) if std is not None else None
        self.cache_dir = cache_dir
        self._cache_handles: dict[str, np.memmap] = {}

    def __len__(self) -> int:
        return len(self.row_ids)

    def _load_patient_cache(self, patient_id: str) -> np.memmap:
        if patient_id not in self._cache_handles:
            self._cache_handles[patient_id] = np.load(cache_path_for_patient(patient_id, self.cache_dir), mmap_mode="r")
        return self._cache_handles[patient_id]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        patient_id = self.patient_ids[index]
        cache = self._load_patient_cache(patient_id)
        window = np.asarray(cache[self.window_indices[index]], dtype=np.float32)
        if self.mean is not None and self.std is not None:
            window = (window - self.mean[:, None]) / self.std[:, None]
        label = self.labels[index]
        return torch.from_numpy(window), torch.tensor(label), int(self.row_ids[index])


class EEGSequenceDataset(Dataset):
    def __init__(
        self,
        rows: pd.DataFrame,
        sequence_length: int,
        mean: np.ndarray | None = None,
        std: np.ndarray | None = None,
        cache_dir: Path = CACHE_DIR,
    ) -> None:
        if sequence_length < 2:
            raise ValueError("sequence_length must be >= 2 for the temporal LSTM model.")
        self.sequence_length = int(sequence_length)
        self.mean = mean.astype(np.float32) if mean is not None else None
        self.std = std.astype(np.float32) if std is not None else None
        self.cache_dir = cache_dir
        self._cache_handles: dict[str, np.memmap] = {}
        self.samples = self._build_samples(rows)

    def _build_samples(self, rows: pd.DataFrame) -> list[dict[str, Any]]:
        ordered = rows.sort_values(["patient_id", "filename", "window_idx_in_patient", "row_id"]).reset_index(drop=True)
        samples: list[dict[str, Any]] = []
        for (patient_id, filename), group in ordered.groupby(["patient_id", "filename"], sort=False):
            group = group.sort_values("window_idx_in_patient").reset_index(drop=True)
            window_indices = group["window_idx_in_patient"].to_numpy(dtype=np.int64)
            labels = group["class_label"].to_numpy(dtype=np.float32)
            row_ids = group["row_id"].to_numpy(dtype=np.int64)
            for end in range(len(group)):
                start = max(0, end - self.sequence_length + 1)
                sequence_indices = window_indices[start : end + 1]
                if len(sequence_indices) < self.sequence_length:
                    pad = np.repeat(sequence_indices[0], self.sequence_length - len(sequence_indices))
                    sequence_indices = np.concatenate([pad, sequence_indices])
                samples.append(
                    {
                        "patient_id": patient_id,
                        "filename": filename,
                        "sequence_indices": sequence_indices.copy(),
                        "label": float(labels[end]),
                        "row_id": int(row_ids[end]),
                    }
                )
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _load_patient_cache(self, patient_id: str) -> np.memmap:
        if patient_id not in self._cache_handles:
            self._cache_handles[patient_id] = np.load(cache_path_for_patient(patient_id, self.cache_dir), mmap_mode="r")
        return self._cache_handles[patient_id]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        sample = self.samples[index]
        cache = self._load_patient_cache(str(sample["patient_id"]))
        windows = np.asarray(cache[sample["sequence_indices"]], dtype=np.float32)
        if self.mean is not None and self.std is not None:
            windows = (windows - self.mean[None, :, None]) / self.std[None, :, None]
        label = float(sample["label"])
        row_id = int(sample["row_id"])
        return torch.from_numpy(windows), torch.tensor(label), row_id


class CNNWindowEncoder(nn.Module):
    def __init__(self, embedding_dim: int = DEFAULT_ENCODER_DIM) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(N_CHANNELS, 16, kernel_size=5, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
        )
        self.pool = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten())
        self.projection = nn.Identity() if embedding_dim == 64 else nn.Linear(64, embedding_dim)
        self.embedding_dim = int(embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        pooled = self.pool(features)
        return self.projection(pooled)


class SeizureCNN1D(nn.Module):
    def __init__(self, encoder_dim: int = DEFAULT_ENCODER_DIM) -> None:
        super().__init__()
        self.encoder = CNNWindowEncoder(embedding_dim=encoder_dim)
        self.classifier = nn.Linear(self.encoder.embedding_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embeddings = self.encoder(x)
        return self.classifier(embeddings).squeeze(-1)


class SeizureTemporalLSTM(nn.Module):
    def __init__(
        self,
        sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
        encoder_dim: int = DEFAULT_ENCODER_DIM,
        hidden_dim: int = DEFAULT_LSTM_HIDDEN_DIM,
        num_layers: int = DEFAULT_LSTM_NUM_LAYERS,
        dropout: float = DEFAULT_LSTM_DROPOUT,
    ) -> None:
        super().__init__()
        self.sequence_length = int(sequence_length)
        self.encoder = CNNWindowEncoder(embedding_dim=encoder_dim)
        self.lstm = nn.LSTM(
            input_size=self.encoder.embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            x = x.unsqueeze(1)
        batch_size, seq_len, _, _ = x.shape
        flattened = x.reshape(batch_size * seq_len, N_CHANNELS, -1)
        encoded = self.encoder(flattened).reshape(batch_size, seq_len, -1)
        outputs, _ = self.lstm(encoded)
        return self.classifier(outputs[:, -1, :]).squeeze(-1)


@dataclass
class TrainingArtifacts:
    checkpoint_path: Path
    log_path: Path
    metrics_path: Path
    norm_stats_path: Path
    threshold: float


def compute_channel_norm_stats(rows: pd.DataFrame, cache_dir: Path = CACHE_DIR) -> tuple[np.ndarray, np.ndarray]:
    sampled_rows = rows.reset_index(drop=True)
    if len(sampled_rows) > NORM_MAX_WINDOWS:
        sampled_rows = sampled_rows.sample(n=NORM_MAX_WINDOWS, random_state=2026).reset_index(drop=True)

    sum_channels = np.zeros(N_CHANNELS, dtype=np.float64)
    sumsq_channels = np.zeros(N_CHANNELS, dtype=np.float64)
    count_per_channel = 0
    for patient_id, group in sampled_rows.groupby("patient_id", sort=False):
        cache = np.load(cache_path_for_patient(patient_id, cache_dir), mmap_mode="r")
        indices = np.sort(group["window_idx_in_patient"].to_numpy(dtype=np.int64))
        for start in range(0, len(indices), NORM_CHUNK_SIZE):
            block_idx = indices[start : start + NORM_CHUNK_SIZE]
            windows = np.asarray(cache[block_idx], dtype=np.float32)
            sum_channels += windows.sum(axis=(0, 2), dtype=np.float64)
            sumsq_channels += np.square(windows, dtype=np.float64).sum(axis=(0, 2), dtype=np.float64)
            count_per_channel += windows.shape[0] * windows.shape[2]
    mean = sum_channels / max(count_per_channel, 1)
    variance = np.maximum((sumsq_channels / max(count_per_channel, 1)) - np.square(mean), 1e-8)
    std = np.sqrt(variance)
    return mean.astype(np.float32), std.astype(np.float32)


def build_model(model_name: str, **model_kwargs: Any) -> nn.Module:
    if model_name == "cnn":
        return SeizureCNN1D(encoder_dim=int(model_kwargs.get("encoder_dim", DEFAULT_ENCODER_DIM)))
    if model_name == "lstm":
        return SeizureTemporalLSTM(
            sequence_length=int(model_kwargs.get("sequence_length", DEFAULT_SEQUENCE_LENGTH)),
            encoder_dim=int(model_kwargs.get("encoder_dim", DEFAULT_ENCODER_DIM)),
            hidden_dim=int(model_kwargs.get("hidden_dim", DEFAULT_LSTM_HIDDEN_DIM)),
            num_layers=int(model_kwargs.get("num_layers", DEFAULT_LSTM_NUM_LAYERS)),
            dropout=float(model_kwargs.get("dropout", DEFAULT_LSTM_DROPOUT)),
        )
    raise ValueError(f"Unsupported model_name={model_name!r}. Expected one of {MODEL_NAMES}.")


def create_dataset(
    rows: pd.DataFrame,
    model_name: str,
    mean: np.ndarray | None,
    std: np.ndarray | None,
    cache_dir: Path = CACHE_DIR,
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
) -> Dataset:
    if model_name == "cnn":
        return EEGWindowDataset(rows=rows, mean=mean, std=std, cache_dir=cache_dir)
    if model_name == "lstm":
        return EEGSequenceDataset(
            rows=rows,
            sequence_length=sequence_length,
            mean=mean,
            std=std,
            cache_dir=cache_dir,
        )
    raise ValueError(f"Unsupported model_name={model_name!r}. Expected one of {MODEL_NAMES}.")


def create_dataloader(
    rows: pd.DataFrame,
    batch_size: int,
    shuffle: bool,
    mean: np.ndarray | None,
    std: np.ndarray | None,
    num_workers: int = 0,
    cache_dir: Path = CACHE_DIR,
    model_name: str = "cnn",
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
) -> DataLoader:
    dataset = create_dataset(
        rows=rows,
        model_name=model_name,
        mean=mean,
        std=std,
        cache_dir=cache_dir,
        sequence_length=sequence_length,
    )
    if len(dataset) == 0:
        raise RuntimeError(
            f"No usable samples were created for model_name={model_name!r}. "
            f"Check the current split sparsity or reduce sequence_length={sequence_length}."
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=bool(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
        drop_last=False,
    )


def predict_with_model(
    model: nn.Module,
    rows: pd.DataFrame,
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
    batch_size: int,
    num_workers: int = 0,
    cache_dir: Path = CACHE_DIR,
    model_name: str = "cnn",
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
) -> pd.DataFrame:
    loader = create_dataloader(
        rows,
        batch_size=batch_size,
        shuffle=False,
        mean=mean,
        std=std,
        num_workers=num_workers,
        cache_dir=cache_dir,
        model_name=model_name,
        sequence_length=sequence_length,
    )
    predictions: list[pd.DataFrame] = []
    model.eval()
    with torch.no_grad():
        for inputs, labels, row_ids in loader:
            inputs = inputs.to(device, non_blocking=True)
            logits = model(inputs)
            scores = torch.sigmoid(logits).cpu().numpy()
            batch = pd.DataFrame(
                {
                    "row_id": row_ids.cpu().numpy(),
                    "y_true": labels.cpu().numpy().astype(int),
                    "y_score": scores,
                }
            )
            predictions.append(batch)
    return pd.concat(predictions, ignore_index=True).sort_values("row_id").reset_index(drop=True)


def train_model(
    train_rows: pd.DataFrame,
    val_rows: pd.DataFrame,
    output_dir: Path,
    device: str,
    seed: int,
    batch_size: int = 1024,
    num_workers: int = 2,
    max_epochs: int = 30,
    patience: int = 5,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    force: bool = False,
    model_name: str = "cnn",
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
    encoder_dim: int = DEFAULT_ENCODER_DIM,
    lstm_hidden_dim: int = DEFAULT_LSTM_HIDDEN_DIM,
    lstm_num_layers: int = DEFAULT_LSTM_NUM_LAYERS,
    lstm_dropout: float = DEFAULT_LSTM_DROPOUT,
) -> TrainingArtifacts:
    ensure_dir(output_dir)
    checkpoint_path = output_dir / "best.pt"
    log_path = output_dir / "train_log.csv"
    metrics_path = output_dir / "val_metrics.json"
    norm_stats_path = output_dir / "norm_stats.json"
    run_config_path = output_dir / "run_config.yaml"

    if checkpoint_path.exists() and log_path.exists() and metrics_path.exists() and norm_stats_path.exists() and not force:
        val_metrics = pd.read_json(metrics_path, typ="series")
        return TrainingArtifacts(
            checkpoint_path=checkpoint_path,
            log_path=log_path,
            metrics_path=metrics_path,
            norm_stats_path=norm_stats_path,
            threshold=float(val_metrics["threshold"]),
        )

    if model_name not in MODEL_NAMES:
        raise ValueError(f"Unsupported model_name={model_name!r}. Expected one of {MODEL_NAMES}.")

    set_seed(seed)
    torch_device = torch.device(device)
    mean, std = compute_channel_norm_stats(train_rows)
    model_kwargs = {"encoder_dim": int(encoder_dim)}
    if model_name == "lstm":
        model_kwargs.update(
            {
                "sequence_length": int(sequence_length),
                "hidden_dim": int(lstm_hidden_dim),
                "num_layers": int(lstm_num_layers),
                "dropout": float(lstm_dropout),
            }
        )

    write_json(norm_stats_path, {"mean": mean.tolist(), "std": std.tolist()})
    train_dataset = create_dataset(
        rows=train_rows,
        model_name=model_name,
        mean=mean,
        std=std,
        cache_dir=CACHE_DIR,
        sequence_length=sequence_length,
    )
    val_dataset = create_dataset(
        rows=val_rows,
        model_name=model_name,
        mean=mean,
        std=std,
        cache_dir=CACHE_DIR,
        sequence_length=sequence_length,
    )
    write_yaml(
        run_config_path,
        {
            "seed": seed,
            "model_name": model_name,
            "batch_size": batch_size,
            "max_epochs": max_epochs,
            "patience": patience,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "device": str(torch_device),
            "train_rows": int(len(train_rows)),
            "val_rows": int(len(val_rows)),
            "train_samples": int(len(train_dataset)),
            "val_samples": int(len(val_dataset)),
            "model_kwargs": model_kwargs,
        },
    )

    train_loader = create_dataloader(
        train_rows,
        batch_size=batch_size,
        shuffle=True,
        mean=mean,
        std=std,
        num_workers=num_workers,
        cache_dir=CACHE_DIR,
        model_name=model_name,
        sequence_length=sequence_length,
    )
    model = build_model(model_name, **model_kwargs).to(torch_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    best_auc = -np.inf
    best_state: dict[str, torch.Tensor] | None = None
    best_threshold = 0.5
    best_history_index: int | None = None
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        running_loss = 0.0
        total_examples = 0
        for inputs, labels, _ in train_loader:
            inputs = inputs.to(torch_device, non_blocking=True)
            labels = labels.to(torch_device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            batch_size_now = inputs.size(0)
            running_loss += float(loss.item()) * batch_size_now
            total_examples += batch_size_now

        train_loss = running_loss / max(total_examples, 1)
        val_predictions = predict_with_model(
            model=model,
            rows=val_rows,
            mean=mean,
            std=std,
            device=torch_device,
            batch_size=batch_size,
            num_workers=num_workers,
            cache_dir=CACHE_DIR,
            model_name=model_name,
            sequence_length=sequence_length,
        )
        threshold = select_best_threshold(val_predictions["y_true"].to_numpy(), val_predictions["y_score"].to_numpy())
        val_metrics = compute_binary_metrics(val_predictions["y_true"].to_numpy(), val_predictions["y_score"].to_numpy(), threshold)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_auc": val_metrics["roc_auc"],
                "val_f1": val_metrics["f1"],
                "val_precision": val_metrics["precision"],
                "val_recall": val_metrics["recall"],
            }
        )
        selection_metric = float(val_metrics["roc_auc"])
        if np.isnan(selection_metric):
            selection_metric = float(val_metrics["f1"])
        if best_state is None or selection_metric > best_auc:
            best_auc = selection_metric
            best_threshold = threshold
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_history_index = len(history) - 1
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            break

    history_df = pd.DataFrame(history)
    history_df.to_csv(log_path, index=False)
    if best_state is None:
        raise RuntimeError("Training finished without a best checkpoint.")

    torch.save(
        {
            "model_name": model_name,
            "model_kwargs": model_kwargs,
            "model_state_dict": best_state,
            "mean": mean,
            "std": std,
            "threshold": best_threshold,
            "seed": seed,
        },
        checkpoint_path,
    )

    final_metrics = history_df.iloc[best_history_index if best_history_index is not None else 0].to_dict()
    final_metrics["threshold"] = float(best_threshold)
    final_metrics["model_name"] = model_name
    write_json(metrics_path, final_metrics)
    return TrainingArtifacts(
        checkpoint_path=checkpoint_path,
        log_path=log_path,
        metrics_path=metrics_path,
        norm_stats_path=norm_stats_path,
        threshold=float(best_threshold),
    )


def load_trained_model(checkpoint_path: Path, device: str) -> tuple[nn.Module, np.ndarray, np.ndarray, float]:
    payload = torch.load(checkpoint_path, map_location=device)
    model_name = str(payload.get("model_name", "cnn"))
    model_kwargs = dict(payload.get("model_kwargs", {}))
    model = build_model(model_name, **model_kwargs)
    model.load_state_dict(payload["model_state_dict"])
    model.to(device)
    model.eval()
    mean = np.asarray(payload["mean"], dtype=np.float32)
    std = np.asarray(payload["std"], dtype=np.float32)
    threshold = float(payload["threshold"])
    return model, mean, std, threshold
