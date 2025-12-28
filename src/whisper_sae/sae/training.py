"""SAE training loop with logging and checkpointing."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import torch
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from ..config import TrainingConfig
from .model import TopKSAE, SAEOutput


@dataclass
class TrainingMetrics:
    """Metrics from a training step or epoch."""

    loss: float
    reconstruction_loss: float
    sparsity_loss: float
    l0: float
    dead_feature_ratio: float
    learning_rate: float
    step: int


class SAETrainer:
    """Trainer for Sparse Autoencoders."""

    def __init__(
        self,
        model: TopKSAE,
        config: TrainingConfig,
        device: torch.device | str = "cpu",
        run_dir: Path | None = None,
        resample_dead_every: int = 5000,
        resample_batch_size: int = 8192,
    ):
        """Initialize trainer.

        Args:
            model: The SAE model to train.
            config: Training configuration.
            device: Device to train on.
            run_dir: Directory for checkpoints and logs.
            resample_dead_every: Resample dead features every N steps.
            resample_batch_size: Batch size for resampling (need many examples).
        """
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.run_dir = run_dir or Path("outputs")
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.resample_dead_every = resample_dead_every
        self.resample_batch_size = resample_batch_size

        # Set up optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Scheduler will be set up when we know the total steps
        self.scheduler = None

        # Mixed precision
        self.scaler = torch.amp.GradScaler("cuda", enabled=config.use_amp and device != "cpu")
        self.use_amp = config.use_amp and str(device) != "cpu"

        # Tracking
        self.global_step = 0
        self.epoch = 0
        self.metrics_history: list[TrainingMetrics] = []
        self.num_resampled_total = 0

        # W&B (optional)
        self.wandb_run = None

        # Store reference to full dataset for resampling
        self._resample_dataset = None

    def set_resample_dataset(self, dataset: torch.utils.data.Dataset) -> None:
        """Set the dataset used for dead feature resampling.

        Args:
            dataset: Dataset of activation tensors for sampling high-error examples.
        """
        self._resample_dataset = dataset

    def _maybe_resample_dead_features(self) -> int:
        """Resample dead features if conditions are met.

        Returns:
            Number of features resampled.
        """
        if self._resample_dataset is None:
            return 0

        if not hasattr(self.model, "resample_dead_features"):
            return 0

        if self.global_step % self.resample_dead_every != 0:
            return 0

        if self.global_step == 0:
            return 0

        # Sample a batch from the dataset for resampling
        indices = torch.randperm(len(self._resample_dataset))[: self.resample_batch_size]
        samples = [self._resample_dataset[i] for i in indices]

        # Handle TensorDataset which returns tuples
        if isinstance(samples[0], tuple):
            samples = [s[0] for s in samples]

        resample_batch = torch.stack(samples).to(self.device)

        num_resampled = self.model.resample_dead_features(resample_batch)
        self.num_resampled_total += num_resampled

        if num_resampled > 0 and self.wandb_run is not None:
            self.wandb_run.log(
                {"train/features_resampled": num_resampled},
                step=self.global_step,
            )

        return num_resampled

    def setup_scheduler(self, total_steps: int) -> None:
        """Set up learning rate scheduler.

        Args:
            total_steps: Total number of training steps.
        """
        warmup_steps = min(self.config.warmup_steps, total_steps // 10)

        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        cosine_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps - warmup_steps,
            eta_min=self.config.learning_rate * 0.1,
        )
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps],
        )

    def train_step(self, batch: Tensor) -> TrainingMetrics:
        """Run a single training step.

        Args:
            batch: Batch of activations [batch, input_dim].

        Returns:
            Training metrics for this step.
        """
        self.model.train()
        batch = batch.to(self.device)

        # Forward pass with AMP
        with torch.amp.autocast("cuda", enabled=self.use_amp):
            output: SAEOutput = self.model(batch)

        # Backward pass
        self.optimizer.zero_grad()
        self.scaler.scale(output.loss).backward()

        # Gradient clipping
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config.gradient_clip,
        )

        self.scaler.step(self.optimizer)
        self.scaler.update()

        # Normalize decoder weights after optimizer step
        if hasattr(self.model, "normalize_decoder_weights"):
            self.model.normalize_decoder_weights()

        # Update scheduler
        if self.scheduler is not None:
            self.scheduler.step()

        self.global_step += 1

        # Get metrics
        metrics = TrainingMetrics(
            loss=output.loss.item(),
            reconstruction_loss=output.reconstruction_loss.item(),
            sparsity_loss=output.sparsity_loss.item(),
            l0=output.l0.item(),
            dead_feature_ratio=self.model.get_dead_feature_ratio(),
            learning_rate=self.optimizer.param_groups[0]["lr"],
            step=self.global_step,
        )

        return metrics

    def train_epoch(
        self,
        dataloader: DataLoader,
        progress: Progress | None = None,
        task_id: int | None = None,
    ) -> list[TrainingMetrics]:
        """Train for one epoch.

        Args:
            dataloader: DataLoader with activation batches.
            progress: Rich progress bar (optional).
            task_id: Progress task ID (optional).

        Returns:
            List of metrics for each step.
        """
        epoch_metrics = []

        for batch in dataloader:
            metrics = self.train_step(batch)
            epoch_metrics.append(metrics)
            self.metrics_history.append(metrics)

            if progress is not None and task_id is not None:
                progress.update(task_id, advance=1)

            # Log to wandb
            if self.wandb_run is not None and self.global_step % 100 == 0:
                self.wandb_run.log(
                    {
                        "train/loss": metrics.loss,
                        "train/reconstruction_loss": metrics.reconstruction_loss,
                        "train/l0": metrics.l0,
                        "train/dead_ratio": metrics.dead_feature_ratio,
                        "train/lr": metrics.learning_rate,
                    },
                    step=self.global_step,
                )

        self.epoch += 1
        return epoch_metrics

    def train(
        self,
        dataloader: DataLoader,
        epochs: int | None = None,
        checkpoint_every: int | None = None,
    ) -> None:
        """Full training loop.

        Args:
            dataloader: DataLoader with activation batches.
            epochs: Number of epochs (uses config if not provided).
            checkpoint_every: Checkpoint frequency (uses config if not provided).
        """
        epochs = epochs or self.config.epochs
        checkpoint_every = checkpoint_every or self.config.checkpoint_every

        total_steps = len(dataloader) * epochs
        self.setup_scheduler(total_steps)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
        ) as progress:
            epoch_task = progress.add_task(
                f"[cyan]Training {epochs} epochs",
                total=epochs,
            )

            for epoch in range(epochs):
                step_task = progress.add_task(
                    f"[green]Epoch {epoch + 1}/{epochs}",
                    total=len(dataloader),
                )

                epoch_metrics = self.train_epoch(dataloader, progress, step_task)

                # Compute epoch averages
                avg_loss = sum(m.loss for m in epoch_metrics) / len(epoch_metrics)
                avg_l0 = sum(m.l0 for m in epoch_metrics) / len(epoch_metrics)
                dead_ratio = epoch_metrics[-1].dead_feature_ratio

                progress.remove_task(step_task)
                progress.update(epoch_task, advance=1)
                progress.console.print(
                    f"Epoch {epoch + 1}: loss={avg_loss:.4f}, L0={avg_l0:.1f}, "
                    f"dead={dead_ratio:.1%}"
                )

                # Checkpoint
                if (epoch + 1) % checkpoint_every == 0:
                    self.save_checkpoint(f"checkpoint_epoch{epoch + 1}.pt")

        # Final checkpoint
        self.save_checkpoint("final.pt")

    def save_checkpoint(self, filename: str) -> Path:
        """Save a training checkpoint.

        Args:
            filename: Name of the checkpoint file.

        Returns:
            Path to saved checkpoint.
        """
        path = self.run_dir / filename
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
                "global_step": self.global_step,
                "epoch": self.epoch,
                "config": self.config.model_dump(),
            },
            path,
        )
        return path

    def load_checkpoint(self, path: str | Path) -> None:
        """Load a training checkpoint.

        Args:
            path: Path to checkpoint file.
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if checkpoint["scheduler_state_dict"] and self.scheduler:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.global_step = checkpoint["global_step"]
        self.epoch = checkpoint["epoch"]

    def save_metrics(self, filename: str = "metrics.json") -> Path:
        """Save training metrics to JSON.

        Args:
            filename: Name of the metrics file.

        Returns:
            Path to saved metrics.
        """
        path = self.run_dir / filename
        metrics_dicts = [
            {
                "step": m.step,
                "loss": m.loss,
                "reconstruction_loss": m.reconstruction_loss,
                "sparsity_loss": m.sparsity_loss,
                "l0": m.l0,
                "dead_feature_ratio": m.dead_feature_ratio,
                "learning_rate": m.learning_rate,
            }
            for m in self.metrics_history
        ]
        with open(path, "w") as f:
            json.dump(metrics_dicts, f, indent=2)
        return path
