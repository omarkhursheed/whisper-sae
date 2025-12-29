"""Modal app for GPU-accelerated SAE training.

Usage:
    # Train SAE on all layers
    modal run modal_app/train.py

    # Train on specific layer
    modal run modal_app/train.py --component encoder --layer 0

    # Train with custom config
    modal run modal_app/train.py --config configs/tiny_default.yaml
"""

import modal

# Create Modal app
app = modal.App("whisper-sae-train")

# Define image with dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.1.0",
        "torchaudio>=2.1.0",
        "transformers>=4.36.0",
        "einops>=0.7.0",
        "pydantic>=2.5.0",
        "pyyaml>=6.0.0",
        "rich>=13.7.0",
        "wandb>=0.16.0",
    )
)

# Persistent volumes
cache_volume = modal.Volume.from_name("whisper-sae-cache", create_if_missing=True)
output_volume = modal.Volume.from_name("whisper-sae-outputs", create_if_missing=True)

CACHE_DIR = "/cache"
OUTPUT_DIR = "/outputs"


@app.function(
    image=image,
    gpu="A10G",
    timeout=3600 * 6,  # 6 hours max
    volumes={CACHE_DIR: cache_volume, OUTPUT_DIR: output_volume},
    secrets=[modal.Secret.from_name("wandb-secret", required=False)],
)
def train_sae(
    component: str = "encoder",
    layer_idx: int = 0,
    model_name: str = "openai/whisper-tiny",
    expansion_factor: int = 8,
    k: int = 32,
    batch_size: int = 128,
    learning_rate: float = 1e-4,
    epochs: int = 50,
    warmup_steps: int = 1000,
    checkpoint_every: int = 10,
    use_wandb: bool = True,
    experiment_name: str | None = None,
) -> dict:
    """Train SAE on cached features.

    Args:
        component: 'encoder' or 'decoder'.
        layer_idx: Layer index to train on.
        model_name: Model name (for loading cached features).
        expansion_factor: SAE expansion factor.
        k: Number of active features for TopK.
        batch_size: Training batch size.
        learning_rate: Learning rate.
        epochs: Number of epochs.
        warmup_steps: Warmup steps for scheduler.
        checkpoint_every: Checkpoint every N epochs.
        use_wandb: Whether to log to W&B.
        experiment_name: Name for this experiment.

    Returns:
        Training results dictionary.
    """
    import json
    import os
    from dataclasses import asdict
    from datetime import datetime
    from pathlib import Path

    import torch
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
    from torch.utils.data import DataLoader, TensorDataset

    console = Console()

    # Print banner
    console.print(Panel.fit(
        f"[bold cyan]SAE Training[/bold cyan]\n"
        f"{component} layer {layer_idx}",
        border_style="cyan",
    ))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.print(f"Device: {device}")

    # Load cached features
    model_short = model_name.split("/")[-1]
    cache_dir = Path(CACHE_DIR) / "features" / model_short
    feature_path = cache_dir / f"{component}_layer{layer_idx}.pt"

    if not feature_path.exists():
        raise FileNotFoundError(
            f"Features not found at {feature_path}. "
            "Run extract_features.py first."
        )

    console.print(f"\n[bold]Loading features from {feature_path}...[/bold]")
    features = torch.load(feature_path, weights_only=True)
    console.print(f"Loaded features: {features.shape}")

    input_dim = features.shape[1]
    hidden_dim = input_dim * expansion_factor

    # Create SAE model (inline to avoid import issues)
    class TopKSAE(torch.nn.Module):
        def __init__(self, input_dim, hidden_dim, k):
            super().__init__()
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim
            self.k = k

            self.encoder = torch.nn.Linear(input_dim, hidden_dim, bias=True)
            self.decoder = torch.nn.Linear(hidden_dim, input_dim, bias=True)
            self.b_pre = torch.nn.Parameter(torch.zeros(input_dim))

            # Initialize decoder with unit-norm columns
            with torch.no_grad():
                torch.nn.init.xavier_uniform_(self.decoder.weight)
                self.decoder.weight.data = torch.nn.functional.normalize(
                    self.decoder.weight.data, dim=0
                ) * 0.1

            # Dead feature tracking
            self.register_buffer("feature_last_activated", torch.zeros(hidden_dim, dtype=torch.long))
            self.register_buffer("step_count", torch.tensor(0, dtype=torch.long))

        def encode(self, x):
            x_centered = x - self.b_pre
            pre_activation = self.encoder(x_centered)
            topk_values, topk_indices = torch.topk(pre_activation, self.k, dim=-1)
            hidden = torch.zeros_like(pre_activation)
            hidden.scatter_(-1, topk_indices, torch.relu(topk_values))
            return hidden

        def decode(self, hidden):
            return self.decoder(hidden) + self.b_pre

        def forward(self, x):
            hidden = self.encode(x)
            reconstructed = self.decode(hidden)
            loss = torch.nn.functional.mse_loss(reconstructed, x)
            l0 = (hidden > 0).float().sum(dim=-1).mean()

            # Update dead feature tracking
            if self.training:
                self.step_count += 1
                active = (hidden > 0).any(dim=0)
                self.feature_last_activated[active] = self.step_count

            return {
                "reconstructed": reconstructed,
                "hidden": hidden,
                "loss": loss,
                "l0": l0,
            }

        def normalize_decoder(self):
            with torch.no_grad():
                self.decoder.weight.data = torch.nn.functional.normalize(
                    self.decoder.weight.data, dim=0
                )

        def get_dead_ratio(self, threshold=10000):
            steps_since = self.step_count - self.feature_last_activated
            return (steps_since > threshold).float().mean().item()

    # Create model
    sae = TopKSAE(input_dim, hidden_dim, k).to(device)
    console.print(f"Created SAE: {input_dim} -> {hidden_dim} (k={k})")

    # Create dataloader
    dataset = TensorDataset(features)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    # Set up optimizer and scheduler
    optimizer = torch.optim.AdamW(sae.parameters(), lr=learning_rate)
    total_steps = len(dataloader) * epochs

    # Set up AMP for CUDA
    use_amp = torch.cuda.is_available()
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    console.print(f"Mixed precision (AMP): {use_amp}")

    # Warmup + cosine decay
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Set up W&B
    run = None
    if use_wandb:
        try:
            import wandb
            exp_name = experiment_name or f"{model_short}_{component}_layer{layer_idx}"
            run = wandb.init(
                project="whisper-sae",
                name=exp_name,
                config={
                    "model": model_name,
                    "component": component,
                    "layer": layer_idx,
                    "expansion_factor": expansion_factor,
                    "k": k,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "epochs": epochs,
                },
            )
        except Exception as e:
            console.print(f"[yellow]W&B init failed: {e}[/yellow]")
            run = None

    # Set up output directory
    exp_name = experiment_name or f"{model_short}_{component}_layer{layer_idx}"
    output_dir = Path(OUTPUT_DIR) / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    console.print(f"\n[bold]Training for {epochs} epochs...[/bold]")
    metrics_history = []

    for epoch in range(epochs):
        sae.train()
        epoch_loss = 0
        epoch_l0 = 0
        num_batches = 0

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
        ) as progress:
            task = progress.add_task(f"[cyan]Epoch {epoch + 1}/{epochs}", total=len(dataloader))

            for batch in dataloader:
                batch = batch[0].to(device)

                optimizer.zero_grad()

                # Forward pass with AMP
                if use_amp:
                    with torch.amp.autocast("cuda"):
                        output = sae(batch)
                        loss = output["loss"]
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    output = sae(batch)
                    loss = output["loss"]
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0)
                    optimizer.step()

                scheduler.step()

                # Normalize decoder
                sae.normalize_decoder()

                epoch_loss += loss.item()
                epoch_l0 += output["l0"].item()
                num_batches += 1

                progress.update(task, advance=1)

        # Epoch stats
        avg_loss = epoch_loss / num_batches
        avg_l0 = epoch_l0 / num_batches
        dead_ratio = sae.get_dead_ratio()
        current_lr = scheduler.get_last_lr()[0]

        metrics = {
            "epoch": epoch + 1,
            "loss": avg_loss,
            "l0": avg_l0,
            "dead_ratio": dead_ratio,
            "lr": current_lr,
        }
        metrics_history.append(metrics)

        console.print(
            f"Epoch {epoch + 1}: loss={avg_loss:.4f}, L0={avg_l0:.1f}, "
            f"dead={dead_ratio * 100:.1f}%, lr={current_lr:.2e}"
        )

        # Log to W&B
        if run is not None:
            run.log(metrics)

        # Checkpoint with atomic save
        if (epoch + 1) % checkpoint_every == 0:
            checkpoint_path = output_dir / f"checkpoint_epoch{epoch + 1}.pt"
            temp_path = output_dir / f"checkpoint_epoch{epoch + 1}.pt.tmp"
            checkpoint_data = {
                "model_state_dict": sae.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "epoch": epoch + 1,
                "metrics": metrics,
                "config": {
                    "input_dim": input_dim,
                    "hidden_dim": hidden_dim,
                    "k": k,
                    "component": component,
                    "layer_idx": layer_idx,
                    "model_name": model_name,
                },
            }
            if scaler is not None:
                checkpoint_data["scaler_state_dict"] = scaler.state_dict()
            torch.save(checkpoint_data, temp_path)
            temp_path.rename(checkpoint_path)
            console.print(f"[green]Saved checkpoint: {checkpoint_path}[/green]")
            # Commit volume after each checkpoint
            output_volume.commit()

    # Save final model with full metadata (atomic)
    final_path = output_dir / "sae_final.pt"
    temp_path = output_dir / "sae_final.pt.tmp"
    final_data = {
        "model_state_dict": sae.state_dict(),
        "config": {
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "k": k,
            "component": component,
            "layer_idx": layer_idx,
            "model_name": model_name,
            "expansion_factor": expansion_factor,
        },
        "training": {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "warmup_steps": warmup_steps,
            "final_loss": metrics_history[-1]["loss"],
            "final_l0": metrics_history[-1]["l0"],
            "final_dead_ratio": metrics_history[-1]["dead_ratio"],
        },
    }
    torch.save(final_data, temp_path)
    temp_path.rename(final_path)
    console.print(f"[green]Saved final model: {final_path}[/green]")

    # Save metrics
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_history, f, indent=2)
    console.print(f"[green]Saved metrics: {metrics_path}[/green]")

    # Save training config for reproducibility
    config_path = output_dir / "training_config.json"
    training_config = {
        "model_name": model_name,
        "component": component,
        "layer_idx": layer_idx,
        "input_dim": input_dim,
        "hidden_dim": hidden_dim,
        "expansion_factor": expansion_factor,
        "k": k,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "epochs": epochs,
        "warmup_steps": warmup_steps,
        "checkpoint_every": checkpoint_every,
        "use_wandb": use_wandb,
        "experiment_name": exp_name,
        "completed_at": datetime.now().isoformat(),
    }
    with open(config_path, "w") as f:
        json.dump(training_config, f, indent=2)
    console.print(f"[green]Saved training config: {config_path}[/green]")

    # Commit volumes
    output_volume.commit()

    # Close W&B
    if run is not None:
        run.finish()

    console.print(f"\n[bold green]Training complete![/bold green]")

    return {
        "component": component,
        "layer": layer_idx,
        "final_loss": metrics_history[-1]["loss"],
        "final_l0": metrics_history[-1]["l0"],
        "final_dead_ratio": metrics_history[-1]["dead_ratio"],
        "output_dir": str(output_dir),
    }


@app.function(
    image=image,
    gpu="A10G",
    timeout=3600 * 24,  # 24 hours for all layers
    volumes={CACHE_DIR: cache_volume, OUTPUT_DIR: output_volume},
    secrets=[modal.Secret.from_name("wandb-secret", required=False)],
)
def train_all_layers(
    model_name: str = "openai/whisper-tiny",
    encoder_layers: list[int] | None = None,
    decoder_layers: list[int] | None = None,
    **kwargs,
) -> list[dict]:
    """Train SAEs on all specified layers sequentially.

    Args:
        model_name: Model name.
        encoder_layers: Encoder layers to train.
        decoder_layers: Decoder layers to train.
        **kwargs: Additional arguments passed to train_sae.

    Returns:
        List of results for each layer.
    """
    from rich.console import Console

    console = Console()

    if encoder_layers is None:
        encoder_layers = [0, 1, 2, 3]
    if decoder_layers is None:
        decoder_layers = [0, 1, 2, 3]

    results = []

    # Train encoder layers
    for layer_idx in encoder_layers:
        console.print(f"\n[bold]Training encoder layer {layer_idx}...[/bold]")
        result = train_sae.local(
            component="encoder",
            layer_idx=layer_idx,
            model_name=model_name,
            **kwargs,
        )
        results.append(result)

    # Train decoder layers
    for layer_idx in decoder_layers:
        console.print(f"\n[bold]Training decoder layer {layer_idx}...[/bold]")
        result = train_sae.local(
            component="decoder",
            layer_idx=layer_idx,
            model_name=model_name,
            **kwargs,
        )
        results.append(result)

    console.print(f"\n[bold green]All training complete![/bold green]")
    return results


@app.local_entrypoint()
def main(
    component: str = "encoder",
    layer: int = 0,
    model_name: str = "openai/whisper-tiny",
    expansion_factor: int = 8,
    k: int = 32,
    batch_size: int = 128,
    learning_rate: float = 1e-4,
    epochs: int = 50,
    all_layers: bool = False,
    no_wandb: bool = False,
):
    """Train SAE on Modal.

    Args:
        component: 'encoder' or 'decoder'.
        layer: Layer index.
        model_name: HuggingFace model name.
        expansion_factor: SAE expansion factor.
        k: Number of active features.
        batch_size: Training batch size.
        learning_rate: Learning rate.
        epochs: Number of epochs.
        all_layers: Train all encoder and decoder layers.
        no_wandb: Disable W&B logging.
    """
    print(f"Running SAE training on Modal...")

    if all_layers:
        results = train_all_layers.remote(
            model_name=model_name,
            expansion_factor=expansion_factor,
            k=k,
            batch_size=batch_size,
            learning_rate=learning_rate,
            epochs=epochs,
            use_wandb=not no_wandb,
        )
        print(f"\nAll layers trained!")
        for r in results:
            print(f"  {r['component']} layer {r['layer']}: loss={r['final_loss']:.4f}")
    else:
        result = train_sae.remote(
            component=component,
            layer_idx=layer,
            model_name=model_name,
            expansion_factor=expansion_factor,
            k=k,
            batch_size=batch_size,
            learning_rate=learning_rate,
            epochs=epochs,
            use_wandb=not no_wandb,
        )
        print(f"\nTraining complete!")
        print(f"Results: {result}")
