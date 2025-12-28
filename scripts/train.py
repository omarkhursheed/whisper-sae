#!/usr/bin/env python3
"""Main training script for Whisper SAE.

Usage:
    # Train with default config
    uv run python scripts/train.py

    # Train with custom config
    uv run python scripts/train.py --config configs/tiny_default.yaml

    # Train single layer
    uv run python scripts/train.py --layer encoder:0

    # Train with W&B disabled (for local testing)
    uv run python scripts/train.py --no-wandb

    # Extract features only (no training)
    uv run python scripts/train.py --extract-only
"""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from rich.console import Console
from rich.panel import Panel
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from whisper_sae.config import ExperimentConfig, LayerConfig
from whisper_sae.data.feature_cache import FeatureCache, extract_and_cache_features
from whisper_sae.data.librispeech import LibriSpeechDataset, LibriSpeechFeaturesOnly
from whisper_sae.sae.model import create_sae
from whisper_sae.sae.training import SAETrainer

console = Console()


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train Sparse Autoencoders on Whisper activations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/tiny_default.yaml"),
        help="Path to configuration YAML file",
    )
    parser.add_argument(
        "--layer",
        type=str,
        default=None,
        help="Train single layer (format: encoder:0 or decoder:2)",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable W&B logging",
    )
    parser.add_argument(
        "--extract-only",
        action="store_true",
        help="Extract features only, don't train SAEs",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda, mps, cpu). Auto-detected if not specified.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (overrides config)",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device_arg: str | None) -> torch.device:
    """Get the appropriate device."""
    if device_arg:
        return torch.device(device_arg)

    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def parse_layer_arg(layer_arg: str) -> tuple[str, int]:
    """Parse layer argument like 'encoder:0' or 'decoder:2'."""
    parts = layer_arg.split(":")
    if len(parts) != 2:
        raise ValueError(f"Invalid layer format: {layer_arg}. Use encoder:N or decoder:N")
    component = parts[0]
    if component not in ("encoder", "decoder"):
        raise ValueError(f"Invalid component: {component}. Use encoder or decoder")
    layer_idx = int(parts[1])
    return component, layer_idx


def train_layer(
    config: ExperimentConfig,
    component: str,
    layer_idx: int,
    feature_cache: FeatureCache,
    device: torch.device,
) -> None:
    """Train SAE for a single layer.

    Args:
        config: Experiment configuration.
        component: 'encoder' or 'decoder'.
        layer_idx: Layer index.
        feature_cache: Feature cache for loading activations.
        device: Device to train on.
    """
    console.print(f"\n[bold blue]Training SAE for {component} layer {layer_idx}[/bold blue]")

    # Check if features are cached
    if not feature_cache.has_cache(component, layer_idx):
        console.print(f"[red]No cached features found for {component} layer {layer_idx}[/red]")
        console.print("Run with --extract-only first to extract features")
        return

    # Load cached features
    features, metadata = feature_cache.load(component, layer_idx)
    console.print(
        f"Loaded {features.shape[0]:,} tokens, dim={features.shape[1]}"
    )

    # Create SAE
    input_dim = features.shape[1]
    sae = create_sae(config.sae, input_dim)
    console.print(
        f"Created SAE: {input_dim} -> {sae.hidden_dim} (k={config.sae.k})"
    )

    # Create dataloader
    dataloader = feature_cache.get_dataloader(
        component=component,
        layer_idx=layer_idx,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
    )

    # Create run directory
    run_name = f"{config.experiment_name}_{component}_layer{layer_idx}"
    run_dir = config.output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Create trainer
    trainer = SAETrainer(
        model=sae,
        config=config.training,
        device=device,
        run_dir=run_dir,
    )

    # Set up resampling dataset
    from torch.utils.data import TensorDataset
    resample_dataset = TensorDataset(features)
    trainer.set_resample_dataset(resample_dataset)

    # Set up W&B if enabled
    if config.wandb.enabled:
        try:
            import wandb

            trainer.wandb_run = wandb.init(
                project=config.wandb.project,
                entity=config.wandb.entity,
                name=run_name,
                tags=config.wandb.tags + [component, f"layer{layer_idx}"],
                config={
                    "whisper": config.whisper.model_dump(),
                    "sae": config.sae.model_dump(),
                    "training": config.training.model_dump(),
                    "component": component,
                    "layer_idx": layer_idx,
                },
            )
        except Exception as e:
            console.print(f"[yellow]W&B initialization failed: {e}[/yellow]")
            console.print("Continuing without W&B logging...")

    # Train
    console.print(f"Training for {config.training.epochs} epochs...")
    trainer.train(dataloader, epochs=config.training.epochs)

    # Save final model
    final_path = run_dir / "sae_final.pt"
    torch.save(sae.state_dict(), final_path)
    console.print(f"[green]Saved model to {final_path}[/green]")

    # Save metrics
    trainer.save_metrics()
    console.print(f"[green]Saved metrics to {run_dir / 'metrics.json'}[/green]")

    # Clean up W&B
    if trainer.wandb_run is not None:
        trainer.wandb_run.finish()


def main() -> None:
    """Main entry point."""
    args = parse_args()

    # Print banner
    console.print(Panel.fit(
        "[bold cyan]Whisper SAE Training[/bold cyan]\n"
        "Train Sparse Autoencoders on Whisper activations",
        border_style="cyan",
    ))

    # Load configuration
    if args.config.exists():
        config = ExperimentConfig.from_yaml(args.config)
        console.print(f"Loaded config from {args.config}")
    else:
        config = ExperimentConfig()
        console.print("Using default configuration")

    # Override from command line
    if args.seed is not None:
        config.training.seed = args.seed
    if args.no_wandb:
        config.wandb.enabled = False

    # Set up
    set_seed(config.training.seed)
    device = get_device(args.device)
    console.print(f"Using device: {device}")

    # Print configuration summary
    console.print(Panel.fit(
        f"Model: {config.whisper.model_name}\n"
        f"Encoder layers: {config.encoder_layers}\n"
        f"Decoder layers: {config.decoder_layers}\n"
        f"SAE expansion: {config.sae.expansion_factor}x, k={config.sae.k}\n"
        f"Batch size: {config.training.batch_size}\n"
        f"Epochs: {config.training.epochs}\n"
        f"W&B: {'enabled' if config.wandb.enabled else 'disabled'}",
        title="Configuration",
        border_style="blue",
    ))

    # Load Whisper model and processor
    console.print("\n[bold]Loading Whisper model...[/bold]")
    whisper_model = WhisperForConditionalGeneration.from_pretrained(
        config.whisper.model_name
    ).to(device)
    processor = WhisperProcessor.from_pretrained(config.whisper.model_name)
    whisper_model.eval()
    console.print(f"Loaded {config.whisper.model_name}")

    # Set up feature cache
    cache_dir = Path(config.data.cache_dir) / "features"
    feature_cache = FeatureCache(
        cache_dir=cache_dir,
        whisper_config=config.whisper,
        data_config=config.data,
    )

    # Determine which layers to process
    encoder_layers = config.encoder_layers
    decoder_layers = config.decoder_layers

    if args.layer:
        component, layer_idx = parse_layer_arg(args.layer)
        if component == "encoder":
            encoder_layers = [layer_idx]
            decoder_layers = []
        else:
            encoder_layers = []
            decoder_layers = [layer_idx]

    # Check if we need to extract features
    need_extraction = False
    for layer in encoder_layers:
        if not feature_cache.has_cache("encoder", layer):
            need_extraction = True
            break
    for layer in decoder_layers:
        if not feature_cache.has_cache("decoder", layer):
            need_extraction = True
            break

    if need_extraction or args.extract_only:
        console.print("\n[bold]Extracting features...[/bold]")

        # Load LibriSpeech
        librispeech = LibriSpeechDataset(processor, config.data)
        audio_dataloader = torch.utils.data.DataLoader(
            LibriSpeechFeaturesOnly(librispeech),
            batch_size=16,  # Smaller batch for extraction
            shuffle=False,
            num_workers=config.training.num_workers,
            pin_memory=True,
        )

        # Extract and cache
        extract_and_cache_features(
            whisper_model=whisper_model,
            processor=processor,
            audio_dataloader=audio_dataloader,
            cache=feature_cache,
            encoder_layers=encoder_layers,
            decoder_layers=decoder_layers,
            device=device,
            max_samples=config.data.max_samples,
        )

        console.print("[green]Feature extraction complete[/green]")

    if args.extract_only:
        console.print("\n[yellow]Extract-only mode, skipping training[/yellow]")
        return

    # Train SAEs for each layer
    for layer_idx in encoder_layers:
        train_layer(config, "encoder", layer_idx, feature_cache, device)

    for layer_idx in decoder_layers:
        train_layer(config, "decoder", layer_idx, feature_cache, device)

    console.print("\n[bold green]Training complete![/bold green]")


if __name__ == "__main__":
    main()
