"""Modal app for GPU-accelerated Whisper feature extraction.

Usage:
    # Deploy and run extraction
    modal run modal_app/extract_features.py

    # Extract with custom config
    modal run modal_app/extract_features.py --config configs/tiny_default.yaml

    # Extract specific layers only
    modal run modal_app/extract_features.py --encoder-layers 0,1,2,3 --decoder-layers ""
"""

import modal

# Create Modal app
app = modal.App("whisper-sae-extract")

# Define image with dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.1.0",
        "torchaudio>=2.1.0",
        "transformers>=4.36.0",
        "datasets>=2.16.0",
        "einops>=0.7.0",
        "pydantic>=2.5.0",
        "pyyaml>=6.0.0",
        "rich>=13.7.0",
        "soundfile>=0.13.1",
    )
    .run_commands("apt-get update && apt-get install -y libsndfile1")
)

# Persistent volume for caching features
volume = modal.Volume.from_name("whisper-sae-cache", create_if_missing=True)

CACHE_DIR = "/cache"


@app.function(
    image=image,
    gpu="A10G",
    timeout=3600 * 4,  # 4 hours max
    volumes={CACHE_DIR: volume},
)
def extract_features(
    model_name: str = "openai/whisper-tiny",
    encoder_layers: list[int] | None = None,
    decoder_layers: list[int] | None = None,
    max_samples: int = 100_000,
    batch_size: int = 16,
) -> dict:
    """Extract Whisper features and cache to volume.

    Args:
        model_name: HuggingFace model name.
        encoder_layers: Which encoder layers to extract.
        decoder_layers: Which decoder layers to extract.
        max_samples: Maximum samples to process.
        batch_size: Batch size for extraction.

    Returns:
        Dictionary with extraction stats.
    """
    import io
    import json
    from datetime import datetime
    from itertools import islice
    from pathlib import Path

    import soundfile as sf
    import torch
    from datasets import Audio, load_dataset
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    console = Console()

    # Set defaults
    if encoder_layers is None:
        encoder_layers = [0, 1, 2, 3]
    if decoder_layers is None:
        decoder_layers = [0, 1, 2, 3]

    console.print(f"[bold cyan]Whisper Feature Extraction[/bold cyan]")
    console.print(f"Model: {model_name}")
    console.print(f"Encoder layers: {encoder_layers}")
    console.print(f"Decoder layers: {decoder_layers}")
    console.print(f"Max samples: {max_samples}")

    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.print(f"Device: {device}")

    # Load model
    console.print("\n[bold]Loading Whisper model...[/bold]")
    model = WhisperForConditionalGeneration.from_pretrained(model_name).to(device)
    processor = WhisperProcessor.from_pretrained(model_name)
    model.eval()

    # Get layer norms for post-processing
    encoder_layer_norm = model.model.encoder.layer_norm
    decoder_layer_norm = model.model.decoder.layer_norm

    # Load dataset with audio decoding disabled
    console.print("\n[bold]Loading LibriSpeech...[/bold]")
    dataset = load_dataset(
        "librispeech_asr",
        "clean",
        split="train.100",
        streaming=True,
    ).cast_column("audio", Audio(decode=False))

    # Prepare cache directory
    cache_dir = Path(CACHE_DIR) / "features" / model_name.split("/")[-1]
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Initialize accumulators
    encoder_features: dict[int, list[torch.Tensor]] = {l: [] for l in encoder_layers}
    decoder_features: dict[int, list[torch.Tensor]] = {l: [] for l in decoder_layers}

    # Register hooks
    hooks = []
    encoder_outputs: dict[int, torch.Tensor] = {}
    decoder_outputs: dict[int, torch.Tensor] = {}

    def make_encoder_hook(layer_idx: int):
        def hook(module, input, output):
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output
            # Apply layer norm
            hidden = encoder_layer_norm(hidden)
            encoder_outputs[layer_idx] = hidden.detach().cpu()
        return hook

    def make_decoder_hook(layer_idx: int):
        def hook(module, input, output):
            hidden = output[0]
            hidden = decoder_layer_norm(hidden)
            decoder_outputs[layer_idx] = hidden.detach().cpu()
        return hook

    for layer_idx in encoder_layers:
        layer = model.model.encoder.layers[layer_idx]
        hooks.append(layer.register_forward_hook(make_encoder_hook(layer_idx)))

    for layer_idx in decoder_layers:
        layer = model.model.decoder.layers[layer_idx]
        hooks.append(layer.register_forward_hook(make_decoder_hook(layer_idx)))

    # Process samples
    num_processed = 0
    batch_inputs = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
    ) as progress:
        task = progress.add_task("[cyan]Extracting features...", total=max_samples)

        for sample in islice(dataset, max_samples):
            try:
                # Decode audio
                audio_bytes = sample["audio"]["bytes"]
                speech_array, sr = sf.read(io.BytesIO(audio_bytes))

                # Process through Whisper processor
                inputs = processor(
                    speech_array,
                    sampling_rate=16000,
                    return_tensors="pt",
                ).input_features

                batch_inputs.append(inputs)

                # Process batch
                if len(batch_inputs) >= batch_size:
                    batch = torch.cat(batch_inputs, dim=0).to(device)
                    batch_inputs = []

                    with torch.no_grad():
                        # Run encoder
                        encoder_out = model.model.encoder(batch)
                        encoder_hidden = encoder_out.last_hidden_state

                        # Run decoder with start token
                        if decoder_layers:
                            decoder_input_ids = torch.full(
                                (batch.size(0), 1),
                                model.config.decoder_start_token_id,
                                dtype=torch.long,
                                device=device,
                            )
                            _ = model.model.decoder(
                                input_ids=decoder_input_ids,
                                encoder_hidden_states=encoder_hidden,
                            )

                    # Collect from hooks
                    for layer_idx in encoder_layers:
                        if layer_idx in encoder_outputs:
                            # Flatten: [batch, seq, hidden] -> [batch*seq, hidden]
                            flat = encoder_outputs[layer_idx].view(-1, encoder_outputs[layer_idx].size(-1))
                            encoder_features[layer_idx].append(flat)

                    for layer_idx in decoder_layers:
                        if layer_idx in decoder_outputs:
                            flat = decoder_outputs[layer_idx].view(-1, decoder_outputs[layer_idx].size(-1))
                            decoder_features[layer_idx].append(flat)

                    # Clear hook outputs
                    encoder_outputs.clear()
                    decoder_outputs.clear()

                num_processed += 1
                progress.update(task, completed=num_processed)

            except Exception as e:
                console.print(f"[yellow]Error processing sample: {e}[/yellow]")
                continue

    # Process remaining batch
    if batch_inputs:
        batch = torch.cat(batch_inputs, dim=0).to(device)

        with torch.no_grad():
            encoder_out = model.model.encoder(batch)
            encoder_hidden = encoder_out.last_hidden_state

            if decoder_layers:
                decoder_input_ids = torch.full(
                    (batch.size(0), 1),
                    model.config.decoder_start_token_id,
                    dtype=torch.long,
                    device=device,
                )
                _ = model.model.decoder(
                    input_ids=decoder_input_ids,
                    encoder_hidden_states=encoder_hidden,
                )

        for layer_idx in encoder_layers:
            if layer_idx in encoder_outputs:
                flat = encoder_outputs[layer_idx].view(-1, encoder_outputs[layer_idx].size(-1))
                encoder_features[layer_idx].append(flat)

        for layer_idx in decoder_layers:
            if layer_idx in decoder_outputs:
                flat = decoder_outputs[layer_idx].view(-1, decoder_outputs[layer_idx].size(-1))
                decoder_features[layer_idx].append(flat)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    # Save features with intermediate checkpoints
    console.print("\n[bold]Saving features...[/bold]")
    stats = {"encoder": {}, "decoder": {}, "num_samples": num_processed}

    for layer_idx in encoder_layers:
        if encoder_features[layer_idx]:
            features = torch.cat(encoder_features[layer_idx], dim=0)
            path = cache_dir / f"encoder_layer{layer_idx}.pt"
            # Atomic save: write to temp then rename
            temp_path = cache_dir / f"encoder_layer{layer_idx}.pt.tmp"
            torch.save(features, temp_path)
            temp_path.rename(path)
            console.print(f"Saved encoder layer {layer_idx}: {features.shape}")
            stats["encoder"][layer_idx] = {
                "shape": list(features.shape),
                "path": str(path),
            }

    for layer_idx in decoder_layers:
        if decoder_features[layer_idx]:
            features = torch.cat(decoder_features[layer_idx], dim=0)
            path = cache_dir / f"decoder_layer{layer_idx}.pt"
            temp_path = cache_dir / f"decoder_layer{layer_idx}.pt.tmp"
            torch.save(features, temp_path)
            temp_path.rename(path)
            console.print(f"Saved decoder layer {layer_idx}: {features.shape}")
            stats["decoder"][layer_idx] = {
                "shape": list(features.shape),
                "path": str(path),
            }

    # Save metadata
    metadata = {
        "model_name": model_name,
        "encoder_layers": encoder_layers,
        "decoder_layers": decoder_layers,
        "num_samples": num_processed,
        "hidden_dim": model.config.d_model,
        "created_at": datetime.now().isoformat(),
        "stats": stats,
    }
    with open(cache_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Save extraction log for debugging
    log = {
        "model_name": model_name,
        "num_samples_requested": max_samples,
        "num_samples_processed": num_processed,
        "batch_size": batch_size,
        "encoder_layers": encoder_layers,
        "decoder_layers": decoder_layers,
        "completed_at": datetime.now().isoformat(),
    }
    with open(cache_dir / "extraction_log.json", "w") as f:
        json.dump(log, f, indent=2)

    # Commit volume
    volume.commit()

    console.print(f"\n[bold green]Extraction complete![/bold green]")
    console.print(f"Processed {num_processed} samples")
    console.print(f"Saved to: {cache_dir}")

    return stats


@app.local_entrypoint()
def main(
    model_name: str = "openai/whisper-tiny",
    encoder_layers: str = "0,1,2,3",
    decoder_layers: str = "0,1,2,3",
    max_samples: int = 100_000,
    batch_size: int = 16,
):
    """Run feature extraction on Modal.

    Args:
        model_name: HuggingFace model name.
        encoder_layers: Comma-separated encoder layer indices.
        decoder_layers: Comma-separated decoder layer indices.
        max_samples: Maximum samples to process.
        batch_size: Batch size for extraction.
    """
    # Parse layer lists
    enc_layers = [int(x) for x in encoder_layers.split(",") if x.strip()]
    dec_layers = [int(x) for x in decoder_layers.split(",") if x.strip()]

    print(f"Running feature extraction on Modal...")
    print(f"Model: {model_name}")
    print(f"Encoder layers: {enc_layers}")
    print(f"Decoder layers: {dec_layers}")

    stats = extract_features.remote(
        model_name=model_name,
        encoder_layers=enc_layers,
        decoder_layers=dec_layers,
        max_samples=max_samples,
        batch_size=batch_size,
    )

    print(f"\nExtraction complete!")
    print(f"Stats: {stats}")
