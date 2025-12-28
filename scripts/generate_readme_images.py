"""Generate images for README from Whisper analysis."""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import soundfile as sf
import io
from pathlib import Path

import datasets as ds
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from jiwer import wer

# Setup
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('husl')
output_dir = Path('docs/images')
output_dir.mkdir(parents=True, exist_ok=True)

device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

# Load model with eager attention
print('Loading Whisper...')
model_name = 'openai/whisper-tiny'
model = WhisperForConditionalGeneration.from_pretrained(
    model_name,
    attn_implementation='eager',
).to(device)
processor = WhisperProcessor.from_pretrained(model_name)
model.eval()

# Load sample
print('Loading audio sample...')
dataset = ds.load_dataset(
    'librispeech_asr',
    'clean',
    split='validation',
    streaming=True,
).cast_column('audio', ds.Audio(decode=False))

raw_samples = list(dataset.take(1))
audio_bytes = raw_samples[0]['audio']['bytes']
audio_array, sr = sf.read(io.BytesIO(audio_bytes))
ground_truth = raw_samples[0]['text']

# Process audio
if sr != 16000:
    import torchaudio
    resampler = torchaudio.transforms.Resample(sr, 16000)
    audio_array = resampler(torch.from_numpy(audio_array).float()).numpy()

inputs = processor(audio_array, sampling_rate=16000, return_tensors='pt')
input_features = inputs.input_features.to(device)

# Generate and get outputs
print('Running inference...')
with torch.no_grad():
    generated = model.generate(input_features, max_new_tokens=128)
    encoder_outputs = model.model.encoder(
        input_features,
        output_hidden_states=True,
        output_attentions=True,
        return_dict=True,
    )

generated_text = processor.batch_decode(generated, skip_special_tokens=True)[0]
print(f'Generated: "{generated_text}"')
print(f'Ground truth: "{ground_truth}"')

# 1. Encoder Self-Attention
print('Generating encoder attention plot...')
encoder_attentions = encoder_outputs.attentions

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for layer_idx in range(len(encoder_attentions)):
    attn = encoder_attentions[layer_idx][0]
    mean_attn = attn.mean(dim=0).cpu().numpy()
    step = max(1, mean_attn.shape[0] // 100)
    mean_attn = mean_attn[::step, ::step]

    ax = axes[layer_idx]
    sns.heatmap(mean_attn, cmap='Blues', ax=ax, cbar=False)
    ax.set_title(f'Encoder Layer {layer_idx}')
    ax.set_xlabel('Key')
    ax.set_ylabel('Query')

plt.suptitle('Encoder Self-Attention (Mean over Heads)', fontsize=14)
plt.tight_layout()
plt.savefig(output_dir / 'encoder_attention.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved encoder_attention.png')

# 2. Layer Similarity
print('Generating layer similarity plot...')
hidden_states = encoder_outputs.hidden_states
n_layers = len(hidden_states)

sims = np.zeros((n_layers, n_layers))
for i in range(n_layers):
    for j in range(n_layers):
        h_i = hidden_states[i][0].flatten()
        h_j = hidden_states[j][0].flatten()
        sims[i, j] = torch.nn.functional.cosine_similarity(h_i.unsqueeze(0), h_j.unsqueeze(0)).item()

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(sims, annot=True, fmt='.2f', cmap='RdYlBu_r', ax=ax, vmin=0, vmax=1)
ax.set_xlabel('Layer')
ax.set_ylabel('Layer')
ax.set_title('Encoder Layer Representation Similarity')
plt.tight_layout()
plt.savefig(output_dir / 'layer_similarity.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved layer_similarity.png')

# 3. Cross-Attention
print('Generating cross-attention plot...')
with torch.no_grad():
    decoder_outputs = model.model.decoder(
        input_ids=generated,
        encoder_hidden_states=encoder_outputs.last_hidden_state,
        output_attentions=True,
        return_dict=True,
    )

cross_attentions = decoder_outputs.cross_attentions
tokens = processor.batch_decode(generated[0], skip_special_tokens=False)

layer_idx = -1
attn = cross_attentions[layer_idx][0]
mean_attn = attn.mean(dim=0).cpu().numpy()

max_tokens = min(25, mean_attn.shape[0])
display_attn = mean_attn[:max_tokens]
display_tokens = [t.replace(' ', '_') for t in tokens[:max_tokens]]

fig, ax = plt.subplots(figsize=(14, 8))
sns.heatmap(display_attn, cmap='Blues', ax=ax, xticklabels=50)
ax.set_yticks(range(len(display_tokens)))
ax.set_yticklabels(display_tokens, fontsize=9)
ax.set_xlabel('Encoder Position (Audio Frame)')
ax.set_ylabel('Decoder Token')
ax.set_title('Cross-Attention: Which Audio Frames Each Token Attends To')
plt.tight_layout()
plt.savefig(output_dir / 'cross_attention.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved cross_attention.png')

# 4. Activation Patching
print('Generating activation patching plot...')

def patch_encoder_layer(model, input_features, layer_idx):
    def patch_hook(module, input, output):
        if isinstance(output, tuple):
            return (torch.zeros_like(output[0]),) + output[1:]
        return torch.zeros_like(output)

    handle = model.model.encoder.layers[layer_idx].register_forward_hook(patch_hook)
    try:
        with torch.no_grad():
            gen = model.generate(input_features, max_new_tokens=128)
        return processor.batch_decode(gen, skip_special_tokens=True)[0]
    finally:
        handle.remove()

original_wer = wer(ground_truth.lower(), generated_text.lower())
patching_results = []

for layer_idx in range(model.config.encoder_layers):
    patched_text = patch_encoder_layer(model, input_features, layer_idx)
    patched_wer = wer(ground_truth.lower(), patched_text.lower())
    wer_delta = patched_wer - original_wer
    patching_results.append({'layer': layer_idx, 'wer': patched_wer, 'delta': wer_delta})
    print(f'  Layer {layer_idx}: WER delta = {wer_delta:+.2f}')

layers = [r['layer'] for r in patching_results]
deltas = [r['delta'] for r in patching_results]

fig, ax = plt.subplots(figsize=(10, 5))
colors = ['#e74c3c' if d > 0 else '#27ae60' for d in deltas]
bars = ax.bar(layers, deltas, color=colors, alpha=0.8, edgecolor='black', linewidth=1.2)
ax.axhline(y=0, color='black', linewidth=1)
ax.set_xlabel('Encoder Layer', fontsize=12)
ax.set_ylabel('WER Increase When Layer Zeroed', fontsize=12)
ax.set_title('Causal Importance of Encoder Layers', fontsize=14)
ax.set_xticks(layers)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(output_dir / 'activation_patching.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved activation_patching.png')

# 5. Encoder Lens
print('Generating encoder lens results...')

from transformers.modeling_outputs import BaseModelOutput

lens_results = {}
for layer_idx, hidden in enumerate(hidden_states):
    normed = model.model.encoder.layer_norm(hidden)
    fake_outputs = BaseModelOutput(last_hidden_state=normed)
    with torch.no_grad():
        gen = model.generate(
            encoder_outputs=fake_outputs,
            max_new_tokens=20,
        )
    lens_results[layer_idx] = processor.batch_decode(gen, skip_special_tokens=True)[0]

# Create text summary image
fig, ax = plt.subplots(figsize=(12, 6))
ax.axis('off')

text = f"Ground Truth:\n\"{ground_truth}\"\n\n"
text += f"Full Model Output:\n\"{generated_text}\"\n\n"
text += "Encoder Lens (projecting each layer through decoder):\n"
for layer_idx, result in lens_results.items():
    text += f"  Layer {layer_idx}: \"{result}\"\n"

ax.text(0.02, 0.98, text, transform=ax.transAxes, fontsize=11,
        verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
ax.set_title('Encoder Lens: When Does Transcription Emerge?', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(output_dir / 'encoder_lens.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved encoder_lens.png')

print(f'\nAll images saved to {output_dir}/')
print('Done!')
