# V1 Reference Document

Reference patterns from the initial whisper-sae implementation. Use these patterns in v2 where applicable.

## 1. LibriSpeech Dataset Caching Pattern

The caching approach worked well - process once, reuse many times:

```python
class LibriSpeechDataset(Dataset):
    def __init__(self, dataset, processor, max_samples=1000):
        self.dataset = dataset
        self.processor = processor
        self.max_samples = max_samples
        self.cache_file = 'librispeech_features_cache_tiny.pt'
        self.samples = self.load_or_process_samples()

    def load_or_process_samples(self):
        if os.path.exists(self.cache_file):
            print("Loading cached features...")
            return torch.load(self.cache_file)
        else:
            print("Processing samples...")
            samples = []
            for sample in tqdm(islice(self.dataset, self.max_samples), total=self.max_samples):
                processed_sample = self.process_sample(sample)
                samples.append(processed_sample)
            torch.save(samples, self.cache_file)
            return samples
```

**Keep**: Cache file approach, streaming dataset processing
**Improve**: Add metadata JSON alongside .pt files, version cache filenames by model/layer

## 2. Whisper Feature Extraction

Extracting encoder and decoder hidden states:

```python
def extract_whisper_features(batch):
    batch = batch.to(device)
    with torch.no_grad():
        # Encoder: gets mel spectrogram features
        outputs = whisper_model.model.encoder(batch)
        encoder_features = outputs.last_hidden_state  # [batch, 1500, 384]

        # Decoder: needs encoder states + start token
        batch_size = batch.size(0)
        decoder_input_ids = torch.full(
            (batch_size, 1),
            whisper_model.config.decoder_start_token_id,
            dtype=torch.long, device=device
        )
        decoder_outputs = whisper_model.model.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_features
        )
        decoder_features = decoder_outputs.last_hidden_state  # [batch, 1, 384]

    return encoder_features, decoder_features
```

**Keep**: Basic extraction pattern
**Improve**:
- Extract from ALL layers, not just last_hidden_state
- Apply layer norm before SAE (aiOla paper finding)
- Use hooks instead of direct calls for cleaner multi-layer extraction

## 3. SAE Architecture (V1 - Needs Improvement)

```python
class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(SparseAutoencoder, self).__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        encoded = torch.relu(self.encoder(x))
        decoded = self.decoder(encoded)
        return encoded, decoded
```

**Issues**:
- Only 3x expansion (384 -> 1152) - too small
- No unit-norm decoder columns
- No dead neuron detection
- No separate reconstruction vs sparsity loss tracking
- L1 regularization instead of TopK

**V2 requirements**:
- 8x-16x expansion (384 -> 3072 or 6144)
- Unit-norm decoder columns (normalize after each step)
- Dead neuron resampling (count tokens since last activation)
- BatchTopK activation (Mozilla found more stable for Whisper)
- Separate loss tracking for reconstruction, L0 sparsity

## 4. Mixed Precision Training Pattern

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_amp = True if torch.cuda.is_available() else False
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

# In training loop:
with torch.cuda.amp.autocast(enabled=use_amp):
    encoded, decoded = sae(batch)
    reconstruction_loss = criterion(decoded, batch)
    sparsity_loss = torch.mean(torch.abs(encoded))
    loss = reconstruction_loss + sparsity_weight * sparsity_loss

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
optimizer.zero_grad()
```

**Keep**: AMP pattern (update deprecated API)
**V2**: Use `torch.amp.autocast('cuda')` and `torch.amp.GradScaler('cuda')`

## 5. Analysis Patterns from Notebooks

### t-SNE Visualization
```python
from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
encoder_tsne = tsne.fit_transform(encoder_sample)

plt.scatter(encoder_tsne[:, 0], encoder_tsne[:, 1], s=2, alpha=0.7)
plt.title('t-SNE Visualization of Encoder Encoded Features')
```

### Sparsity Calculation
```python
def calculate_sparsity(tensor, threshold=1e-5):
    total_elements = tensor.numel()
    zero_elements = (tensor.abs() < threshold).sum().item()
    sparsity = zero_elements / total_elements
    return sparsity
```

### Gender Classification with Random Forest
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

clf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
clf.fit(X_train_small, y_train_small)
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred, target_names=le.classes_))
```

**Results from v1**:
- Female: precision 0.95, recall 0.07 (very poor recall)
- Male: precision 0.85, recall 1.00
- Accuracy: 85% overall but heavily biased

**V2 improvement**:
- Use Cohen's d effect sizes instead of classification accuracy
- Bootstrap confidence intervals
- Power analysis to determine sample sizes
- Balanced sampling from CommonVoice

### Accent Analysis with t-SNE
```python
# Collected 125 samples across 5 accents (25 each):
# - United States English
# - India and South Asia
# - Southern African
# - Filipino
# - West Indies and Bermuda

# Mean feature per sample (average over time dimension)
mean_encoded_feature = encoder_encoded.mean(dim=0)  # [hidden_dim]
```

### Hierarchical Clustering for Accents
```python
from scipy.cluster.hierarchy import dendrogram, linkage

linked = linkage(accent_feature_means, 'ward')
dendrogram(linked, labels=accent_feature_means.index.tolist())
```

## 6. Data Dimensions (Whisper-tiny)

| Component | Shape | Notes |
|-----------|-------|-------|
| Input mel spectrogram | [batch, 80, 3000] | 80 mel bins, 30 sec audio |
| Encoder output | [batch, 1500, 384] | 1500 time positions, 384 dim |
| Decoder output (1 token) | [batch, 1, 384] | Single start token |
| Encoder SAE activations | [batch*1500, hidden_dim] | Flattened time |
| Decoder SAE activations | [batch, hidden_dim] | Per-sample |

**Time resolution**: 1500 positions for 30 seconds = 20ms per position

## 7. What Worked

1. **Caching**: Processing LibriSpeech once and caching saves significant time
2. **Streaming datasets**: Memory efficient for large datasets
3. **Mixed precision**: ~2x speedup with no quality loss
4. **Batch processing**: DataLoader with pin_memory and num_workers
5. **Separate encoder/decoder SAEs**: Different characteristics warrant separate models

## 8. What To Fix in V2

1. **SAE expansion**: 3x is way too small, use 8x-16x
2. **No layer norm**: Need to apply Whisper's layer norm before SAE
3. **Only last layer**: Need all encoder layers 0-3
4. **No dead neuron handling**: Add resampling
5. **Combined loss**: Track reconstruction and sparsity separately
6. **No causal validation**: Add activation patching + WER
7. **Underpowered statistics**: Need effect sizes, not just p-values
8. **No feature interpretation**: Need audio clip extraction per feature

## 9. Hardware Notes

### V1 Training
- Used CUDA GPU (unspecified)
- 100k samples, 200 epochs encoder, 50 epochs decoder
- ~7 hours total training time

### V2 Plan
- Local: M3 Max 64GB for prototyping
- Modal: A10G GPU for production training
- Feature extraction: ~3 hours
- SAE training per layer: ~4 hours
