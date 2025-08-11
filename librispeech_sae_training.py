import torchaudio
from datasets import load_dataset
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from itertools import islice
import numpy as np
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
# Set up device and mixed precision
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_amp = True if torch.cuda.is_available() else False
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

# Load the dataset
dataset_name = "librispeech_asr"
subset = "clean"
split = "train.100"

librispeech_dataset = load_dataset(dataset_name, subset, split=split, streaming=True)

# Load the Whisper Tiny processor and model
model_name = "openai/whisper-tiny"  # Changed to Whisper Tiny
processor = WhisperProcessor.from_pretrained(model_name)
whisper_model = WhisperForConditionalGeneration.from_pretrained(model_name).to(device)

class LibriSpeechDataset(Dataset):
    def __init__(self, dataset, processor, max_samples=1000):
        self.dataset = dataset
        self.processor = processor
        self.max_samples = max_samples
        self.cache_file = 'librispeech_features_cache_tiny.pt'  # Updated cache file name
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

    def process_sample(self, sample):
        speech_array = sample['audio']['array']
        sampling_rate = sample['audio']['sampling_rate']

        if sampling_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=16000)
            speech_array = resampler(torch.from_numpy(speech_array).float()).numpy()

        if speech_array.ndim > 1:
            speech_array = speech_array.mean(axis=0)

        input_features = self.processor(speech_array, sampling_rate=16000, return_tensors="pt").input_features

        return input_features.squeeze(0)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def extract_whisper_features(batch):
    batch = batch.to(device)
    with torch.no_grad():
        outputs = whisper_model.model.encoder(batch)
        encoder_features = outputs.last_hidden_state

        batch_size = batch.size(0)
        decoder_input_ids = torch.full((batch_size, 1), whisper_model.config.decoder_start_token_id, dtype=torch.long, device=device)
        decoder_outputs = whisper_model.model.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_features
        )
        decoder_features = decoder_outputs.last_hidden_state

    return encoder_features, decoder_features

class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(SparseAutoencoder, self).__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim)
    
    def forward(self, x):
        encoded = torch.relu(self.encoder(x))
        decoded = self.decoder(encoded)
        return encoded, decoded

def train_sae(sae, dataloader, optimizer, epochs, device, sparsity_weight=0.1):
    sae.to(device)
    criterion = nn.MSELoss()
    
    epoch_losses = []
    
    for epoch in range(epochs):
        total_loss = 0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            batch = batch.to(device)
            
            with torch.cuda.amp.autocast(enabled=use_amp):
                encoded, decoded = sae(batch)
                reconstruction_loss = criterion(decoded, batch)
                sparsity_loss = torch.mean(torch.abs(encoded))
                loss = reconstruction_loss + sparsity_weight * sparsity_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        epoch_losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs}, Avg Loss: {avg_loss:.4f}")
    
    return epoch_losses

def plot_loss(encoder_losses, decoder_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(encoder_losses, label='Encoder SAE')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.title('Training Loss over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig('sae_encoder_training_loss_100k_1152.png')
    plt.close()
    plt.figure(figsize=(10, 5))
    plt.plot(decoder_losses, label='Decoder SAE')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.title('Training Loss over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig('sae_decoder_training_loss_100k_1152.png')



def main():
    num_samples = 100000
    batch_size = 128
    dataset = LibriSpeechDataset(librispeech_dataset, processor, max_samples=num_samples)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=4, pin_memory=True)

    print("Extracting features...")
    encoder_features, decoder_features = [], []
    for batch in tqdm(dataloader):
        enc_feat, dec_feat = extract_whisper_features(batch)
        encoder_features.append(enc_feat.cpu())
        decoder_features.append(dec_feat.cpu())

    encoder_features = torch.cat(encoder_features)
    decoder_features = torch.cat(decoder_features)

    encoder_dataloader = DataLoader(encoder_features, batch_size=64, shuffle=True, pin_memory=True)
    decoder_dataloader = DataLoader(decoder_features, batch_size=64, shuffle=True, pin_memory=True)

    encoder_input_dim = encoder_features.shape[-1]
    decoder_input_dim = decoder_features.shape[-1]
    hidden_dim = 1152  # Reduced hidden dimension for Tiny model

    encoder_sae = SparseAutoencoder(encoder_input_dim, hidden_dim)
    decoder_sae = SparseAutoencoder(decoder_input_dim, hidden_dim)

    encoder_optimizer = optim.Adam(encoder_sae.parameters(), lr=1e-3)
    decoder_optimizer = optim.Adam(decoder_sae.parameters(), lr=1e-3)

    print("Training Encoder SAE...")
    encoder_losses = train_sae(encoder_sae, encoder_dataloader, encoder_optimizer, epochs=200, device=device)

    print("Training Decoder SAE...")
    decoder_losses = train_sae(decoder_sae, decoder_dataloader, decoder_optimizer, epochs=50, device=device)

    # Plot and save the loss curves
    plot_loss(encoder_losses, decoder_losses)

    torch.save(encoder_sae.state_dict(), "encoder_sae_tiny_100k_1152_space.pth")
    torch.save(decoder_sae.state_dict(), "decoder_sae_tiny_100k_1152_space.pth")

    print("Training complete. Loss curve saved as 'sae_training_loss_100k.png'")

if __name__ == "__main__":
    main()