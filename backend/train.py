"""
Training Pipeline for Speech Deepfake Detection Model

This script provides a complete training pipeline for the DeepfakeDetectionCNN model
using ASVspoof 2019/2021, FakeAVCeleb, and Mozilla Common Voice datasets.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import os
from pathlib import Path
from typing import Tuple, List
import librosa
from tqdm import tqdm

from model import DeepfakeDetectionCNN, count_parameters
from audio_utils import compute_mel_spectrogram, normalize_spectrogram


class DeepfakeAudioDataset(Dataset):
    """
    Dataset class for loading audio files and their labels.
    
    Expected directory structure:
    data/
      ├── fake/       # AI-generated audio (label: 1)
      │   ├── audio1.wav
      │   ├── audio2.wav
      │   └── ...
      └── real/       # Human speech (label: 0)
          ├── audio1.wav
          ├── audio2.wav
          └── ...
    """
    
    def __init__(self, data_dir: str, target_sr: int = 16000, n_mels: int = 128):
        """
        Initialize dataset.
        
        Args:
            data_dir: Root directory containing 'fake' and 'real' subdirectories
            target_sr: Target sample rate
            n_mels: Number of mel bands
        """
        self.data_dir = Path(data_dir)
        self.target_sr = target_sr
        self.n_mels = n_mels
        
        # Load file paths and labels
        self.samples = []
        
        # Load fake audio (AI-generated, label=1)
        fake_dir = self.data_dir / "fake"
        if fake_dir.exists():
            for audio_file in fake_dir.glob("*.wav"):
                self.samples.append((str(audio_file), 1))
            # Also check for mp3, flac, ogg
            for ext in ["*.mp3", "*.flac", "*.ogg"]:
                for audio_file in fake_dir.glob(ext):
                    self.samples.append((str(audio_file), 1))
        
        # Load real audio (Human, label=0)
        real_dir = self.data_dir / "real"
        if real_dir.exists():
            for audio_file in real_dir.glob("*.wav"):
                self.samples.append((str(audio_file), 0))
            for ext in ["*.mp3", "*.flac", "*.ogg"]:
                for audio_file in real_dir.glob(ext):
                    self.samples.append((str(audio_file), 0))
        
        print(f"Loaded {len(self.samples)} audio files")
        print(f"  Fake: {sum(1 for _, label in self.samples if label == 1)}")
        print(f"  Real: {sum(1 for _, label in self.samples if label == 0)}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """Load and preprocess audio sample."""
        audio_path, label = self.samples[idx]
        
        try:
            # Load audio
            waveform, sr = librosa.load(audio_path, sr=self.target_sr)
            
            # Limit duration to 10 seconds maximum
            max_length = self.target_sr * 10
            if len(waveform) > max_length:
                waveform = waveform[:max_length]
            
            # Compute mel spectrogram
            mel_spec = compute_mel_spectrogram(waveform, sr=self.target_sr, n_mels=self.n_mels)
            
            # Normalize
            mel_spec_norm = normalize_spectrogram(mel_spec)
            
            # Convert to tensor
            mel_tensor = torch.FloatTensor(mel_spec_norm).unsqueeze(0)  # (1, n_mels, time)
            
            return mel_tensor, torch.FloatTensor([label])
            
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            # Return a dummy sample on error
            return torch.zeros(1, self.n_mels, 100), torch.FloatTensor([0])


def collate_fn(batch):
    """Custom collate function to handle variable-length spectrograms."""
    mel_specs, labels = zip(*batch)
    
    # Find maximum time dimension
    max_time = max(spec.shape[2] for spec in mel_specs)
    
    # Pad all spectrograms to max_time
    padded_specs = []
    for spec in mel_specs:
        if spec.shape[2] < max_time:
            padding = torch.zeros(1, spec.shape[1], max_time - spec.shape[2])
            spec = torch.cat([spec, padding], dim=2)
        padded_specs.append(spec)
    
    # Stack into batch
    mel_batch = torch.stack(padded_specs)
    label_batch = torch.stack(labels)
    
    return mel_batch, label_batch


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for mel_specs, labels in pbar:
        mel_specs = mel_specs.to(device)
        labels = labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(mel_specs)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        predictions = (outputs > 0.5).float()
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
        
        pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{100*correct/total:.2f}%"})
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy


def validate(model, dataloader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for mel_specs, labels in tqdm(dataloader, desc="Validation"):
            mel_specs = mel_specs.to(device)
            labels = labels.to(device)
            
            outputs = model(mel_specs)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            predictions = (outputs > 0.5).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy


def train_model(
    data_dir: str,
    epochs: int = 20,
    batch_size: int = 16,
    learning_rate: float = 0.001,
    val_split: float = 0.2,
    save_path: str = "model.pt"
):
    """
    Complete training pipeline.
    
    Args:
        data_dir: Directory containing fake/ and real/ subdirectories
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate for optimizer
        val_split: Validation split ratio
        save_path: Path to save the best model
    """
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load dataset
    print("\nLoading dataset...")
    dataset = DeepfakeAudioDataset(data_dir)
    
    # Split into train and validation
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0  # Set to 0 for Windows compatibility
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    # Initialize model
    print("\nInitializing model...")
    model = DeepfakeDetectionCNN()
    model.to(device)
    print(f"Model parameters: {count_parameters(model):,}")
    
    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    # Training loop
    best_val_loss = float('inf')
    
    print(f"\nStarting training for {epochs} epochs...")
    print("="*60)
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc
            }, save_path)
            print(f"  ✓ Saved best model to {save_path}")
    
    print("\n" + "="*60)
    print("Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")


def create_dummy_model():
    """
    Create a dummy pre-trained model for testing purposes.
    
    This is useful when you don't have the full datasets available yet.
    """
    print("Creating dummy pre-trained model for testing...")
    
    model = DeepfakeDetectionCNN()
    
    # Initialize with some reasonable weights (not random)
    for param in model.parameters():
        if len(param.shape) >= 2:
            nn.init.xavier_uniform_(param)
    
    # Save as checkpoint
    torch.save({
        'epoch': 0,
        'model_state_dict': model.state_dict(),
        'val_loss': 0.5,
        'val_acc': 75.0,
        'note': 'Dummy model for testing - please train on real data'
    }, "model.pt")
    
    print("✓ Dummy model saved to model.pt")
    print("  NOTE: This is for testing only. Train on real datasets for production use.")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Speech Deepfake Detection Model")
    parser.add_argument("--data-dir", type=str, help="Path to data directory with fake/ and real/ subdirs")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--save-path", type=str, default="model.pt", help="Path to save model")
    parser.add_argument("--create-dummy", action="store_true", help="Create dummy model for testing")
    
    args = parser.parse_args()
    
    if args.create_dummy:
        create_dummy_model()
    elif args.data_dir:
        train_model(
            data_dir=args.data_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            save_path=args.save_path
        )
    else:
        print("Please specify --data-dir or use --create-dummy")
        print("\nExample usage:")
        print("  # Create dummy model for testing:")
        print("  python train.py --create-dummy")
        print("\n  # Train on real data:")
        print("  python train.py --data-dir ./data --epochs 20 --batch-size 16")
