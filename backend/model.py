"""
CNN Model for Speech Deepfake Detection

This model classifies mel spectrograms as either AI-generated or human speech
based on forensic acoustic patterns (pitch consistency, spectral artifacts, temporal anomalies).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepfakeDetectionCNN(nn.Module):
    """
    Convolutional Neural Network for detecting AI-generated speech.
    
    Architecture:
    - Input: Mel spectrogram (batch_size, 1, 128, time_frames)
    - Conv2D(32) -> ReLU -> MaxPool
    - Conv2D(64) -> ReLU -> MaxPool
    - Flatten -> Dense(128) -> Dropout -> Dense(1)
    - Sigmoid output (probability of AI-generated)
    
    The model learns forensic features like:
    - Unnatural pitch consistency
    - Synthetic spectral artifacts
    - Temporal anomalies in speech patterns
    """
    
    def __init__(self, n_mels=128, dropout_rate=0.3):
        super(DeepfakeDetectionCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=32,
            kernel_size=3,
            padding=1
        )
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            padding=1
        )
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Calculate flattened size after convolutions
        # After 2 max pools: (128/4, time_frames/4) = (32, time_frames/4)
        # We'll use adaptive pooling to handle variable time dimensions
        self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 8))
        
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(128, 1)
        
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, 1, n_mels, time_frames)
            
        Returns:
            Sigmoid probability of AI-generated speech (batch_size, 1)
        """
        # First conv block
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        
        # Second conv block
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        
        # Adaptive pooling to handle variable time lengths
        x = self.adaptive_pool(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        # Sigmoid for binary classification
        x = torch.sigmoid(x)
        
        return x


def count_parameters(model):
    """Count trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the model architecture
    model = DeepfakeDetectionCNN()
    print(f"Model Parameters: {count_parameters(model):,}")
    
    # Test with random input
    batch_size = 4
    n_mels = 128
    time_frames = 100
    
    x = torch.randn(batch_size, 1, n_mels, time_frames)
    output = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.4f}, {output.max():.4f}]")
    print("\n✓ Model architecture validated successfully!")
