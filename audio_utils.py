"""
Audio Processing Pipeline for Speech Deepfake Detection

Handles Base64 decoding, resampling, mel spectrogram extraction, and normalization.
"""

import base64
import io
import torch
import torchaudio
import librosa
import numpy as np
from typing import Tuple


def decode_base64_audio(base64_string: str, audio_format: str = "mp3") -> Tuple[np.ndarray, int]:
    """
    Decode Base64-encoded audio to waveform.
    
    Args:
        base64_string: Base64-encoded audio data
        audio_format: Audio format (mp3, wav, etc.)
        
    Returns:
        Tuple of (waveform, sample_rate)
    """
    try:
        # Decode Base64 to bytes
        audio_bytes = base64.b64decode(base64_string)
        
        # Create file-like object
        audio_buffer = io.BytesIO(audio_bytes)
        
        # Load audio using torchaudio
        waveform, sample_rate = torchaudio.load(audio_buffer, format=audio_format)
        
        # Convert to numpy and ensure mono
        waveform_np = waveform.numpy()
        if waveform_np.shape[0] > 1:
            # Convert stereo to mono by averaging channels
            waveform_np = np.mean(waveform_np, axis=0)
        else:
            waveform_np = waveform_np[0]
        
        return waveform_np, sample_rate
        
    except Exception as e:
        raise ValueError(f"Failed to decode audio: {str(e)}")


def resample_audio(waveform: np.ndarray, orig_sr: int, target_sr: int = 16000) -> np.ndarray:
    """
    Resample audio to target sample rate.
    
    Args:
        waveform: Audio waveform as numpy array
        orig_sr: Original sample rate
        target_sr: Target sample rate (default: 16kHz)
        
    Returns:
        Resampled waveform
    """
    if orig_sr != target_sr:
        waveform = librosa.resample(waveform, orig_sr=orig_sr, target_sr=target_sr)
    return waveform


def compute_mel_spectrogram(
    waveform: np.ndarray,
    sr: int = 16000,
    n_mels: int = 128,
    n_fft: int = 2048,
    hop_length: int = 512,
    fmin: float = 0.0,
    fmax: float = 8000.0
) -> np.ndarray:
    """
    Compute mel spectrogram from audio waveform.
    
    Args:
        waveform: Audio waveform
        sr: Sample rate
        n_mels: Number of mel bands
        n_fft: FFT window size
        hop_length: Hop length for STFT
        fmin: Minimum frequency
        fmax: Maximum frequency
        
    Returns:
        Mel spectrogram (n_mels, time_frames)
    """
    # Compute mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=waveform,
        sr=sr,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        fmin=fmin,
        fmax=fmax
    )
    
    # Convert to log scale (dB)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    return mel_spec_db


def normalize_spectrogram(mel_spec: np.ndarray) -> np.ndarray:
    """
    Normalize mel spectrogram to [0, 1] range.
    
    Args:
        mel_spec: Mel spectrogram in dB scale
        
    Returns:
        Normalized spectrogram
    """
    # Min-max normalization
    spec_min = mel_spec.min()
    spec_max = mel_spec.max()
    
    if spec_max - spec_min > 0:
        normalized = (mel_spec - spec_min) / (spec_max - spec_min)
    else:
        normalized = np.zeros_like(mel_spec)
    
    return normalized


def preprocess_audio(base64_audio: str, audio_format: str = "mp3") -> torch.Tensor:
    """
    Complete preprocessing pipeline: Base64 -> Waveform -> Mel Spectrogram -> Normalized Tensor.
    
    Args:
        base64_audio: Base64-encoded audio string
        audio_format: Audio format (mp3, wav, etc.)
        
    Returns:
        Preprocessed tensor ready for model input (1, 1, n_mels, time_frames)
    """
    # Step 1: Decode Base64 audio
    waveform, orig_sr = decode_base64_audio(base64_audio, audio_format)
    
    # Step 2: Resample to 16kHz
    waveform = resample_audio(waveform, orig_sr, target_sr=16000)
    
    # Step 3: Compute mel spectrogram
    mel_spec = compute_mel_spectrogram(waveform, sr=16000, n_mels=128)
    
    # Step 4: Normalize spectrogram
    mel_spec_norm = normalize_spectrogram(mel_spec)
    
    # Step 5: Convert to PyTorch tensor with shape (1, 1, n_mels, time_frames)
    mel_tensor = torch.FloatTensor(mel_spec_norm).unsqueeze(0).unsqueeze(0)
    
    return mel_tensor


def get_audio_duration(waveform: np.ndarray, sr: int) -> float:
    """
    Calculate audio duration in seconds.
    
    Args:
        waveform: Audio waveform
        sr: Sample rate
        
    Returns:
        Duration in seconds
    """
    return len(waveform) / sr


if __name__ == "__main__":
    # Test audio processing pipeline
    print("Testing audio processing pipeline...")
    
    # Create a simple test audio (1 second of 440Hz sine wave)
    import soundfile as sf
    sr = 16000
    duration = 3.0
    t = np.linspace(0, duration, int(sr * duration))
    test_audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    
    # Save to buffer and encode as base64
    buffer = io.BytesIO()
    sf.write(buffer, test_audio, sr, format='WAV')
    buffer.seek(0)
    audio_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    
    # Test preprocessing
    try:
        mel_tensor = preprocess_audio(audio_base64, audio_format='wav')
        print(f"✓ Preprocessing successful!")
        print(f"  Output shape: {mel_tensor.shape}")
        print(f"  Value range: [{mel_tensor.min():.4f}, {mel_tensor.max():.4f}]")
        print(f"  Expected: (1, 1, 128, time_frames)")
    except Exception as e:
        print(f"✗ Preprocessing failed: {e}")
