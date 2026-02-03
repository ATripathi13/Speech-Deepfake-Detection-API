"""
API Test Script for Speech Deepfake Detection API

Demonstrates how to test the API with sample audio.
"""

import requests
import base64
import json
import os
from pathlib import Path


def test_api_with_audio_file(audio_path: str, api_key: str = "demo-key-12345", language: str = "English"):
    """
    Test the API with an audio file.
    
    Args:
        audio_path: Path to audio file (mp3, wav, etc.)
        api_key: API key for authentication
        language: Language of the audio
    """
    # Read and encode audio file
    with open(audio_path, "rb") as f:
        audio_bytes = f.read()
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
    
    # Get audio format from file extension
    audio_format = Path(audio_path).suffix[1:]  # Remove the dot
    
    # Prepare request
    url = "http://localhost:8000/api/voice-detection"
    headers = {
        "x-api-key": api_key,
        "Content-Type": "application/json"
    }
    payload = {
        "language": language,
        "audioFormat": audio_format,
        "audioBase64": audio_base64
    }
    
    # Make request
    print(f"Testing API with {audio_path}...")
    print(f"  Language: {language}")
    print(f"  Format: {audio_format}")
    print(f"  Size: {len(audio_bytes)} bytes")
    print()
    
    response = requests.post(url, headers=headers, json=payload)
    
    # Display results
    if response.status_code == 200:
        result = response.json()
        print("✓ SUCCESS")
        print(f"  Classification: {result['classification']}")
        print(f"  Confidence: {result['confidenceScore']}")
        print(f"  Explanation: {result['explanation']}")
        print(f"  Processing Time: {result.get('processingTime', 'N/A')}s")
        return result
    else:
        print(f"✗ ERROR: {response.status_code}")
        print(f"  {response.text}")
        return None


def test_health_check():
    """Test the health check endpoint."""
    url = "http://localhost:8000/health"
    
    print("Testing health check endpoint...")
    response = requests.get(url)
    
    if response.status_code == 200:
        result = response.json()
        print("✓ Health check passed")
        print(f"  Status: {result['status']}")
        print(f"  Model Loaded: {result['model_loaded']}")
        print(f"  Version: {result['version']}")
    else:
        print(f"✗ Health check failed: {response.status_code}")
    
    print()


def test_authentication():
    """Test API authentication."""
    url = "http://localhost:8000/api/voice-detection"
    
    # Test without API key
    print("Testing authentication (no API key)...")
    response = requests.post(url, json={
        "language": "English",
        "audioFormat": "mp3",
        "audioBase64": "test"
    })
    
    if response.status_code == 401:
        print("✓ Authentication correctly rejected request without API key")
    else:
        print(f"✗ Unexpected status code: {response.status_code}")
    
    # Test with invalid API key
    print("\nTesting authentication (invalid API key)...")
    response = requests.post(
        url,
        headers={"x-api-key": "invalid-key"},
        json={
            "language": "English",
            "audioFormat": "mp3",
            "audioBase64": "test"
        }
    )
    
    if response.status_code == 401:
        print("✓ Authentication correctly rejected invalid API key")
    else:
        print(f"✗ Unexpected status code: {response.status_code}")
    
    print()


def create_sample_audio():
    """Create a sample audio file for testing."""
    try:
        import numpy as np
        import soundfile as sf
        
        # Generate 3 seconds of 440Hz sine wave (A4 note)
        sample_rate = 16000
        duration = 3.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        
        # Save as WAV
        output_path = "sample_test.wav"
        sf.write(output_path, audio, sample_rate)
        
        print(f"✓ Created sample audio file: {output_path}")
        return output_path
        
    except ImportError:
        print("⚠ numpy or soundfile not available, cannot create sample audio")
        return None


if __name__ == "__main__":
    print("="*60)
    print("  Speech Deepfake Detection API - Test Suite")
    print("="*60)
    print()
    
    # Step 1: Health check
    test_health_check()
    
    # Step 2: Authentication tests
    test_authentication()
    
    # Step 3: API functionality test
    print("API Functionality Test")
    print("-" * 60)
    
    # Try to find a sample audio file or create one
    sample_file = None
    
    # Check for existing audio files
    for ext in [".wav", ".mp3", ".flac"]:
        for file in Path(".").glob(f"*{ext}"):
            sample_file = str(file)
            break
        if sample_file:
            break
    
    # If no file found, try to create one
    if not sample_file:
        sample_file = create_sample_audio()
    
    if sample_file and os.path.exists(sample_file):
        # Test with the sample file
        result = test_api_with_audio_file(sample_file)
    else:
        print("⚠ No audio file available for testing")
        print("  To test with real audio, place an mp3/wav file in the directory")
        print("  and run this script again.")
    
    print()
    print("="*60)
    print("  Test suite complete")
    print("="*60)
