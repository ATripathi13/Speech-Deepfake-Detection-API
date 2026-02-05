"""
Quick Demo Script for Speech Deepfake Detection API

Demonstrates the API functionality without requiring full audio processing libraries.
"""

import sys
import json
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

print("="*70)
print("  Speech Deepfake Detection API - Quick Demo")
print("="*70)
print()

# Test 1: Model Architecture
print("1. Testing Model Architecture...")
print("-" * 70)
try:
    import torch
    from model import DeepfakeDetectionCNN, count_parameters
    
    model = DeepfakeDetectionCNN()
    params = count_parameters(model)
    
    # Test with sample input
    test_input = torch.randn(1, 1, 128, 100)
    output = model(test_input)
    
    print(f"   Model Parameters: {params:,}")
    print(f"   Input Shape: {tuple(test_input.shape)}")
    print(f"   Output Shape: {tuple(output.shape)}")
    print(f"   Output Value: {output.item():.4f} (probability)")
    print(f"   Status: SUCCESS")
except Exception as e:
    print(f"   Status: FAILED - {e}")

print()

# Test 2: Inference Module
print("2. Testing Inference Module...")
print("-" * 70)
try:
    from inference import classify_audio, calculate_confidence, generate_explanation
    
    # Test with different probabilities
    test_cases = [
        ("High AI confidence", 0.92),
        ("Medium AI confidence", 0.67),
        ("Low AI confidence", 0.52),
        ("Likely human", 0.35),
        ("High human confidence", 0.12)
    ]
    
    for desc, prob in test_cases:
        classification = classify_audio(prob)
        confidence = calculate_confidence(prob)
        explanation = generate_explanation(prob)
        
        print(f"   {desc} (prob={prob}):")
        print(f"      Classification: {classification}")
        print(f"      Confidence: {confidence}")
        print(f"      Explanation: {explanation}")
        print()
    
    print(f"   Status: SUCCESS")
except Exception as e:
    print(f"   Status: FAILED - {e}")

print()

# Test 3: Authentication
print("3. Testing Authentication Module...")
print("-" * 70)
try:
    import os
    from auth import get_valid_api_keys
    
    # Set test API keys
    os.environ["VALID_API_KEYS"] = "demo-key-12345,test-key-67890"
    
    valid_keys = get_valid_api_keys()
    print(f"   Valid API Keys: {len(valid_keys)} keys loaded")
    print(f"   Keys: {', '.join(valid_keys)}")
    print(f"   Status: SUCCESS")
except Exception as e:
    print(f"   Status: FAILED - {e}")

print()

# Test 4: FastAPI Application
print("4. Testing FastAPI Application Setup...")
print("-" * 70)
try:
    from main import app
    
    # Get route information
    routes = [route for route in app.routes if hasattr(route, 'path')]
    
    print(f"   App Title: {app.title}")
    print(f"   App Version: {app.version}")
    print(f"   Registered Routes:")
    for route in routes:
        if hasattr(route, 'methods'):
            methods = ', '.join(route.methods)
            print(f"      [{methods}] {route.path}")
    
    print(f"   Status: SUCCESS")
except Exception as e:
    print(f"   Status: FAILED - {e}")

print()
print("="*70)
print("  Demo Complete!")
print("="*70)
print()
print("To start the API server:")
print("  python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000")
print()
print("Then access:")
print("  - API Docs: http://localhost:8000/docs")
print("  - Health Check: http://localhost:8000/health")
print("  - API Endpoint: http://localhost:8000/api/voice-detection")
print()
print("="*70)
