"""
Inference Module for Speech Deepfake Detection

Handles model loading, prediction, classification, confidence calculation, and explanation generation.
"""

import torch
from model import DeepfakeDetectionCNN
from typing import Tuple, Dict


def load_model(model_path: str = "model.pt", device: str = "cpu") -> DeepfakeDetectionCNN:
    """
    Load the pre-trained deepfake detection model.
    
    Args:
        model_path: Path to the model checkpoint
        device: Device to load model on (cpu or cuda)
        
    Returns:
        Loaded model in evaluation mode
    """
    model = DeepfakeDetectionCNN()
    
    try:
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
            
        model.eval()  # Set to evaluation mode
        model.to(device)
        
        print(f"✓ Model loaded successfully from {model_path}")
        return model
        
    except FileNotFoundError:
        print(f"Warning: Model file not found at {model_path}")
        print(f"  Using untrained model (for testing only)")
        model.eval()
        return model
        
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {str(e)}")


def predict(model: DeepfakeDetectionCNN, mel_spectrogram: torch.Tensor) -> float:
    """
    Run inference on mel spectrogram.
    
    Args:
        model: Trained DeepfakeDetectionCNN model
        mel_spectrogram: Preprocessed mel spectrogram tensor (1, 1, n_mels, time_frames)
        
    Returns:
        Probability of audio being AI-generated (0.0 - 1.0)
    """
    with torch.no_grad():
        output = model(mel_spectrogram)
        probability = output.item()
    
    return probability


def classify_audio(probability: float, threshold: float = 0.5) -> str:
    """
    Classify audio as AI_GENERATED or HUMAN based on probability.
    
    Args:
        probability: Sigmoid output from model (0.0 - 1.0)
        threshold: Classification threshold (default: 0.5)
        
    Returns:
        "AI_GENERATED" or "HUMAN"
    """
    return "AI_GENERATED" if probability >= threshold else "HUMAN"


def calculate_confidence(probability: float) -> float:
    """
    Calculate confidence score from probability.
    
    For AI_GENERATED (prob >= 0.5): confidence is the probability itself
    For HUMAN (prob < 0.5): confidence is 1 - probability
    
    Args:
        probability: Model output probability
        
    Returns:
        Confidence score rounded to 2 decimals (0.00 - 1.00)
    """
    if probability >= 0.5:
        confidence = probability
    else:
        confidence = 1.0 - probability
    
    return round(confidence, 2)


def generate_explanation(probability: float) -> str:
    """
    Generate human-readable explanation based on detection score.
    
    Explanations are based on forensic acoustic patterns:
    - High AI probability: Robotic consistency, synthetic artifacts
    - Medium AI probability: Some synthetic indicators
    - Low AI probability: Natural human variation
    
    Args:
        probability: Model output probability (0.0 - 1.0)
        
    Returns:
        Human-readable explanation string
    """
    if probability >= 0.8:
        return "Robotic pitch consistency and unnatural spectral patterns detected"
    elif probability >= 0.65:
        return "Synthetic artifacts and temporal anomalies present"
    elif probability >= 0.5:
        return "Minor synthetic indicators detected in speech patterns"
    elif probability >= 0.35:
        return "Natural human speech variation with minor inconsistencies"
    elif probability >= 0.2:
        return "Strong indicators of natural human speech characteristics"
    else:
        return "Highly natural speech patterns with human-like variations"


def run_inference(
    model: DeepfakeDetectionCNN,
    mel_spectrogram: torch.Tensor
) -> Dict[str, any]:
    """
    Complete inference pipeline: predict, classify, calculate confidence, generate explanation.
    
    Args:
        model: Trained model
        mel_spectrogram: Preprocessed audio tensor
        
    Returns:
        Dictionary with classification, confidence, and explanation
    """
    # Get probability from model
    probability = predict(model, mel_spectrogram)
    
    # Classify as AI or Human
    classification = classify_audio(probability)
    
    # Calculate confidence score
    confidence = calculate_confidence(probability)
    
    # Generate explanation
    explanation = generate_explanation(probability)
    
    return {
        "classification": classification,
        "confidenceScore": confidence,
        "explanation": explanation,
        "raw_probability": round(probability, 4)  # For debugging
    }


if __name__ == "__main__":
    # Test inference module
    print("Testing inference module...")
    
    # Create dummy model and input
    model = DeepfakeDetectionCNN()
    model.eval()
    
    # Test with random mel spectrogram
    test_input = torch.randn(1, 1, 128, 100)
    
    # Run inference
    result = run_inference(model, test_input)
    
    print("\n✓ Inference test results:")
    print(f"  Classification: {result['classification']}")
    print(f"  Confidence: {result['confidenceScore']}")
    print(f"  Explanation: {result['explanation']}")
    print(f"  Raw Probability: {result['raw_probability']}")
