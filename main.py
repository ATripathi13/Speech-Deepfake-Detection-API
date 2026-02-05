"""
Speech Deepfake Detection API - Main FastAPI Application

Production-grade REST API for detecting AI-generated voice samples using ML forensics.
"""

import os
import time
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import Literal, Optional
import torch

# Import custom modules
from model import DeepfakeDetectionCNN
from audio_utils import preprocess_audio
from inference import load_model, run_inference
from auth import verify_api_key


# ========================
# Pydantic Models
# ========================

class VoiceDetectionRequest(BaseModel):
    """Request model for voice detection endpoint."""
    
    language: Literal["Tamil", "English", "Hindi", "Malayalam", "Telugu"] = Field(
        ...,
        description="Language of the audio sample (metadata only, model is language-agnostic)"
    )
    audioFormat: Literal["mp3", "wav", "ogg", "flac"] = Field(
        ...,
        description="Audio format of the Base64-encoded data"
    )
    audioBase64: str = Field(
        ...,
        description="Base64-encoded audio data",
        min_length=100
    )
    
    @validator('audioBase64')
    def validate_base64(cls, v):
        """Validate that audioBase64 is not empty and has reasonable length."""
        if len(v.strip()) < 100:
            raise ValueError("Audio data appears to be too short or empty")
        return v.strip()


class VoiceDetectionResponse(BaseModel):
    """Response model for voice detection endpoint."""
    
    status: Literal["success", "error"] = Field(
        ...,
        description="Status of the request"
    )
    language: str = Field(
        ...,
        description="Language from the request"
    )
    classification: Literal["AI_GENERATED", "HUMAN"] = Field(
        ...,
        description="Classification result"
    )
    confidenceScore: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score (0.0 - 1.0)"
    )
    explanation: str = Field(
        ...,
        description="Human-readable explanation of the detection"
    )
    processingTime: Optional[float] = Field(
        None,
        description="Processing time in seconds"
    )


class ErrorResponse(BaseModel):
    """Error response model."""
    
    status: Literal["error"]
    message: str
    details: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str
    model_loaded: bool
    version: str


# ========================
# FastAPI Application
# ========================

app = FastAPI(
    title="Speech Deepfake Detection API",
    description="Production-grade ML-based API for detecting AI-generated voice samples",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ========================
# Global Model Instance
# ========================

MODEL: Optional[DeepfakeDetectionCNN] = None
DEVICE = "cpu"  # Use CPU for inference as per requirements


@app.on_event("startup")
async def load_model_on_startup():
    """Load the ML model once on application startup."""
    global MODEL
    
    model_path = os.getenv("MODEL_PATH", "model.pt")
    
    try:
        print(f"Loading model from {model_path}...")
        MODEL = load_model(model_path, device=DEVICE)
        print("Model loaded successfully and ready for inference")
    except Exception as e:
        print(f"Warning: Could not load pre-trained model: {e}")
        print("  Loading untrained model for testing purposes")
        MODEL = DeepfakeDetectionCNN()
        MODEL.eval()


# ========================
# API Endpoints
# ========================

@app.get("/", include_in_schema=False)
async def root():
    """Root endpoint redirect."""
    return {
        "message": "Speech Deepfake Detection API",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint to verify API status.
    
    Returns:
        Health status and model information
    """
    return {
        "status": "healthy",
        "model_loaded": MODEL is not None,
        "version": "1.0.0"
    }


@app.post(
    "/api/voice-detection",
    response_model=VoiceDetectionResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Unauthorized - Invalid API key"},
        400: {"model": ErrorResponse, "description": "Bad Request - Invalid input"},
        500: {"model": ErrorResponse, "description": "Internal Server Error"}
    }
)
async def detect_deepfake(
    request: VoiceDetectionRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Detect whether a voice sample is AI-generated or human.
    
    This endpoint uses a CNN-based ML model to analyze mel spectrograms and detect
    forensic acoustic patterns such as:
    - Unnatural pitch consistency
    - Synthetic spectral artifacts
    - Temporal anomalies in speech patterns
    
    The model is language-agnostic and works across Tamil, English, Hindi, Malayalam, and Telugu.
    
    Args:
        request: Voice detection request with Base64 audio
        api_key: Validated API key from header
        
    Returns:
        Detection result with classification, confidence, and explanation
    """
    start_time = time.time()
    
    try:
        # Validate model is loaded
        if MODEL is None:
            raise HTTPException(
                status_code=500,
                detail="Model not loaded. Please contact administrator."
            )
        
        # Step 1: Preprocess audio (Base64 -> Mel Spectrogram)
        try:
            mel_spectrogram = preprocess_audio(
                request.audioBase64,
                request.audioFormat.lower()
            )
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Audio preprocessing failed: {str(e)}"
            )
        
        # Step 2: Run inference
        try:
            inference_result = run_inference(MODEL, mel_spectrogram)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Model inference failed: {str(e)}"
            )
        
        # Step 3: Calculate processing time
        processing_time = round(time.time() - start_time, 3)
        
        # Step 4: Build response
        response = VoiceDetectionResponse(
            status="success",
            language=request.language,
            classification=inference_result["classification"],
            confidenceScore=inference_result["confidenceScore"],
            explanation=inference_result["explanation"],
            processingTime=processing_time
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error: {str(e)}"
        )


# ========================
# Error Handlers
# ========================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler."""
    return {
        "status": "error",
        "message": exc.detail,
        "details": None
    }


if __name__ == "__main__":
    import uvicorn
    
    # Run the API server
    port = int(os.getenv("PORT", 8000))
    
    print(f"\n{'='*60}")
    print(f"  Speech Deepfake Detection API")
    print(f"{'='*60}")
    print(f"  Starting server on http://0.0.0.0:{port}")
    print(f"  Documentation: http://localhost:{port}/docs")
    print(f"  Default API Key: demo-key-12345")
    print(f"{'='*60}\n")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=True
    )
