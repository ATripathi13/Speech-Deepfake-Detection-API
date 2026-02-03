# 🎙️ Speech Deepfake Detection API

Production-grade REST API for detecting AI-generated voice samples using deep learning forensic analysis.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109-green.svg)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.2-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 🌟 Features

- **ML-Based Detection**: CNN model trained on mel spectrograms to detect forensic acoustic patterns
- **Multi-Language Support**: Works with Tamil, English, Hindi, Malayalam, and Telugu (language-agnostic detection)
- **Fast Inference**: <2 second response time on CPU
- **API Authentication**: Secure API key-based access control
- **Explainable AI**: Provides confidence scores with human-readable explanations
- **Production Ready**: Docker-containerized, with health checks and comprehensive error handling

## 🏗️ Architecture

```
Client → REST API → Audio Decoder → Feature Extractor → ML Model → JSON Response
```

### Detection Pipeline

1. **Base64 Decoding**: Decode MP3/WAV audio from Base64 string
2. **Resampling**: Convert to 16kHz sample rate
3. **Feature Extraction**: Generate 128-band mel spectrogram
4. **Normalization**: Min-max normalize spectrogram
5. **ML Inference**: CNN classifies as AI-generated or human
6. **Explainability**: Generate confidence score and explanation

### ML Model Architecture

```
Input: Mel Spectrogram (1, 128, time_frames)
  ↓
Conv2D (32 filters, 3x3) + BatchNorm + ReLU + MaxPool
  ↓
Conv2D (64 filters, 3x3) + BatchNorm + ReLU + MaxPool
  ↓
Adaptive Average Pool (8x8)
  ↓
Flatten → Dense(128) → Dropout(0.3) → Dense(1)
  ↓
Sigmoid → Probability (0-1)
```

**Model learns to detect:**
- Unnatural pitch consistency
- Synthetic spectral artifacts
- Temporal anomalies in speech patterns

## 📦 Installation

### Prerequisites

- Python 3.10+
- pip

### Local Setup

```bash
# Clone repository
git clone https://github.com/ATripathi13/Speech-Deepfake-Detection-API.git
cd Speech-Deepfake-Detection-API

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create environment file
cp .env.example .env
# Edit .env with your API keys

# Generate dummy model for testing
python train.py --create-dummy

# Run the API server
python main.py
```

The API will be available at `http://localhost:8000`

## 🐳 Docker Deployment

```bash
# Build Docker image
docker build -t speech-deepfake-api .

# Run container
docker run -d \
  -p 8000:8000 \
  -e VALID_API_KEYS=your-api-key-here \
  --name deepfake-detector \
  speech-deepfake-api

# Check health
curl http://localhost:8000/health
```

## 🚀 API Usage

### Authentication

All requests require an API key in the header:

```
x-api-key: your-api-key-here
```

### Endpoint: Voice Detection

**POST** `/api/voice-detection`

**Request Body:**

```json
{
  "language": "Tamil",
  "audioFormat": "mp3",
  "audioBase64": "BASE64_ENCODED_AUDIO_STRING"
}
```

**Response:**

```json
{
  "status": "success",
  "language": "Tamil",
  "classification": "AI_GENERATED",
  "confidenceScore": 0.87,
  "explanation": "Synthetic artifacts and temporal anomalies present",
  "processingTime": 1.23
}
```

### Example with cURL

```bash
curl -X POST http://localhost:8000/api/voice-detection \
  -H "x-api-key: demo-key-12345" \
  -H "Content-Type: application/json" \
  -d '{
    "language": "English",
    "audioFormat": "mp3",
    "audioBase64": "UklGRiQAAABXQVZFZm10..."
  }'
```

### Example with Python

```python
import requests
import base64

# Read audio file
with open("sample.mp3", "rb") as f:
    audio_base64 = base64.b64encode(f.read()).decode()

# Make request
response = requests.post(
    "http://localhost:8000/api/voice-detection",
    headers={"x-api-key": "demo-key-12345"},
    json={
        "language": "English",
        "audioFormat": "mp3",
        "audioBase64": audio_base64
    }
)

result = response.json()
print(f"Classification: {result['classification']}")
print(f"Confidence: {result['confidenceScore']}")
print(f"Explanation: {result['explanation']}")
```

## 🧠 Model Training

### Dataset Preparation

Organize your data in this structure:

```
data/
├── fake/       # AI-generated audio samples
│   ├── sample1.wav
│   ├── sample2.mp3
│   └── ...
└── real/       # Human speech samples
    ├── sample1.wav
    ├── sample2.mp3
    └── ...
```

**Recommended Datasets:**
- [ASVspoof 2019](https://www.asvspoof.org/) - Spoofing and countermeasures
- [ASVspoof 2021](https://www.asvspoof.org/) - Deep fake detection
- [FakeAVCeleb](https://github.com/DASH-Lab/FakeAVCeleb) - Deepfake audio-visual dataset
- [Mozilla Common Voice](https://commonvoice.mozilla.org/) - Real human speech

### Training

```bash
# Train on your dataset
python train.py --data-dir ./data --epochs 20 --batch-size 16 --lr 0.001

# Or create a dummy model for testing
python train.py --create-dummy
```

### Training Parameters

- `--data-dir`: Path to data directory
- `--epochs`: Number of training epochs (default: 20)
- `--batch-size`: Batch size (default: 16)
- `--lr`: Learning rate (default: 0.001)
- `--save-path`: Model save path (default: model.pt)

## 📊 API Documentation

Interactive API documentation is available at:

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `VALID_API_KEYS` | Comma-separated API keys | `demo-key-12345` |
| `MODEL_PATH` | Path to model weights | `model.pt` |
| `PORT` | Server port | `8000` |
| `HOST` | Server host | `0.0.0.0` |

## 📁 Project Structure

```
voice-detector/
├── main.py              # FastAPI application
├── model.py             # CNN model architecture
├── inference.py         # Inference and explainability
├── audio_utils.py       # Audio processing pipeline
├── auth.py              # API key authentication
├── train.py             # Training pipeline
├── requirements.txt     # Python dependencies
├── Dockerfile           # Docker configuration
├── .env.example         # Environment template
├── .gitignore          # Git ignore rules
└── README.md           # This file
```

## 🎯 Performance

- **Response Time**: < 2 seconds (CPU inference)
- **Model Size**: ~600KB (lightweight CNN)
- **Memory Usage**: ~200MB
- **Supported Audio**: MP3, WAV, OGG, FLAC
- **Max Audio Length**: 30 seconds (recommended)

## 🔐 Security

- API key authentication required for all endpoints
- Input validation on all requests
- Rate limiting (configure with reverse proxy)
- CORS enabled (configure for production)

## 🌍 Cloud Deployment

### AWS (EC2 / ECS)
```bash
# Build and push to ECR
docker tag speech-deepfake-api:latest <your-ecr-repo>
docker push <your-ecr-repo>
```

### Google Cloud Platform (Cloud Run)
```bash
gcloud run deploy speech-deepfake-api \
  --image speech-deepfake-api \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

### Azure (Container Instances)
```bash
az container create \
  --resource-group myResourceGroup \
  --name speech-deepfake-api \
  --image speech-deepfake-api \
  --dns-name-label deepfake-api \
  --ports 8000
```

## 🧪 Testing

```bash
# Test model architecture
python model.py

# Test audio processing
python audio_utils.py

# Test inference
python inference.py

# Test authentication
python auth.py

# Run API server
python main.py
```

## 📝 License

This project is licensed under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## 📧 Contact

For questions or support, please open an issue on GitHub.

## 🙏 Acknowledgments

- ASVspoof Challenge organizers
- Mozilla Common Voice contributors
- FakeAVCeleb dataset creators
- FastAPI and PyTorch communities

---

**Built with ❤️ for audio forensics and deepfake detection**
