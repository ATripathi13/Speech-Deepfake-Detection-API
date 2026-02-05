# 🎙️ Speech Deepfake Detection API & Frontend

A professional, production-grade system for detecting AI-generated voice samples using deep learning forensic analysis. This project features a robust FastAPI backend and a high-performance React-based frontend dashboard.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109-green.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18.2-blue.svg)](https://reactjs.org/)
[![Vite](https://img.shields.io/badge/Vite-5.1-purple.svg)](https://vitejs.dev/)
[![Tailwind](https://img.shields.io/badge/Tailwind-3.4-cyan.svg)](https://tailwindcss.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.2-red.svg)](https://pytorch.org/)

## 🌟 Features

### Backend (FastAPI)
- **ML-Based Detection**: CNN model trained on mel spectrograms to detect forensic acoustic patterns.
- **Multi-Language Support**: Works across Tamil, English, Hindi, Malayalam, and Telugu.
- **Explainable AI**: Provides confidence scores with detailed forensic markers and explanations.
- **API Authentication**: Secure access via the `x-api-key` header.

### Frontend (React + Vite)
- **Real-time Visualization**: Interactive waveform analysis using **WaveSurfer.js**.
- **Forensic Dashboard**: Advanced result visualization with **Recharts**.
- **Detection History**: Persistent audit log of past analyses using **Zustand**.
- **Premium SaaS UI**: Sleek dark-mode aesthetic with responsive Tailwind CSS design.

## 📁 Project Structure

```
speech-deepfake-detection/
├── backend/             # FastAPI Engine
│   ├── main.py          # API entry point
│   ├── model.py         # CNN Architecture
│   ├── inference.py     # Inference logic
│   ├── audio_utils.py   # Audio preprocessing
│   ├── requirements.txt # Python dependencies
│   └── Dockerfile       # Container setup
├── frontend/            # React Application
│   ├── src/             # Frontend source
│   ├── tailwind.config.js
│   └── package.json     # Node dependencies
└── README.md            # This file
```

## 🚀 Quick Start

### 1. Prerequisites
- Python 3.10+ & Node.js 18+

### 2. Backend Setup
```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
python train.py --create-dummy  # Generate model weights for testing
python main.py
```
- **API Endpoint**: `http://localhost:8000`
- **Interactive Docs**: `http://localhost:8000/docs`

### 3. Frontend Setup
```bash
cd frontend
npm install
npm run dev
```
- **Dashboard**: `http://localhost:3001` (or next available port)

---

## 🏗️ Technical Architecture

### Detection Pipeline
1. **Base64 Decoding**: Decode audio from Base64 string.
2. **Resampling**: Standardize to 16kHz sample rate (Librosa).
3. **Feature Extraction**: Generate 128-band mel spectrogram.
4. **ML Inference**: CNN classifies as AI-generated or human.
5. **Explainability**: Generate confidence scores and forensic alerts.

### ML Model Details
- **Type**: 2-block CNN Architecture
- **Parameters**: 543,553 trainable parameters
- **Input**: Mel Spectrogram (1, 128, 100)
- **Acoustics Analyzed**: Pitch consistency, spectral anomalies, and codec artifacts.

## 🐳 Docker Deployment

```bash
cd backend
docker build -t voice-shield-api .
docker run -d -p 8000:8000 -e VALID_API_KEYS=your-key voice-shield-api
```

## 🤝 Contributing
Contributions are welcome! Please open an issue or submit a pull request for any forensic marker additions or UI/UX enhancements.

## 📧 Contact
For questions or support regarding the AI model or API integration, please open an issue on GitHub.

---
**Built with ❤️ for audio forensics and deepfake detection**
