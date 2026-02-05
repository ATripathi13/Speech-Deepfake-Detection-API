# VoiceShield AI - Frontend

Professional React-based web interface for neural voice forensic analysis.

## Features
- **Neural Waveform Visualization**: Interactive waveform preview using WaveSurfer.js.
- **Forensic Dashboard**: Clear visualization of detection results and confidence scores.
- **AI Explainability**: Breakdown of suspicious artifacts (Spectral, Pitch, Codec).
- **Persistent Audit Log**: Local detection history with filtering capabilities.
- **Responsive Design**: Optimized for mobile and desktop security monitoring.

## Tech Stack
- **Framework**: React + Vite
- **Styling**: Tailwind CSS
- **State**: Zustand (with Persist middleware)
- **Charts**: Recharts
- **Icons**: Lucide React
- **Audio**: WaveSurfer.js

## Setup

1. **Install Dependencies**:
   ```bash
   npm install
   ```

2. **Environment Variables**:
   Create a `.env` file with:
   ```env
   VITE_API_URL=http://localhost:8000
   VITE_API_KEY=your_api_key_here
   ```

3. **Run Development Server**:
   ```bash
   npm run dev
   ```

## Production Build
To create an optimized production bundle:
```bash
npm run build
```
The output will be in the `/dist` directory.
