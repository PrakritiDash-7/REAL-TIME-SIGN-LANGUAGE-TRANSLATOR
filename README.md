# Real-time sign language translator

Flask web app that reads your webcam, tracks hands with [MediaPipe](https://developers.google.com/mediapipe), and classifies static finger-spelling–style shapes with TensorFlow, with a geometric nearest-prototype fallback.

## Recognized shapes

Classes: **A, B, C, I, L, V, Y**, and **open palm** (`OPEN_PALM` in code). Labels can be shown as **ASL** or **ISL** from the UI.

## Requirements

- Python 3.10+ recommended  
- Webcam  
- Dependencies in `requirements.txt` (Flask, OpenCV, MediaPipe, TensorFlow CPU, NumPy)

## Setup

```bash
cd "pd project"
python -m venv .venv
```

Activate the virtual environment (Windows PowerShell):

```powershell
.\.venv\Scripts\Activate.ps1
```

Then install packages:

```bash
python -m pip install -r requirements.txt
```

## Run

```bash
python app.py
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in a browser and allow camera access when prompted.

## First launch and model file

If `models/sign_classifier.keras` is not present, the app **trains and saves** a classifier on first load. That can take **roughly a minute or more on CPU**. After that, startup is faster.

The trained file is listed in `.gitignore` so it is not committed by default.

## Usage tips

Use a plain background and good lighting. Hold steady, clear letter shapes. See the in-app **Tips** section for the same guidance.

## Project layout

| Path | Role |
|------|------|
| `app.py` | Flask routes, MJPEG video stream |
| `hand_pipeline.py` | Hand landmarks via MediaPipe |
| `sign_classifier.py` | TensorFlow + geometric fallback |
| `train_model.py` | Build and save the Keras model |
| `config.py` | Classes, labels, paths, thresholds |
| `templates/` | HTML |
| `static/` | CSS |
| `models/` | Saved `sign_classifier.keras` (generated) |
