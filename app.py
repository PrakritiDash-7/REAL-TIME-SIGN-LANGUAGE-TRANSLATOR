"""Flask app: webcam stream with MediaPipe hands + TensorFlow sign classification."""

from __future__ import annotations

import sys
import threading

import cv2
from flask import Flask, Response, render_template, request

from features import landmarks_to_vector
from hand_pipeline import HandPipeline
from sign_classifier import SignClassifier

app = Flask(__name__)

_cap_lock = threading.Lock()
_cap: cv2.VideoCapture | None = None
_pipeline: HandPipeline | None = None
_classifier: SignClassifier | None = None


def _camera() -> cv2.VideoCapture:
    global _cap
    if _cap is None or not _cap.isOpened():
        if sys.platform == "win32":
            _cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        else:
            _cap = cv2.VideoCapture(0)
        _cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        _cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return _cap


def _get_hand_pipeline() -> HandPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = HandPipeline()
    return _pipeline


def _get_sign_classifier() -> SignClassifier:
    global _classifier
    if _classifier is None:
        _classifier = SignClassifier()
    return _classifier


def _mjpeg_frame(jpg: bytes) -> bytes:
    return b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    lang = request.args.get("lang", "asl").lower()
    if lang not in ("asl", "isl"):
        lang = "asl"

    def generate():
        cap = _camera()
        pipe = _get_hand_pipeline()
        clf = _get_sign_classifier()
        while True:
            with _cap_lock:
                ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            landmarks, frame = pipe.process(frame)
            vec = landmarks_to_vector(landmarks) if landmarks else None
            _key, display, conf, source = clf.classify(vec, lang)

            y0 = 28
            cv2.putText(
                frame,
                f"{display}",
                (16, y0),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.85,
                (20, 230, 20),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"{conf * 100:.0f}%  [{source}]",
                (16, y0 + 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (240, 240, 240),
                2,
                cv2.LINE_AA,
            )
            ok2, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
            if not ok2:
                continue
            yield _mjpeg_frame(buf.tobytes())

    return Response(
        generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


def main():
    print(
        "Loading classifier (first launch trains until models/sign_classifier.keras exists; "
        "may take a minute on CPU)..."
    )
    _get_sign_classifier()._ensure_model()
    print("Open http://127.0.0.1:5000 — allow camera access if the OS prompts.")
    app.run(host="0.0.0.0", port=5000, threaded=True, debug=False)


if __name__ == "__main__":
    main()
