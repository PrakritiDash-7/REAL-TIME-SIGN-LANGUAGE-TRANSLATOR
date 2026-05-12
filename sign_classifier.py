"""TensorFlow softmax classifier with geometric nearest-prototype fallback."""

from __future__ import annotations

import os

import numpy as np
from tensorflow import keras

from config import DISPLAY_LABELS, MODEL_PATH, SIGN_CLASSES, TF_CONFIDENCE_THRESHOLD
from geometry import nearest_prototype, prototype_distance


class SignClassifier:
    def __init__(self):
        self._model: keras.Model | None = None

    def _ensure_model(self) -> keras.Model:
        if self._model is not None:
            return self._model
        if not os.path.isfile(MODEL_PATH):
            from train_model import build_and_save

            build_and_save(verbose=1)
        self._model = keras.models.load_model(MODEL_PATH)
        return self._model

    def classify(self, vec: np.ndarray | None, lang: str) -> tuple[str, str, float, str]:
        """
        Returns (internal_sign_key, display_label, confidence_0_1, source)
        where source is 'tensorflow', 'geometric', or 'none'.
        """
        lang = lang if lang in DISPLAY_LABELS else "asl"
        if vec is None:
            return "", "Show your hand to the camera", 0.0, "none"
        model = self._ensure_model()
        x = np.asarray(vec, dtype=np.float32).reshape(1, -1)
        probs = model.predict(x, verbose=0)[0]
        ti = int(np.argmax(probs))
        tf_name = SIGN_CLASSES[ti]
        tf_conf = float(probs[ti])

        geo_name, geo_sim = nearest_prototype(vec)

        if tf_conf < TF_CONFIDENCE_THRESHOLD:
            label_key = geo_name
            conf = float(min(1.0, geo_sim))
            source = "geometric"
        else:
            label_key = tf_name
            conf = tf_conf
            source = "tensorflow"
            if geo_name != tf_name:
                d_tf = prototype_distance(vec, tf_name)
                d_geo = prototype_distance(vec, geo_name)
                if d_geo + 0.35 < d_tf and geo_sim > 0.35:
                    label_key = geo_name
                    conf = float(min(tf_conf, geo_sim) + 0.1)
                    source = "hybrid"

        display = DISPLAY_LABELS[lang].get(label_key, label_key)
        return label_key, display, conf, source
