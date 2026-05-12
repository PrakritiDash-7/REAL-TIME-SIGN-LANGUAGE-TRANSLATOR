# Static finger-spelling style classes. Many handshapes overlap between ASL and ISL;
# display names can differ per language where you extend the map.

SIGN_CLASSES = ["A", "B", "C", "I", "L", "V", "Y", "OPEN_PALM"]

# User-facing labels per language (same shape, different cultural naming where useful).
DISPLAY_LABELS = {
    "asl": {
        "A": "A",
        "B": "B",
        "C": "C",
        "I": "I",
        "L": "L",
        "V": "V",
        "Y": "Y",
        "OPEN_PALM": "Open palm / B (flat)",
    },
    "isl": {
        "A": "A (ISL)",
        "B": "B (ISL)",
        "C": "C (ISL)",
        "I": "I (ISL)",
        "L": "L (ISL)",
        "V": "V (ISL)",
        "Y": "Y (ISL)",
        "OPEN_PALM": "Open palm (halt / attention)",
    },
}

TF_CONFIDENCE_THRESHOLD = 0.48
MODEL_PATH = "models/sign_classifier.keras"
