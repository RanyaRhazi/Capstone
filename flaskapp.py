import os

import joblib
import numpy as np
from flask import Flask, jsonify, request

N_PIXELS = 64  # 8x8 image, flattened

app = Flask(__name__)

model = joblib.load("knn_model.pkl")
scaler = joblib.load("scaler.pkl")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)
    if not data or "image" not in data:
        return jsonify({"error": "Request body must be JSON with an 'image' field."}), 400

    try:
        pixels = np.asarray(data["image"], dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return jsonify({"error": "'image' must be a list of numbers."}), 400

    if pixels.size != N_PIXELS:
        return jsonify({"error": f"'image' must contain {N_PIXELS} values (an 8x8 image)."}), 400

    # The server applies the same MinMax scaling used in training,
    # so clients send raw pixel values (0-16).
    pixels_scaled = scaler.transform(pixels.reshape(1, -1))
    prediction = model.predict(pixels_scaled)
    return jsonify({"prediction": prediction.tolist()})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5767)))
