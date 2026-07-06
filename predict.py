"""
predict.py
----------
Loads the trained building-damage classification model and runs
inference on a single image.

Provides:
    - load_model(): cached model loader
    - preprocess_image(): resize + normalize an uploaded image
    - predict_damage(): returns (class_name, confidence, all_probabilities)
"""

import os
from typing import Dict, Tuple

import numpy as np
from PIL import Image
import tensorflow as tf

IMG_SIZE = (224, 224)
CLASS_NAMES = ["No Damage", "Minor Damage", "Major Damage", "Destroyed"]
MODEL_PATH = os.path.join("models", "building_damage_model.h5")


def load_model(model_path: str = MODEL_PATH) -> tf.keras.Model:
    """
    Loads the trained Keras model from disk.

    Raises:
        FileNotFoundError if the model file does not exist, with a
        clear message so the Streamlit app can show a helpful error.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found at '{model_path}'. "
            "Please run train_model.py first, or place a trained "
            "building_damage_model.h5 in the models/ directory."
        )
    return tf.keras.models.load_model(model_path)


def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Prepares a PIL image for the model:
        1. Convert to RGB (handles grayscale/RGBA uploads)
        2. Resize to 224x224
        3. Apply EfficientNet preprocessing (scales to [-1, 1])
        4. Add batch dimension

    Args:
        image: PIL Image (as loaded from Streamlit's file_uploader)

    Returns:
        np.ndarray of shape (1, 224, 224, 3)
    """
    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)
    arr = np.array(image, dtype=np.float32)
    arr = tf.keras.applications.efficientnet.preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)
    return arr


def predict_damage(
    model: tf.keras.Model, image: Image.Image
) -> Tuple[str, float, Dict[str, float]]:
    """
    Runs inference on a single image.

    Args:
        model: a loaded tf.keras.Model
        image: a PIL Image

    Returns:
        predicted_class: str, e.g. "Major Damage"
        confidence: float, 0-100 (%)
        all_probs: dict mapping class name -> probability (0-100 %)
    """
    batch = preprocess_image(image)
    preds = model.predict(batch, verbose=0)[0]  # shape (num_classes,)

    predicted_idx = int(np.argmax(preds))
    predicted_class = CLASS_NAMES[predicted_idx]
    confidence = float(preds[predicted_idx]) * 100.0

    all_probs = {
        CLASS_NAMES[i]: float(preds[i]) * 100.0 for i in range(len(CLASS_NAMES))
    }

    return predicted_class, confidence, all_probs
