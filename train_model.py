"""
train_model.py
---------------
Trains a building-damage classifier using transfer learning on top of
EfficientNetB0 (pretrained on ImageNet).

Expected dataset layout (already organized from xBD / xView2 chips):

    dataset/
        No_Damage/
            img1.jpg
            img2.jpg
            ...
        Minor_Damage/
            ...
        Major_Damage/
            ...
        Destroyed/
            ...

Usage:
    python train_model.py --data_dir dataset --epochs 15 --batch_size 32

Output:
    models/building_damage_model.h5
"""

import argparse
import os

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = (224, 224)
CLASS_NAMES = ["No Damage", "Minor Damage", "Major Damage", "Destroyed"]


def build_data_generators(data_dir: str, batch_size: int):
    """
    Creates train/validation data generators with augmentation and
    EfficientNet-style preprocessing (scales pixels to [-1, 1]).
    """
    datagen = ImageDataGenerator(
        preprocessing_function=tf.keras.applications.efficientnet.preprocess_input,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.1,
        validation_split=0.2,  # 80/20 train-val split
    )

    train_gen = datagen.flow_from_directory(
        data_dir,
        target_size=IMG_SIZE,
        batch_size=batch_size,
        class_mode="categorical",
        subset="training",
        shuffle=True,
    )

    val_gen = datagen.flow_from_directory(
        data_dir,
        target_size=IMG_SIZE,
        batch_size=batch_size,
        class_mode="categorical",
        subset="validation",
        shuffle=False,
    )

    return train_gen, val_gen


def build_model(num_classes: int) -> tf.keras.Model:
    """
    Builds an EfficientNetB0 transfer-learning model.
    The convolutional base is frozen; only a new classification head
    is trained (fast, works well on smaller datasets).
    """
    base_model = EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(224, 224, 3),
        pooling="avg",
    )
    base_model.trainable = False  # freeze pretrained weights

    inputs = layers.Input(shape=(224, 224, 3))
    x = base_model(inputs, training=False)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def plot_training_history(history, out_path: str = "assets/training_history.png"):
    """Saves accuracy/loss curves for reference."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(history.history["accuracy"], label="train")
    axes[0].plot(history.history["val_accuracy"], label="val")
    axes[0].set_title("Accuracy")
    axes[0].legend()

    axes[1].plot(history.history["loss"], label="train")
    axes[1].plot(history.history["val_loss"], label="val")
    axes[1].set_title("Loss")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Saved training curves to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Train building damage classifier")
    parser.add_argument("--data_dir", type=str, default="dataset")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--model_out", type=str, default="models/building_damage_model.h5")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)

    print("Building data generators...")
    train_gen, val_gen = build_data_generators(args.data_dir, args.batch_size)

    print("Building model (EfficientNetB0 backbone, frozen)...")
    model = build_model(num_classes=len(CLASS_NAMES))
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(args.model_out, save_best_only=True),
        tf.keras.callbacks.ReduceLROnPlateau(patience=2, factor=0.5),
    ]

    print("Starting training...")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    model.save(args.model_out)
    print(f"Model saved to {args.model_out}")

    plot_training_history(history)


if __name__ == "__main__":
    main()
