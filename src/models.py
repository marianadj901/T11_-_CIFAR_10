import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def create_mlp(input_shape=(32, 32, 3), num_classes=10):
    """Cria um modelo MLP simples para CIFAR-10"""
    model = keras.Sequential([
        layers.Flatten(input_shape=input_shape),
        layers.Dense(512, activation="relu"),
        layers.Dense(256, activation="relu"),
        layers.Dense(num_classes, activation="softmax")
    ])
    return model


def create_cnn(input_shape=(32, 32, 3), num_classes=10):
    """Cria um modelo CNN simples para CIFAR-10"""
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation="relu", padding="same", input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(num_classes, activation="softmax")
    ])
    return model
