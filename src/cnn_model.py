from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    Input,
    MaxPooling2D,
)


def create_cnn_model(input_shape=(64, 64, 1), num_classes=10):
    return Sequential(
        [
            Input(shape=input_shape),
            Conv2D(64, 3, activation="relu", padding="same"),
            BatchNormalization(),
            Conv2D(64, 3, activation="relu", padding="same"),
            BatchNormalization(),
            MaxPooling2D(2),
            Dropout(0.2),
            Conv2D(128, 3, activation="relu", padding="same"),
            BatchNormalization(),
            Conv2D(128, 3, activation="relu", padding="same"),
            BatchNormalization(),
            MaxPooling2D(2),
            Dropout(0.25),
            Conv2D(256, 3, activation="relu", padding="same"),
            BatchNormalization(),
            GlobalAveragePooling2D(),
            Dropout(0.3),
            Dense(512, activation="relu"),
            BatchNormalization(),
            Dropout(0.4),
            Dense(256, activation="relu"),
            BatchNormalization(),
            Dropout(0.3),
            Dense(num_classes, activation="softmax"),
        ]
    )
