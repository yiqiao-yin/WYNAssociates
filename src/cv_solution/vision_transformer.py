from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
from tensorflow.keras import layers


class CustomViT:
    def __init__(
        self,
        num_classes: int = 2,
        input_shape: tuple = (299, 299, 3),
        learning_rate: float = 0.001,
        weight_decay: float = 0.0001,
        batch_size: int = 256,
        num_epochs: int = 800,
        image_size: int = 72,
        patch_size: int = 6,
        projection_dim: int = 64,
        num_heads: int = 4,
        transformer_units: List[int] = [128, 64],
        transformer_layers: int = 10,
        mlp_head_units: List[int] = [2048, 1024],
    ):
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.image_size = image_size
        self.patch_size = patch_size
        self.projection_dim = projection_dim
        self.num_heads = num_heads
        self.transformer_units = transformer_units
        self.transformer_layers = transformer_layers
        self.mlp_head_units = mlp_head_units
        self.num_patches = (image_size // patch_size) ** 2
        self.data_augmentation = keras.Sequential(
            [
                layers.Normalization(),
                layers.Resizing(image_size, image_size),
                layers.RandomFlip("horizontal"),
                layers.RandomRotation(factor=0.02),
                layers.RandomZoom(height_factor=0.2, width_factor=0.2),
            ],
            name="data_augmentation",
        )

    def mlp(
        self, x: tf.Tensor, hidden_units: List[int], dropout_rate: float
    ) -> tf.Tensor:
        for units in hidden_units:
            x = layers.Dense(units, activation=tf.nn.gelu)(x)
            x = layers.Dropout(dropout_rate)(x)
        return x

    class Patches(layers.Layer):
        def __init__(self, patch_size: int):
            super().__init__()
            self.patch_size = patch_size

        def call(self, images: tf.Tensor) -> tf.Tensor:
            batch_size = tf.shape(images)[0]
            patches = tf.image.extract_patches(
                images=images,
                sizes=[1, self.patch_size, self.patch_size, 1],
                strides=[1, self.patch_size, self.patch_size, 1],
                rates=[1, 1, 1, 1],
                padding="VALID",
            )
            patch_dims = patches.shape[-1]
            patches = tf.reshape(patches, [batch_size, -1, patch_dims])
            return patches

    class PatchEncoder(layers.Layer):
        def __init__(self, num_patches: int, projection_dim: int):
            super().__init__()
            self.num_patches = num_patches
            self.projection = layers.Dense(units=projection_dim)
            self.position_embedding = layers.Embedding(
                input_dim=num_patches, output_dim=projection_dim
            )

        def call(self, patch):
            positions = tf.range(start=0, limit=self.num_patches, delta=1)
            encoded = self.projection(patch) + self.position_embedding(positions)
            return encoded

    def create_vit_classifier(self) -> keras.Model:
        inputs = layers.Input(shape=self.input_shape)
        augmented = self.data_augmentation(inputs)
        patches = self.Patches(self.patch_size)(augmented)
        encoded_patches = self.PatchEncoder(self.num_patches, self.projection_dim)(
            patches
        )

        for _ in range(self.transformer_layers):
            x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
            attention_output = layers.MultiHeadAttention(
                num_heads=self.num_heads, key_dim=self.projection_dim, dropout=0.1
            )(x1, x1)
            x2 = layers.Add()([attention_output, encoded_patches])
            x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
            x3 = self.mlp(x3, hidden_units=self.transformer_units, dropout_rate=0.1)
            encoded_patches = layers.Add()([x3, x2])

        representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        representation = layers.Flatten()(representation)
        representation = layers.Dropout(0.5)(representation)
        features = self.mlp(
            representation, hidden_units=self.mlp_head_units, dropout_rate=0.5
        )
        logits = layers.Dense(self.num_classes)(features)
        model = keras.Model(inputs=inputs, outputs=logits)
        return model

    def run_experiment(
        self, model: keras.Model, x_train, y_train, x_test, y_test
    ) -> Dict[str, Any]:
        optimizer = tfa.optimizers.AdamW(
            learning_rate=self.learning_rate, weight_decay=self.weight_decay
        )

        model.compile(
            optimizer=optimizer,
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[
                keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
                keras.metrics.SparseTopKCategoricalAccuracy(5, name="top-5-accuracy"),
            ],
        )

        checkpoint_filepath = "/tmp/checkpoint"
        checkpoint_callback = keras.callbacks.ModelCheckpoint(
            checkpoint_filepath,
            monitor="val_accuracy",
            save_best_only=True,
            save_weights_only=True,
        )

        history = model.fit(
            x=x_train,
            y=y_train,
            batch_size=self.batch_size,
            epochs=self.num_epochs,
            validation_split=0.1,
            callbacks=[checkpoint_callback],
        )

        model.load_weights(checkpoint_filepath)
        _, accuracy, top_5_accuracy = model.evaluate(x_test, y_test)
        print(f"Test accuracy: {accuracy * 100:.2f}%")
        print(f"Test top 5 accuracy: {top_5_accuracy * 100:.2f}%")

        return history

    # Additional methods like plot_patches can be added here following the same pattern
