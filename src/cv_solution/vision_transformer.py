from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
from tensorflow.keras import layers


# learning_rate = 0.001
# weight_decay = 0.0001
# batch_size = 256
# num_epochs = 800
# image_size = 72  # We'll resize input images to this size
# patch_size = 6  # Size of the patches to be extract from the input images
# num_patches = (image_size // patch_size) ** 2
# projection_dim = 64
# num_heads = 4
# transformer_units = [
#     projection_dim * 2,
#     projection_dim,
# ]  # Size of the transformer layers
# transformer_layers = 10
# mlp_head_units = [2048, 1024]  # Size of the dense layers of the final classifier


def mlp(x: tf.Tensor, hidden_units: list[int], dropout_rate: float) -> tf.Tensor:
    """
    Constructs a multi-layer perceptron with Gelu activation and dropout layers.

    Args:
        x (tf.Tensor): Input tensor to the MLP.
        hidden_units (list[int]): A list of integers for the number of units in each hidden layer.
        dropout_rate (float): The rate of dropout to apply after each hidden layer.

    Returns:
        A tensor representing the output of the MLP.
    """
    # For each specified number of hidden units,
    # add a dense layer with gelu activation followed by dropout.
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)

    # Return the final output after passing through all the hidden layers
    return x


class Patches(layers.Layer):
    def __init__(self, patch_size: int):
        """
        A layer that extracts patches from an image tensor.

        Args:
            patch_size (int): The size of each patch to extract.
        """
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, images: tf.Tensor) -> tf.Tensor:
        """
        Extracts patches from the input image tensor.

        Args:
            images (tf.Tensor): A tensor representing a batch of images.

        Returns:
            A tensor representing the extracted patches.
        """
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


def plot_patches(sample_index: int) -> None:
    """
    Plots the original image and patches extracted from it.

    Args:
        sample_index (int): The index of the image to plot from the training set.
    """
    plt.figure(figsize=(4, 4))
    # Select a random image from the training set
    image = x_train[np.random.choice(range(x_train.shape[sample_index]))]
    plt.imshow(image.astype("uint8"))
    plt.axis("off")

    resized_image = tf.image.resize(
        tf.convert_to_tensor([image]), size=(image_size, image_size)
    )
    # Extract patches from the resized image
    patches = Patches(patch_size)(resized_image)
    print(f"Image size: {image_size} X {image_size}")
    print(f"Patch size: {patch_size} X {patch_size}")
    print(f"Patches per image: {patches.shape[1]}")
    print(f"Elements per patch: {patches.shape[-1]}")

    n = int(np.sqrt(patches.shape[1]))
    plt.figure(figsize=(4, 4))
    # Plot all patches as subplots on a grid
    for i, patch in enumerate(patches[sample_index]):
        ax = plt.subplot(n, n, i + 1)
        patch_img = tf.reshape(patch, (patch_size, patch_size, 3))
        plt.imshow(patch_img.numpy().astype("uint8"))
        plt.axis("off")


class PatchEncoder(layers.Layer):
    def __init__(self, num_patches: int, projection_dim: int):
        """
        Initializes the PatchEncoder layer.

        Args:
            num_patches (int): The number of patches to extract from an image.
            projection_dim (int): The dimensionality of the encoding space.
        """
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        # A Dense layer to project the patches into the encoding space
        self.projection = layers.Dense(units=projection_dim)
        # An Embedding layer to provide position embeddings for each patch
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        """
        Encodes a patch by projecting it into the encoding space and adding a
        position embedding.

        Args:
            patch (tf.Tensor): A patch extracted from an image.

        Returns:
            tf.Tensor: The encoded patch, with shape (batch_size, projection_dim).
        """
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded


def create_vit_classifier() -> keras.Model:
    """
    Creates a Vision Transformer (ViT) classifier model.

    Returns:
        keras.Model: A ViT classifier model.
    """
    inputs = layers.Input(shape=input_shape)
    # Augment data.
    augmented = data_augmentation(inputs)
    # Create patches.
    patches = Patches(patch_size)(augmented)
    # Encode patches.
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

    # Create multiple layers of the Transformer block.
    for _ in range(transformer_layers):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        # Skip connection 1.
        x2 = layers.Add()([attention_output, encoded_patches])
        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        # Skip connection 2.
        encoded_patches = layers.Add()([x3, x2])

    # Create a [batch_size, projection_dim] tensor.
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.5)(representation)
    # Add MLP.
    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.5)
    # Classify outputs.
    logits = layers.Dense(num_classes)(features)
    # Create the Keras model.
    model = keras.Model(inputs=inputs, outputs=logits)
    return model


def run_experiment(model: keras.Model) -> Dict[str, Any]:
    """
    Trains a given Keras model on training data, evaluates it on test data,
    and returns the training history.

    Args:
        model (keras.Model): A Keras model to train and evaluate.

    Returns:
        dict: A dictionary containing the training history.
    """
    optimizer = tfa.optimizers.AdamW(
        learning_rate=learning_rate, weight_decay=weight_decay
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
        batch_size=batch_size,
        epochs=num_epochs,
        validation_split=0.1,
        callbacks=[checkpoint_callback],
    )

    model.load_weights(checkpoint_filepath)
    _, accuracy, top_5_accuracy = model.evaluate(x_test, y_test)
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")
    print(f"Test top 5 accuracy: {round(top_5_accuracy * 100, 2)}%")

    return history
