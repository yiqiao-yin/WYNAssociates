from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
from typing import Tuple


class SelfAttention(layers.Layer):
    """
    A custom self-attention layer that computes attention scores to enhance model performance by focusing on relevant parts of the input data.

    This layer creates query, key, and value representations of the input, then calculates attention scores to determine how much focus to put on each part of the input data. The output is a combination of the input and the attention mechanism's weighted focus, which allows the model to pay more attention to certain parts of the data.

    Attributes:
        query_dense (keras.layers.Dense): A dense layer for transforming the input into a query tensor.
        key_dense (keras.layers.Dense): A dense layer for transforming the input into a key tensor.
        value_dense (keras.layers.Dense): A dense layer for transforming the input into a value tensor.
        combine_heads (keras.layers.Dense): A dense layer for combining the attention heads' outputs.
    """

    def __init__(self, **kwargs):
        super(SelfAttention, self).__init__(**kwargs)

    def build(self, input_shape: Tuple[int, ...]):
        """
        Initializes the internal dense layers based on the last dimension of the input shape, setting up the query, key, value, and combine heads layers.

        Args:
            input_shape (Tuple[int, ...]): The shape of the input tensor to the layer.
        """
        self.query_dense = layers.Dense(units=input_shape[-1])
        self.key_dense = layers.Dense(units=input_shape[-1])
        self.value_dense = layers.Dense(units=input_shape[-1])
        self.combine_heads = layers.Dense(units=input_shape[-1])

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Performs the self-attention mechanism on the input tensor and returns the combined output with a residual connection.

        Args:
            inputs (tf.Tensor): The input tensor to the self-attention layer.

        Returns:
            tf.Tensor: The output tensor after applying self-attention and combining with the input tensor through a residual connection.
        """
        # Generate query, key, value tensors
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)

        # Calculate attention scores
        scores = tf.matmul(query, key, transpose_b=True)
        distribution = tf.nn.softmax(scores)
        attention_output = tf.matmul(distribution, value)

        # Combine heads and add residual connection
        combined_output = self.combine_heads(attention_output) + inputs
        return combined_output


from typing import Tuple
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf


def vapaad(input_shape: Tuple[int, int, int], image_size: int = 64) -> keras.Model:
    """
    Builds a video processing model with data augmentation and self-attention mechanisms.

    Args:
        input_shape (Tuple[int, int, int]): The shape of the input frames.
        image_size (int, optional): The target size to resize the frames. Defaults to 64.

    Returns:
        keras.Model: A compiled keras model ready for training.
    """
    # Initialize the data augmentation pipeline
    data_augmentation = keras.Sequential(
        [
            # layers.RandomFlip("horizontal"),  # Randomly flip frames horizontally
            layers.RandomRotation(
                factor=0.02
            ),  # Randomly rotate frames by a small angle
            # layers.RandomZoom(height_factor=0.1, width_factor=0.1),  # Randomly zoom in on frames
        ],
        name="data_augmentation",
    )

    inp = layers.Input(
        shape=input_shape
    )  # Define the input layer with the specified shape

    # Apply data augmentation to each frame using the TimeDistributed layer
    x = layers.TimeDistributed(data_augmentation)(inp)

    # First ConvLSTM2D layer with self-attention
    x = layers.ConvLSTM2D(
        filters=64,
        kernel_size=(5, 5),
        padding="same",
        return_sequences=True,
        activation="relu",
    )(x)
    x = layers.BatchNormalization()(x)  # Normalize the activations of the first layer
    x = SelfAttention()(x)  # Apply self-attention mechanism

    # Second ConvLSTM2D layer with self-attention
    x = layers.ConvLSTM2D(
        filters=64,
        kernel_size=(3, 3),
        padding="same",
        return_sequences=True,
        activation="relu",
    )(x)
    x = layers.BatchNormalization()(x)  # Normalize the activations of the second layer
    x = SelfAttention()(x)  # Apply self-attention mechanism

    # Third ConvLSTM2D layer with self-attention
    x = layers.ConvLSTM2D(
        filters=64,
        kernel_size=(1, 1),
        padding="same",
        return_sequences=True,
        activation="relu",
    )(x)
    x = SelfAttention()(x)  # Apply self-attention mechanism

    # Final Conv3D layer to produce the output
    x = layers.Conv3D(
        filters=1, kernel_size=(3, 3, 3), activation="sigmoid", padding="same"
    )(x)

    # Create the model
    model = keras.models.Model(inputs=inp, outputs=x)

    # Return
    return model


# Instructor model
def instructor_model(
    input_shape: Tuple[int, int, int], image_size: int = 64
) -> keras.Model:
    """
    Builds a video processing model ending with fully connected layers.

    Args:
        input_shape (Tuple[int, int, int]): The shape of the input frames.
        image_size (int, optional): The target size to resize the frames. Defaults to 64.

    Returns:
        keras.Model: A compiled keras model ready for training with a one-dimensional output.
    """
    # Initialize the data augmentation pipeline
    data_augmentation = keras.Sequential(
        [
            layers.RandomRotation(factor=0.02),
        ],
        name="data_augmentation",
    )

    inp = layers.Input(shape=input_shape)

    # Apply data augmentation to each frame using the TimeDistributed layer
    x = layers.TimeDistributed(data_augmentation)(inp)

    # ConvLSTM2D layers with self-attention
    x = layers.ConvLSTM2D(
        filters=64,
        kernel_size=(5, 5),
        padding="same",
        return_sequences=True,
        activation="relu",
    )(x)
    x = layers.BatchNormalization()(x)
    x = SelfAttention()(x)

    x = layers.ConvLSTM2D(
        filters=64,
        kernel_size=(3, 3),
        padding="same",
        return_sequences=True,
        activation="relu",
    )(x)
    x = layers.BatchNormalization()(x)
    x = SelfAttention()(x)

    x = layers.ConvLSTM2D(
        filters=64,
        kernel_size=(1, 1),
        padding="same",
        return_sequences=True,
        activation="relu",
    )(x)
    x = SelfAttention()(x)

    # Adding global average pooling to reduce the dimensionality before dense layers
    x = layers.GlobalAveragePooling3D()(x)

    # Fully connected dense layers
    x = layers.Dense(2048, activation="relu")(x)
    x = layers.Dense(1024, activation="relu")(x)

    # Final dense layer for one-dimensional output
    output = layers.Dense(1, activation="sigmoid")(x)

    # Create the model
    model = keras.models.Model(inputs=inp, outputs=output)

    return model


# Example usage:
x_train = None
# Assuming x_train.shape is (900, 19, 64, 64, 1), indicating (samples, frames, height, width, channels)
generator = vapaad(input_shape=(None, *x_train.shape[2:]))


# Example usage:
# Assuming x_train.shape is (900, 19, 64, 64, 1), indicating (samples, frames, height, width, channels)
instructor = instructor_model(input_shape=(None, *x_train.shape[2:]))


# This method returns a helper function to compute cross entropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def generator_loss(fake_output: tf.Tensor) -> tf.Tensor:
    """
    Calculates the loss for the generator model based on its output for generated (fake) images.

    The loss encourages the generator to produce images that the instructor model classifies as real.
    This is achieved by comparing the generator's output for fake images against a target tensor of ones,
    indicating that the ideal output of the generator would be classified as real by the instructor model.

    Args:
    fake_output (tf.Tensor): The generator model's output logits for generated (fake) images.

    Returns:
    tf.Tensor: The loss for the generator model, encouraging it to generate more realistic images.
    """
    return cross_entropy(tf.ones_like(fake_output), fake_output)


def instructor_loss(real_output: tf.Tensor, fake_output: tf.Tensor) -> tf.Tensor:
    """
    Calculates the loss for the instructor model based on its output for real and generated (fake) images.

    The loss is computed as the sum of the cross-entropy losses for the real and fake outputs. For real images,
    the target is a tensor of ones, and for fake images, the target is a tensor of zeros.

    Args:
    real_output (tf.Tensor): The instructor model's output logits for real images.
    fake_output (tf.Tensor): The instructor model's output logits for generated (fake) images.

    Returns:
    tf.Tensor: The total loss for the instructor model, combining the real and fake loss components.
    """
    # Cross-entropy loss for real images (targets are ones)
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    # Cross-entropy loss for fake images (targets are zeros)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    # Sum of real and fake losses
    total_loss = real_loss + fake_loss
    return total_loss


# define two separate optimizer for the generator and the instructor
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
instructor_optimizer = tf.keras.optimizers.Adam(1e-4)


import numpy as np
import tensorflow as tf
import time
from tqdm import tqdm

# Ensure the generator and instructor models, as well as their optimizers,
# are defined outside of this function, at a global level or before calling this train function.


@tf.function
def train_step(images, future_images):
    """
    Performs a single training step for the generator and instructor models.

    This function computes the loss for both the generator and instructor models using the provided
    images and future_images, then updates both models' weights based on these losses.

    Args:
    images: The input images for the generator model.
    future_images: The real images to compare against the generated images by the instructor model.
    """
    # Open a GradientTape scope to record the operations for automatic differentiation.
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        # Generate images from the input images using the generator model.
        generated_images = generator(images, training=True)
        # Get the instructor model's output for the real future images.
        real_output = instructor(future_images, training=True)
        # Get the instructor model's output for the generated images.
        fake_output = instructor(generated_images, training=True)

        # Calculate the loss for the generator based on the instructor's output for the generated images.
        gen_loss = generator_loss(fake_output)
        # Calculate the loss for the instructor based on its output for both real and generated images.
        disc_loss = instructor_loss(real_output, fake_output)

    # Calculate the gradients of the loss with respect to the generator's variables.
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    # Calculate the gradients of the loss with respect to the instructor's variables.
    gradients_of_discriminator = disc_tape.gradient(
        disc_loss, instructor.trainable_variables
    )

    # Apply the gradients to the generator's variables to update its weights.
    generator_optimizer.apply_gradients(
        zip(gradients_of_generator, generator.trainable_variables)
    )
    # Apply the gradients to the instructor's variables to update its weights.
    instructor_optimizer.apply_gradients(
        zip(gradients_of_discriminator, instructor.trainable_variables)
    )


import numpy as np
import tensorflow as tf
import time
from typing import Tuple


def train(
    x_train: np.ndarray, y_train: np.ndarray, epochs: int, batch_size: int = 64
) -> None:
    """
    Trains the model for a given number of epochs with specified batch size.

    This function iterates over the entire dataset for a specified number of epochs,
    randomly selecting batches of data to perform training steps. The selection is random
    and without replacement within each epoch, ensuring diverse exposure of data.

    Args:
    x_train (np.ndarray): The input training data.
    y_train (np.ndarray): The target training data.
    epochs (int): The number of times to iterate over the entire dataset.
    batch_size (int, optional): The number of samples per batch of computation. Defaults to 64.

    Returns:
    None
    """
    # Determine the number of samples in the training dataset.
    n_samples = x_train.shape[0]

    # Iterate over the dataset for the specified number of epochs.
    for epoch in range(epochs):
        start = time.time()  # Record the start time of the epoch.
        indices = np.arange(
            n_samples
        )  # Create an array of indices corresponding to the dataset.
        np.random.shuffle(
            indices
        )  # Shuffle the indices to ensure random batch selection.

        # Iterate over the dataset in batches.
        for i in range(0, n_samples, batch_size):
            selected_indices = np.random.choice(
                indices, size=batch_size, replace=False
            )  # Randomly select indices for the batch.
            x_batch = x_train[selected_indices]  # Extract the batch of input data.
            y_batch = y_train[selected_indices]  # Extract the batch of target data.
            train_step(
                x_batch, y_batch
            )  # Perform a training step with the selected batch.

        # Print the time taken to complete the epoch.
        print(f"Time for epoch {epoch + 1} is {time.time() - start} sec")
