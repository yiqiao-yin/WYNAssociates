import time
from datetime import datetime
from typing import Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


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


class VAPAAD:
    """A class to handle video processing with data augmentation and self-attention mechanisms."""

    def __init__(self, input_shape: Tuple[int, int, int]):
        self.input_shape = input_shape
        # Initialize generator and instructor models
        self.gen_main = self.build_generator()
        self.gen_aux = self.build_generator()
        self.instructor = self.build_instructor()
        # Define loss functions and optimizers
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.generator_optimizer = tf.keras.optimizers.Adam(1e-4)
        self.instructor_optimizer = tf.keras.optimizers.Adam(1e-4)

    def build_generator(self) -> keras.Model:
        """
        Constructs the generator model for video processing with data augmentation and self-attention.

        This method is responsible for creating a generator model that performs augmentations on input
        frames and then processes them through ConvLSTM2D layers with self-attention, finally applying a
        convolution across the time dimension to generate output frames.

        The model is part of a generative approach and could be used in tasks such as video frame prediction,
        unsupervised learning, or as a part of a Generative Adversarial Network (GAN).

        Returns:
            A Keras model that takes a sequence of frames as input, augments them via random zooming, rotations,
            and translations, and then outputs processed frames with the same sequence length as the input.

        Note: 'input_shape' should be an attribute of the class instance, and 'SelfAttention' is expected
        to be either a predefined layer in Keras or a custom implementation provided in the code.

        Example usage:
            generator = build_generator()
        """
        # Data augmentation layers intended to increase robustness and generalization
        data_augmentation = keras.Sequential(
            [
                layers.RandomZoom(height_factor=0.05, width_factor=0.05),
                layers.RandomRotation(factor=0.02),
                layers.RandomTranslation(height_factor=0.05, width_factor=0.05),
            ],
            name="data_augmentation",
        )

        # Input layer defining the shape of the input frames
        inp = layers.Input(shape=self.input_shape)
        # Apply time distributed data augmentation which applies the augmentation to each frame independently
        x = layers.TimeDistributed(data_augmentation)(inp)
        # Convolutional LSTM layer with relu activation to capture temporal features
        x = layers.ConvLSTM2D(
            filters=64,
            kernel_size=(5, 5),
            padding="same",
            return_sequences=True,
            activation="relu",
        )(x)
        # Batch normalization to help maintain the stability of the network
        x = layers.BatchNormalization()(x)
        # Self-attention layer for capturing long-range dependencies within the sequences
        x = SelfAttention()(x)
        # Conv3D layer to process the features obtained from previous layers and produce a sequence of frames
        x = layers.Conv3D(
            filters=1, kernel_size=(3, 3, 3), activation="sigmoid", padding="same"
        )(x)

        # Construct the model with the specified input and output tensors
        return keras.models.Model(inputs=inp, outputs=x)

    def build_instructor(self) -> keras.Model:
        """
        Constructs the instructor model with convolutional LSTM and fully connected layers.

        This method specifically builds a video processing instructor model that uses ConvLSTM2D layers,
        followed by self-attention, global average pooling, and dense layers to process the input frames
        and predict a one-dimensional output.

        The architecture is designed for sequential data processing ideal for video or time-series data.

        Returns:
            A compiled Keras model that takes a sequence of frames as input and outputs a
            one-dimensional tensor after processing through ConvLSTM2D, self-attention,
            and dense layers. The output can be interpreted as the probability of a certain
            class or a value depending on the final activation function used (sigmoid in this case).

        Note: 'input_shape' should be an attribute of the class instance, and 'SelfAttention' is
        assumed to be a pre-defined layer or a custom layer implemented elsewhere in the code.

        Example usage:
            model = build_instructor()
        """
        # Input layer defining the shape of the input frames
        inp = layers.Input(shape=self.input_shape)
        # Convolutional LSTM layer with relu activation
        x = layers.ConvLSTM2D(
            filters=64,
            kernel_size=(3, 3),
            padding="same",
            return_sequences=True,
            activation="relu",
        )(inp)
        # Batch Normalization layer
        x = layers.BatchNormalization()(x)
        # Self-attention layer for sequence learning
        x = SelfAttention()(x)
        # Global Average Pooling across the frames to get a feature vector
        x = layers.GlobalAveragePooling3D()(x)
        # Fully connected layers with relu activation
        x = layers.Dense(1024, activation="relu")(x)
        x = layers.Dense(512, activation="relu")(x)
        # Output layer with sigmoid activation for binary classification or regression tasks
        output = layers.Dense(1, activation="sigmoid")(x)

        # Construct the model with specified layers
        return keras.models.Model(inputs=inp, outputs=output)

    def train_step(
        self, images: tf.Tensor, future_images: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Perform a single training step by updating the generator and instructor models.

        This method applies gradient descent to both the generator and the instructor models
        based on the loss computed from the real and generated images.

        Args:
            images (tf.Tensor): A tensor of input images for the current time step provided
                                to the generator model 'gen_main'.
            future_images (tf.Tensor): A tensor of target images for the future time step provided
                                    to the generator model 'gen_aux'.

        Returns:
            Tuple[tf.Tensor, tf.Tensor]: A tuple containing the loss values for the generator model
                                        ('gen_loss') and the instructor model ('inst_loss').

        Note: 'gen_optimizer' and 'inst_optimizer' should be attributes of the class instance.

        The function uses TensorFlow operations and assumes that 'gen_main', 'gen_aux', 'instructor',
        'generator_optimizer', 'instructor_optimizer', 'generator_loss', and 'instructor_loss' are
        defined as attributes of the class in which this method is implemented.
        """
        with tf.GradientTape() as gen_tape, tf.GradientTape() as inst_tape:
            # Generate outputs for both current and future inputs
            output_main = self.gen_main(images, training=True)
            output_aux = self.gen_aux(future_images, training=True)
            real_output = self.instructor(output_aux, training=True)
            fake_output = self.instructor(output_main, training=True)

            # Calculate losses for both models
            gen_loss = self.generator_loss(fake_output)
            inst_loss = self.instructor_loss(real_output, fake_output)

        # Apply gradients to update model weights
        gradients_of_gen = gen_tape.gradient(
            gen_loss, self.gen_main.trainable_variables
        )
        gradients_of_inst = inst_tape.gradient(
            inst_loss, self.instructor.trainable_variables
        )
        self.generator_optimizer.apply_gradients(
            zip(gradients_of_gen, self.gen_main.trainable_variables)
        )
        self.instructor_optimizer.apply_gradients(
            zip(gradients_of_inst, self.instructor.trainable_variables)
        )

        return gen_loss, inst_loss

    def generator_loss(self, fake_output):
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
        return self.cross_entropy(tf.ones_like(fake_output), fake_output)

    def instructor_loss(self, real_output, fake_output):
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
        # Define real_loss and fake_loss
        real_loss = self.cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = self.cross_entropy(tf.zeros_like(fake_output), fake_output)
        return real_loss + fake_loss

    def train(self, x_train, y_train, batch_size=64):
        """
        Trains the model for a specified batch size.

        This function iterates over the entire dataset for a epoch,
        randomly selecting batches of data to perform training steps. The selection is random
        and without replacement within each epoch, ensuring diverse exposure of data.

        Args:
        x_train (np.ndarray): The input training data.
        y_train (np.ndarray): The target training data.
        batch_size (int, optional): The number of samples per batch of computation. Defaults to 64.

        Returns:
        None
        """
        n_samples = x_train.shape[0]
        start = time.time()
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        for i in range(0, n_samples, batch_size):
            if i + batch_size > n_samples:
                continue  # Avoid index error on the last batch if it's smaller than the batch size
            selected_indices = indices[i : i + batch_size]
            x_batch = x_train[selected_indices]
            y_batch = y_train[selected_indices]
            curr_gen_loss, curr_inst_loss = self.train_step(x_batch, y_batch)
            if curr_gen_loss < 0.2:  # Early stopping condition
                print(f"Stopping early at epoch {i + 1}")
                return

            print(
                f"> running: current sample {i + 1}, gen_loss={curr_gen_loss}, inst_loss={curr_inst_loss}, time={time.time() - start} sec"
            )

    def __read_me__(self):
        """
        This function prints a multi-line formatted instruction manual for running a VAPAAD model.

        The instructions include how to inspect the data shapes of training and validation datasets,
        initializing the VAPAAD model, selecting a random subset of the training data for training,
        and finally, running the model with GPU support if available.

        There are no parameters for this function and it doesn't return anything.
        It simply prints the instructional text to the console when called.
        """
        now = datetime.now()
        current_year = now.year
        print(
            f"""
            ## Instructions

            Assume you have data as the follows:

            ```py
            # Inspect the dataset.
            print("Training Dataset Shapes: " + str(x_train.shape) + ", " + str(y_train.shape))
            print("Validation Dataset Shapes: " + str(x_val.shape) + ", " + str(y_val.shape))

            # output
            # Training Dataset Shapes: (900, 19, 64, 64, 1), (900, 19, 64, 64, 1)
            # Validation Dataset Shapes: (100, 19, 64, 64, 1), (100, 19, 64, 64, 1)
            ```

            To run the model, execute the following:
            ```py
            # Initializing a new VAPAAD model
            vapaad_model = VAPAAD(input_shape=(19, 64, 64, 1))

            # Assuming x_train and y_train are already defined and loaded
            num_samples = 64
            indices = np.random.choice(x_train.shape[0], num_samples, replace=True)
            print(indices[0:6])
            x_train_sub = x_train[indices]
            y_train_sub = y_train[indices]
            print(x_train_sub.shape, y_train_sub.shape)

            # Example usage:
            BATCH_SIZE = 3
            if tf.test.gpu_device_name() != '':
                with tf.device('/device:GPU:0'):
                    vapaad_model.train(x_train_sub, y_train_sub, batch_size=BATCH_SIZE)
            else:
                vapaad_model.train(x_train_sub, y_train_sub, batch_size=BATCH_SIZE)
            ```

            Copyright © 2010-{current_year} Present Yiqiao Yin
            """
        )
