from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
from tensorflow.keras import layers
from tqdm import tqdm


# Define VapadModel Class Object:
class VapadModel:
    def __init__(
        self,
        num_classes: int = 2,  # Number of output classes
        input_shape: Tuple[int, int, int] = (299, 299, 3),  # Shape of input images
        learning_rate: float = 0.001,  # Learning rate for the optimizer
        weight_decay: float = 0.0001,  # Weight decay for regularization
        batch_size: int = 256,  # Batch size for training
        num_epochs: int = 800,  # Number of epochs for training
        image_size: int = 72,  # Size to resize images to after augmentation
        patch_size: int = 6,  # Size of the patches to extract from images
        projection_dim: int = 64,  # Dimensionality of the projection space
        num_heads: int = 4,  # Number of attention heads
        transformer_units: List[int] = [128, 64],  # Sizes of the transformer layers
        transformer_layers: int = 10,  # Number of transformer layers
        mlp_head_units: List[int] = [
            2048,
            1024,
        ],  # Sizes of the dense layers in the final classifier
    ) -> None:
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
        self.num_patches = (
            image_size // patch_size
        ) ** 2  # Compute the number of patches
        self.data_augmentation = (
            keras.Sequential(  # Define the data augmentation pipeline
                [
                    layers.Normalization(),
                    layers.Resizing(image_size, image_size),
                    layers.RandomFlip("horizontal"),
                    layers.RandomRotation(factor=0.02),
                    layers.RandomZoom(height_factor=0.2, width_factor=0.2),
                ],
                name="data_augmentation",
            )
        )
        self.encoder = self.create_encoder()  # Create the encoder model
        self.decoder = self.create_decoder()  # Create the decoder model

    class Patches(layers.Layer):
        """
        Custom Keras layer for extracting square patches from images.

        This layer takes a 4D tensor of images as input and outputs a 3D tensor containing flattened patches,
        which can be used in subsequent layers of a neural network, e.g., for processing with a transformer model.
        """

        def __init__(self, patch_size: int) -> None:
            """
            Initializes the `Patches` layer with a specific patch size.

            Parameters:
            patch_size (int): The size of the square patches to be extracted.
            """
            super().__init__()
            self.patch_size = patch_size  # Store the patch size for later use.

        def call(self, images: tf.Tensor) -> tf.Tensor:
            """
            Extracts patches from the provided images when the layer is called during the forward pass.

            Parameters:
            images (tf.Tensor): A 4D input tensor with shape (batch_size, height, width, channels).

            Returns:
            tf.Tensor: A 3D output tensor with shape (batch_size, num_patches, patch_dimensions),
                    where `num_patches` is determined by the image dimensions and `patch_size`,
                    and `patch_dimensions` is the flattened dimensionality of each patch.
            """
            batch_size = tf.shape(images)[
                0
            ]  # Extract the dynamic batch size from the incoming tensor.
            # Use TensorFlow's built-in function to extract patches from the input images.
            patches = tf.image.extract_patches(
                images=images,
                sizes=[
                    1,
                    self.patch_size,
                    self.patch_size,
                    1,
                ],  # Define the size of the patches.
                strides=[
                    1,
                    self.patch_size,
                    self.patch_size,
                    1,
                ],  # Define the stride between patches.
                rates=[
                    1,
                    1,
                    1,
                    1,
                ],  # Set the sampling rate for each patch; here it's 1, so no dilution.
                padding="VALID",  # Use 'VALID' padding to ensure patches fit within the image boundaries.
            )
            patch_dims = patches.shape[
                -1
            ]  # Determine the number of elements in each patch after flattening.
            # Reshape the patches to a 3D tensor suitable for further processing.
            patches = tf.reshape(patches, [batch_size, -1, patch_dims])
            return patches  # Return the reshaped patches.

    class PatchEncoder(layers.Layer):
        """
        Custom Keras layer for encoding patches with position information.

        This layer projects the patches to a desired dimension and adds positional embeddings to provide
        order information, which is crucial for models that do not inherently capture sequential data, like transformers.
        """

        def __init__(self, num_patches: int, projection_dim: int) -> None:
            """
            Initializes the `PatchEncoder` layer with the total number of patches and the dimension for projection.

            Parameters:
            num_patches (int): The total number of patches per image.
            projection_dim (int): The embedding size for projecting the patches.
            """
            super().__init__()
            self.num_patches = num_patches  # Store the number of patches for later use.
            self.projection = layers.Dense(
                units=projection_dim
            )  # Dense layer to project patch features.
            self.position_embedding = (
                layers.Embedding(  # Embedding layer for positional information.
                    input_dim=num_patches, output_dim=projection_dim
                )
            )

        def call(self, patch: tf.Tensor) -> tf.Tensor:
            """
            Processes the patches through the dense layer and adds positional embeddings on the forward pass.

            Parameters:
            patch (tf.Tensor): A 3D input tensor with shape (batch_size, num_patches, patch_dim).

            Returns:
            tf.Tensor: An encoded tensor with positional information, same shape as `patch`.
            """
            positions = tf.range(
                start=0, limit=self.num_patches, delta=1
            )  # Create a range of position indices.
            # Project patches to 'projection_dim' dimensions and add positional embeddings.
            encoded = self.projection(patch) + self.position_embedding(positions)
            return (
                encoded  # Return the encoded patches with added positional information.
            )

    def mlp(
        self, x: tf.Tensor, hidden_units: List[int], dropout_rate: float
    ) -> tf.Tensor:
        """
        Multi-layer perceptron function that applies multiple Dense layers with GELU activation and Dropout.

        This function takes an input tensor and sequentially applies Dense and Dropout layers.
        The number of Dense layers is determined by the length of the 'hidden_units' list where each layer's
        units are specified by the corresponding item in the list.

        Parameters:
        x (tf.Tensor): Input tensor to the MLP.
        hidden_units (List[int]): A list of integers specifying the number of units in each dense layer.
        dropout_rate (float): Float between 0 and 1, representing the fraction of the input units to drop.

        Returns:
        tf.Tensor: Output tensor after applying the Dense layers and Dropout layers.
        """
        for (
            units
        ) in hidden_units:  # Loop through the defined sizes of the hidden layers.
            # Create a Dense layer with specified units and GELU activation applied to the previous output.
            x = layers.Dense(units, activation=tf.nn.gelu)(x)
            # Apply Dropout to the output of the Dense layer with the given dropout rate.
            x = layers.Dropout(dropout_rate)(x)
        return x  # Return the final output tensor after passing through all the layers.

    def create_encoder(self) -> keras.Model:
        """
        Create the encoder part of a transformer model.

        This method constructs the encoder model using data augmentation, patch extraction,
        patch encoding, application of multiple transformer layers, and ends with the generation
        of class logits. The model is built using Keras' functional API.

        Returns:
            keras.Model: An instance of keras.Model containing the architecture of the encoder.
        """
        inputs = layers.Input(shape=self.input_shape)  # Input layer for the model.

        augmented = self.data_augmentation(inputs)  # Data augmentation on input images.

        patches = self.Patches(self.patch_size)(
            augmented
        )  # Extracting patches from augmented images.

        encoded_patches = self.PatchEncoder(self.num_patches, self.projection_dim)(
            patches  # Encode the extracted patches.
        )

        # Construct transformer layers.
        for _ in range(self.transformer_layers):
            x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
            attention_output = layers.MultiHeadAttention(
                num_heads=self.num_heads, key_dim=self.projection_dim, dropout=0.1
            )(x1, x1)

            x2 = layers.Add()([attention_output, encoded_patches])
            x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
            x3 = self.mlp(x3, hidden_units=self.transformer_units, dropout_rate=0.1)
            encoded_patches = layers.Add()([x3, x2])

        # Apply final layer normalization.
        representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)

        representation = layers.Flatten()(
            representation
        )  # Flatten the representations to a vector.
        representation = layers.Dropout(0.5)(
            representation
        )  # Apply dropout for regularization.

        # Apply another MLP to get the final feature representation.
        features = self.mlp(
            representation, hidden_units=self.mlp_head_units, dropout_rate=0.5
        )

        logits = layers.Dense(self.num_classes)(
            features
        )  # Output layer for class predictions.

        model = keras.Model(inputs=inputs, outputs=logits)  # Construct the Keras model.

        return model  # Return the constructed encoder model.

    def create_decoder(self) -> keras.Model:
        """
        Create the decoder part of a transformer model.

        This method constructs the decoder model that takes an input vector representing
        class probabilities and upscales it through several deconvolutional layers with
        self-attention to output an image. The model is built using Keras' functional API.

        Returns:
            keras.Model: An instance of keras.Model containing the architecture of the decoder.
        """
        # Define the input layer for the classes' probabilities.
        decoder_inputs = layers.Input(shape=self.num_classes)

        # Dense layer followed by Batch Normalization and LeakyReLU activation.
        x = layers.Dense(7 * 7 * 256, use_bias=False)(decoder_inputs)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)

        # Reshaping the flat tensor to a shape that can be used for the attention mechanism.
        x = layers.Reshape((49, 256))(x)  # 7*7 grid with depth 256.

        # Applying self-attention mechanism on the features.
        attention_output = layers.MultiHeadAttention(num_heads=8, key_dim=64)(x, x, x)
        attention_output = layers.LeakyReLU()(attention_output)

        # Concatenate the attention output with the input feature map.
        x = layers.Concatenate(axis=-1)([x, attention_output])

        # Flattening and then reshaping to fit the next convolutional tranpose layers.
        x = layers.Flatten()(x)
        x = layers.Reshape((7, 7, 512))(
            x
        )  # Shape adjusted to match the output from concatenation.

        # Convolutional transpose blocks with upscaling.
        x = layers.Conv2DTranspose(
            128, (5, 5), strides=(1, 1), padding="same", use_bias=False
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)

        x = layers.Conv2DTranspose(
            64, (5, 5), strides=(2, 2), padding="same", use_bias=False
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)

        # Final deconvolution step that outputs the image with 'tanh' activation.
        x = layers.Conv2DTranspose(
            1, (5, 5), strides=(2, 2), padding="same", use_bias=False, activation="tanh"
        )(x)

        # Construct the Keras model with the defined layers.
        model = keras.Model(inputs=decoder_inputs, outputs=x)

        return model  # Return the constructed decoder model.

    def build_vapad_model(self) -> keras.Model:
        """
        Build the Vision Agumented Prediction w/ Attention Design (VAPAD Model).

        This method constructs the full variational autoencoder using the class's encoder and
        decoder attributes. It connects the output of the encoder to the input of the decoder
        to form the complete model.

        Returns:
            keras.Model: An instance of keras.Model containing the full autoencoder architecture.
        """
        # Retrieve the output from the encoder part of the VAE.
        encoded_features = self.encoder.output

        # Pass the encoder output through the decoder.
        decoded_output = self.decoder(encoded_features)

        # Instantiate the entire VAE model linking its input to the encoder input and output to the decoder output.
        model = keras.Model(inputs=self.encoder.input, outputs=decoded_output)

        return model  # Return the complete VAE model.


# Example usage:
# vapad_model = VapadModel(input_shape=(28, 28, 1))
# vapad_model.build_vapad_model().summary()


# Define discriminator model
def make_discriminator_model() -> tf.keras.Model:
    """
    Creates a discriminator model using the Keras Sequential API tailored for classifying images.

    The discriminator is a convolutional neural network that classifies images as real or fake.
    It works by down-sampling the input images twice using convolutional layers and then passing
    the result through a dense layer to obtain a single output. LeakyReLU activations and Dropout
    layers are used to add non-linearity and prevent overfitting.

    Returns:
        tf.keras.Model: A compiled Keras Sequential model representing the discriminator.
    """

    # Initialize a Keras Sequential model.
    model = tf.keras.Sequential()

    # First Conv2D layer with 64 filters, kernel size of 5x5, stride of 2x2, 'same' padding, and input shape specified for 28x28 images with 1 channel.
    model.add(
        tf.keras.layers.Conv2D(
            64, (5, 5), strides=(2, 2), padding="same", input_shape=[28, 28, 1]
        )
    )
    # Add LeakyReLU activation function.
    model.add(tf.keras.layers.LeakyReLU())
    # Add dropout for regularization with a rate of 0.3.
    model.add(tf.keras.layers.Dropout(0.3))

    # Second Conv2D layer with 128 filters and the same kernel size, stride, and padding as the first Conv2D layer.
    model.add(tf.keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding="same"))
    # Add LeakyReLU activation function.
    model.add(tf.keras.layers.LeakyReLU())
    # Add dropout for regularization with a rate of 0.3.
    model.add(tf.keras.layers.Dropout(0.3))

    # Flatten the output of the last Conv2D layer to feed it into a Dense layer.
    model.add(tf.keras.layers.Flatten())
    # Output Dense layer with a single neuron to provide the classification output.
    model.add(tf.keras.layers.Dense(1))

    # Return the constructed model.
    return model


# Define discriminator model
def make_discriminator_model(
    input_shape: tuple = (28, 28, 1),
    num_blocks: int = 4,
    num_heads: int = 8,
    ff_dim: int = 512,
    dropout_rate: float = 0.1,
) -> tf.keras.Model:
    """
    Builds a discriminator model incorporating convolutional layers followed by Transformer Encoder blocks.

    The model takes images and processes them through convolutions for feature extraction
    prior to applying multiple Transformer Encoder blocks for further processing. The output is generated
    through a final Dense layer after global average pooling.

    Parameters:
        input_shape (tuple): Shape of the input images, defaults to (28, 28, 1).
        num_blocks (int): Number of Transformer Encoder blocks to be used, defaults to 4.
        num_heads (int): Number of attention heads within each Multi-Head Attention layer, defaults to 8.
        ff_dim (int): The dimensionality of the inner Feed Forward layer, defaults to 512.
        dropout_rate (float): Dropout rate for regularization, defaults to 0.1.

    Returns:
        tf.keras.Model: A TensorFlow Keras model object representing the discriminator neural network.
    """

    # Define the input layer with the given shape
    inputs = tf.keras.Input(shape=input_shape)
    x = inputs

    # Convolutional base for initial feature extraction
    x = layers.Conv2D(64, (3, 3), strides=(2, 2), padding="same")(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv2D(128, (3, 3), strides=(2, 2), padding="same")(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.3)(x)

    # Add Transformer Encoder Blocks
    for _ in range(num_blocks):
        # Apply Multi-Head Self-Attention
        attention_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=64)(
            x, x
        )
        attention_output = layers.Dropout(dropout_rate)(attention_output)
        # Include Residual Connection followed by Layer Normalization
        attention_output = layers.LayerNormalization(epsilon=1e-6)(x + attention_output)

        # Construct Feed Forward Network part of the Transformer block
        ffn_output = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(
            attention_output
        )
        ffn_output = layers.Conv1D(filters=x.shape[-1], kernel_size=1)(ffn_output)
        ffn_output = layers.Dropout(dropout_rate)(ffn_output)
        # Include another Residual Connection followed by Layer Normalization
        x = layers.LayerNormalization(epsilon=1e-6)(attention_output + ffn_output)

    # Reduce dimensions and aggregate features using Global Average Pooling
    x = layers.GlobalAveragePooling2D()(x)

    # Final output layer with a single neuron to perform classification
    outputs = layers.Dense(1)(x)

    # Create the Keras model with the specified inputs and outputs
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    return model


# This method returns a helper function to compute cross entropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


# Define loss for generator
def generator_loss(fake_output: tf.Tensor) -> tf.Tensor:
    """
    Calculates the loss for the generator in a GAN by comparing the fake output
    to an array of ones, representing the target for successful fake image generation.

    The generator's goal is to produce images that are indistinguishable from real ones,
    which would result in the discriminator outputting a value close to one for each fake image.

    Parameters:
        fake_output (tf.Tensor): The discriminator's output probabilities for the fake images.

    Returns:
        tf.Tensor: Loss value for the generator.
    """

    # Create a tensor of the same shape as fake_output but filled with ones,
    # indicating that the generator wants to fool the discriminator into thinking
    # the generated images are real (which should be classified as ones).
    return cross_entropy(tf.ones_like(fake_output), fake_output)


# Define loss for discriminator
def discriminator_loss(real_output: tf.Tensor, fake_output: tf.Tensor) -> tf.Tensor:
    """
    Computes the discriminator's loss in a GAN by summing up the losses for the real and fake images.

    The discriminator's goal is to correctly classify real images as real (output close to one) and
    fake images as fake (output close to zero). Therefore, we compare its outputs on real images to ones
    and its outputs on fake images to zeros.

    Parameters:
        real_output (tf.Tensor): The discriminator's output probabilities for the real images.
        fake_output (tf.Tensor): The discriminator's output probabilities for the fake images.

    Returns:
        tf.Tensor: The total loss for the discriminator, summed over both real and fake images.
    """

    # Calculate loss on real images by comparing discriminator output to an array of ones
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)

    # Calculate loss on fake images by comparing discriminator output to an array of zeros
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)

    # Sum both losses to get the total loss for the discriminator
    total_loss = real_loss + fake_loss

    return total_loss


# define two separate optimizer for the generator and the discriminator
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)


# Helper function
def _mask_gen_helper_(
    length_to_block: int = 20, input_shape: Tuple[int, int] = (28, 28)
) -> tf.Tensor:
    """
    Generates a mask tensor that blocks a specified square area of an image.

    This function creates a 2D mask with ones and then sets a square region starting from
    the provided offset length_to_block to zeros. The mask is then flattened and converted
    to a TensorFlow tensor. This can be used to simulate occlusion in images.

    Parameters:
        length_to_block (int): The size of the square block (in pixels) starting from the top-left
            corner that will be set to zero.
        input_shape (Tuple[int, int]): The shape of the 2D input space where the mask will be applied.

    Returns:
        tf.Tensor: A flattened TensorFlow tensor representing the mask with the specified blocked area.

    Note:
        The values are set to float32 to ensure compatibility when applying this mask to other tensors.
    """

    # Create a numpy array with ones in the given input shape and set data type to float32
    mask = np.ones(input_shape, dtype=np.float32)

    # Block off a square region in the mask by setting it to zeros
    mask[length_to_block:, length_to_block:] = 0

    # Flatten the mask to a 1D array
    mask = mask.reshape(
        -1,
    )

    # Convert the numpy array mask to a TensorFlow tensor
    mask = tf.convert_to_tensor(mask)

    return mask


# create generator model
BATCH_SIZE = 64
vapad_model = VapadModel(input_shape=(28, 28, 1))
generator = vapad_model.build_vapad_model()
# define discriminator
discriminator = make_discriminator_model()


# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def train_step(images: tf.Tensor, twoDim: bool = True, print_loss: bool = False):
    """
    Performs one training step for the generator and discriminator in a GAN setup.

    This function applies a mask to the incoming batch of images, then uses these masked images
    to generate new images via the generator. Both the original and generated images are passed
    through the discriminator. The losses for both the generator and discriminator are calculated,
    gradients are computed, and optimizers are used to update the models' weights.

    Parameters:
        images (tf.Tensor): A batch of images to be used during the training step.
        twoDim (bool, optional): If True, reshapes the flat masked images back into 2D before being fed to the generator.
            Defaults to True.
        print_loss (bool, optional): If True, prints out the total loss after the step is completed.
            Defaults to False.

    Note:
        BATCH_SIZE and other variables like generator, discriminator, generator_optimizer,
        discriminator_optimizer must be defined outside this function as they are not passed as arguments.
        This function assumes that `_mask_gen_helper_` has been defined previously and is accessible within its scope.
    """

    # Flatten and mask the images instead of using random noise
    mask = _mask_gen_helper_()
    flat_images = tf.reshape(images, [BATCH_SIZE, 784])
    masked_images = flat_images * mask

    if twoDim:
        # Reshape the masked images back to 2D, if `twoDim` is set to True
        masked_images = tf.reshape(masked_images, [BATCH_SIZE, 28, 28, 1])

    # Record operations for automatic differentiation.
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        # Generating images through the generator network
        generated_images = generator(masked_images, training=True)

        # Discriminator decisions for real images and generated images
        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        # Calculate generator and discriminator loss
        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    # Compute gradient of losses and apply updates to model weights
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(
        disc_loss, discriminator.trainable_variables
    )
    generator_optimizer.apply_gradients(
        zip(gradients_of_generator, generator.trainable_variables)
    )
    discriminator_optimizer.apply_gradients(
        zip(gradients_of_discriminator, discriminator.trainable_variables)
    )

    # Calculate and print the total loss if `print_loss` is True
    total_loss = gen_loss + disc_loss
    if print_loss:
        tf.print("Total Loss:", total_loss)


# Define make_inference function
def make_inference(
    input_image, generator: tf.keras.Model, twoDim: bool = True
) -> Tuple:
    """
    Perform inference using the generator model on an input image after applying a mask.

    This function takes an input image and a pretrained generator model and first applies a masking operation
    to the image. The masked image is then fed into the generator to produce an output which should be a
    completion of the original masked input.

    Parameters:
        input_image (Union[np.ndarray, tf.Tensor]): The raw image data that will be used for inference.
            Can be either a NumPy ndarray or a TensorFlow Tensor.
        generator (tf.keras.Model): Pretrained generator model used for image generation.
        twoDim (bool, optional): If set to True, reshapes the flat masked image back to its original 2D shape
            before feeding it to the generator. Defaults to True.

    Returns:
        tuple: A tuple containing the masked input image and the image generated by the generator model.

    Note:
        `_mask_gen_helper_` is assumed to be defined elsewhere and accessible in this scope. It is used to generate
        a mask for the input image.

        This function assumes the input_image has the correct dimensions for the generator and that
        the generator was trained on flattened images.
    """

    # Convert input to TensorFlow Tensor if not already one
    if not isinstance(input_image, tf.Tensor):
        input_image = tf.convert_to_tensor(input_image, dtype=tf.float32)

    # Apply the masking process to the input image
    mask = _mask_gen_helper_()
    flat_image = tf.reshape(
        input_image, [784]
    )  # Assuming input image is 28x28, flattening to 1D
    masked_image = flat_image * mask  # Element-wise multiplication with the mask

    if twoDim:
        # Optionally reshape back to 2D format
        masked_image = tf.reshape(masked_image, [28, 28, 1])

    # Add a batch dimension for compatibility with the generator model
    masked_image_with_batch = tf.expand_dims(masked_image, 0)

    # Generate the image from the masked image using the generator
    generated_image = generator(masked_image_with_batch, training=False)

    return masked_image, generated_image
