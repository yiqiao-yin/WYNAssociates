from typing import Tuple

import tensorflow as tf
from tensorflow import Tensor  # Importing Tensor type for type hints.
from tensorflow.keras.layers import Activation, BatchNormalization, Conv2D, MaxPool2D


def conv_block(input: tf.Tensor, num_filters: int) -> tf.Tensor:
    """
    Applies a 2D convolutional neural network block to the input tensor.

    Args:
        input (tf.Tensor): Input tensor.
        num_filters (int): Number of filters in the convolutional layers.

    Returns:
        A tensor resulting from applying the CNN block to the input tensor.
    """

    # Apply a convolutional layer with `num_filters` filters and 3x3 kernel size with padding same
    x = Conv2D(num_filters, 3, padding="same")(input)

    # Apply batch normalization layer
    x = BatchNormalization()(x)

    # Apply ReLU activation function element-wise
    x = Activation("relu")(x)

    # Again apply a convolutional layer with `num_filters` filters and 3x3 kernel size with padding same
    x = Conv2D(num_filters, 3, padding="same")(x)

    # Apply batch normalization layer
    x = BatchNormalization()(x)

    # Apply ReLU activation function element-wise
    x = Activation("relu")(x)

    # Return the resulting tensor
    return x


def encoder_block(input: tf.Tensor, num_filters: int) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Constructs an encoder block.

    Args:
        input (Tensor): Input tensor.
        num_filters (int): Number of filters in the convolutional layers.

    Returns:
        A tuple of two tensors: output of the convolutional block and max-pooled output from the same block.
    """

    # Apply a conv_block to the input tensor
    x = conv_block(input, num_filters)

    # Apply 2x2 MaxPooling to the output of conv_block
    p = MaxPool2D((2, 2))(x)

    # Return two tensors as a tuple
    return x, p


def decoder_block(
    input: tf.Tensor, skip_features: tf.Tensor, num_filters: int
) -> tf.Tensor:
    """
    Constructs a decoder block.

    Args:
        input (Tensor): Input tensor.
        skip_features (Tensor): Tensor from the encoder block that will be concatenated with the input tensor.
        num_filters (int): Number of filters in the convolutional layers.

    Returns:
        The output tensor after passing through the decoder block.
    """

    # Apply transpose convolution with specified parameters to upsample
    x = tf.keras.layers.Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(
        input
    )

    # Concatenate the result of transposed convolution and skip features
    x = tf.keras.layers.Concatenate()([x, skip_features])

    # Apply conv_block to concatenated tensor
    x = conv_block(x, num_filters)

    # Return the output tensor
    return x


def build_unet(input_shape: tuple, output_dim: int) -> tf.keras.Model:
    """
    Constructs a U-Net model with auxiliary classification.

    Args:
        input_shape (tuple): Input tensor shape as a tuple of integers.

    Returns:
        A keras Model object for the constructed U-Net model.
    """

    # Define input layer of specified shape
    inputs = tf.keras.Input(input_shape)

    # Pass through encoding blocks to get skip features and poolings at each level
    s1, p1 = encoder_block(inputs, 32)
    s2, p2 = encoder_block(p1, 64)
    s3, p3 = encoder_block(p2, 128)
    s4, p4 = encoder_block(p3, 256)
    s5, p5 = encoder_block(p4, 512)

    # Pass through convolutions and fully connected layers for auxiliary classification task
    b0 = conv_block(p5, 512)
    flt_layer = tf.keras.layers.Flatten()(b0)
    aux_dense_layer1 = tf.keras.layers.Dense(128, "relu")(flt_layer)
    aux_dense_layer2 = tf.keras.layers.Dense(128, "relu")(aux_dense_layer1)
    aux_dense_layer3 = tf.keras.layers.Dense(output_dim, "softmax")(aux_dense_layer2)

    # Pass through decoding blocks to upsample and generate output mask
    d0 = decoder_block(b0, s5, 512)
    d1 = decoder_block(d0, s4, 256)
    d2 = decoder_block(d1, s3, 128)
    d3 = decoder_block(d2, s2, 64)
    d4 = decoder_block(d3, s1, 32)

    # Generate final output mask using convolutional layer with sigmoid activation and return the model
    outputs = tf.keras.layers.Conv2D(1, 1, padding="same", activation="sigmoid")(d4)
    model = tf.keras.Model(inputs, [outputs, aux_dense_layer2], name="UNetClassifier")

    return model
