import tensorflow as tf
import tensorflow.keras.layers as L
from tensorflow.keras.models import Model


from tensorflow import keras

L = keras.layers


def conv_block(x: keras.layers.Input, num_filters: int) -> keras.layers.Layer:
    """
    Applies two convolutional blocks consisting of Conv2D, BatchNormalization,
    and Activation layers to the input tensor.

    Parameters:
    - x (keras.layers.Input): The input tensor.
    - num_filters (int): Number of filters for the convolutional layers.

    Returns:
    - keras.layers.Layer: The output tensor after applying two sets of Conv2D,
      BatchNormalization, and Activation layers.
    """
    # First convolutional block
    x = L.Conv2D(num_filters, 3, padding="same")(
        x
    )  # Apply a 2D convolution with 'num_filters' filters and a kernel size of 3
    x = L.BatchNormalization()(
        x
    )  # Normalize the activations of the previous layer at each batch
    x = L.Activation("relu")(x)  # Apply the ReLU activation function

    # Second convolutional block
    x = L.Conv2D(num_filters, 3, padding="same")(
        x
    )  # Apply another 2D convolution with the same number of filters and kernel size
    x = L.BatchNormalization()(x)  # Apply batch normalization again
    x = L.Activation("relu")(x)  # Apply ReLU activation function again

    return x  # Return the final tensor


def encoder_block(
    x: keras.layers.Input, num_filters: int
) -> (keras.layers.Layer, keras.layers.Layer):
    """
    Defines an encoder block that applies a convolutional block followed by a MaxPooling operation.

    Parameters:
    - x (keras.layers.Input): The input tensor to the encoder block.
    - num_filters (int): The number of filters to use in the convolutional block.

    Returns:
    Tuple[keras.layers.Layer, keras.layers.Layer]: A tuple containing:
    - The output tensor of the convolutional block.
    - The output tensor of the MaxPooling layer.
    """

    # Apply a convolutional block to the input tensor
    x = conv_block(x, num_filters)

    # Apply max pooling to the result of the convolutional block
    p = L.MaxPool2D((2, 2))(x)

    return x, p  # Return both the output of the conv_block and the max pooled output


def attention_gate(
    g: keras.layers.Layer, s: keras.layers.Layer, num_filters: int
) -> keras.layers.Layer:
    """
    Builds an attention gate which helps the model to focus on important features by
    applying gating mechanisms, used in UNet-like architectures.

    Parameters:
    - g (keras.layers.Layer): The gating tensor usually obtained from a coarser scale.
    - s (keras.layers.Layer): The input tensor which should be focused on by the gate.
    - num_filters (int): The number of filters to use in convolution operations.

    Returns:
    keras.layers.Layer: The output tensor after applying the attention gate to the input tensor `s`.
    """

    # Apply 1x1 convolutions to the gating tensor with batch normalization
    Wg = L.Conv2D(num_filters, 1, padding="same")(g)
    Wg = L.BatchNormalization()(Wg)

    # Apply 1x1 convolutions to the input tensor 's' with batch normalization
    Ws = L.Conv2D(num_filters, 1, padding="same")(s)
    Ws = L.BatchNormalization()(Ws)

    # Element-wise sum of the gated feature map and the input,
    # followed by relu activation
    out = L.Activation("relu")(Wg + Ws)

    # Apply 1x1 convolution followed by a sigmoid activation function to learn the gating coefficients
    out = L.Conv2D(num_filters, 1, padding="same")(out)
    out = L.Activation("sigmoid")(out)

    # Multiply the input tensor 's' by the learned gating coefficients to obtain
    # the final attended output which highlights salient features in 's'
    return out * s


def decoder_block(
    x: keras.layers.Layer, s: keras.layers.Layer, num_filters: int
) -> keras.layers.Layer:
    """
    Decoder block that upsamples the feature map `x`, applies an attention gate to the skip connection `s`,
    concatenates the resulting tensors, and then passes them through a convolution block.

    Parameters:
    - x (keras.layers.Layer): The input tensor from the previous decoder or bottleneck layer.
    - s (keras.layers.Layer): The tensor from the corresponding encoder (skip connection) to be combined
      with `x` after attention gating.
    - num_filters (int): The number of filters to be used in convolutional operations within this block.

    Returns:
    keras.layers.Layer: The output tensor of the decoder block, ready to be fed into the next layer or block.
    """

    # Upsample the input feature map using bilinear interpolation
    x = L.UpSampling2D(interpolation="bilinear")(x)

    # Apply the attention gate to the skip connection 's'
    s = attention_gate(x, s, num_filters)

    # Concatenate the upsampled input with the gated skip connection
    x = L.Concatenate()([x, s])

    # Apply a convolution block to the concatenated tensor
    x = conv_block(x, num_filters)

    return x


def attention_unet(input_shape: tuple) -> keras.Model:
    """
    Constructs an Attention U-Net model based on the specified input shape.

    Parameters:
    - input_shape (tuple): A tuple representing the shape of the input data
      (height, width, channels).

    Returns:
    keras.Model: The constructed Attention U-Net Keras model.
    """

    # Input layer of the U-Net
    inputs = L.Input(input_shape)

    # Encoder blocks that downsample the input and create skip connections
    s1, p1 = encoder_block(
        inputs, 64
    )  # First encoder block with output as skip connection s1 and pooled result p1
    s2, p2 = encoder_block(p1, 128)  # Second encoder block
    s3, p3 = encoder_block(p2, 256)  # Third encoder block

    # Bottleneck convolution block with no pooling
    b1 = conv_block(p3, 512)

    # Decoder blocks that upsample the input and combine it with corresponding skip connections
    d1 = decoder_block(
        b1, s3, 256
    )  # First decoder block using bottleneck output and third skip connection
    d2 = decoder_block(d1, s2, 128)  # Second decoder block
    d3 = decoder_block(d2, s1, 64)  # Third decoder block

    # Output layer with a single feature map; uses a sigmoid activation function for binary segmentation tasks
    outputs = L.Conv2D(1, 1, padding="same", activation="sigmoid")(d3)

    # Building the Keras model instance by specifying inputs and outputs
    model = Model(inputs, outputs, name="Attention-UNET")

    # Returning the constructed Attention U-Net model
    return model
