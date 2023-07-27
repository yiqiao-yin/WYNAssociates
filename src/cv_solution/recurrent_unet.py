import tensorflow as tf
from tensorflow.keras.layers import (
    Activation,
    Add,
    BatchNormalization,
    Conv2D,
    Conv2DTranspose,
    Input,
    MaxPooling2D,
    concatenate,
)
from tensorflow.keras.models import Model


def r2_unet(
    filters,
    output_channels,
    width=None,
    height=None,
    input_channels=1,
    conv_layers=2,
    rr_layers=2,
):
    def recurrent_block(layer_input, filters, conv_layers=2, rr_layers=2):
        convs = []
        for i in range(conv_layers - 1):
            a = Conv2D(filters, kernel_size=(3, 3), strides=(1, 1), padding="same")
            convs.append(a)

        d = layer_input
        for i in range(len(convs)):
            a = convs[i]
            d = a(d)
            d = BatchNormalization()(d)
            d = Activation("relu")(d)

        for j in range(rr_layers):
            d = Add()([d, layer_input])
            for i in range(len(convs)):
                a = convs[i]
                d = a(d)
                d = BatchNormalization()(d)
                d = Activation("relu")(d)

        return d

    def RRCNN_block(layer_input, filters, conv_layers=2, rr_layers=2):
        d = Conv2D(filters, kernel_size=(3, 3), strides=(1, 1), padding="same")(
            layer_input
        )
        d1 = recurrent_block(d, filters, conv_layers=conv_layers, rr_layers=rr_layers)
        return Add()([d, d1])

    def deconv2d(layer_input, filters):
        u = Conv2DTranspose(filters, 2, strides=(2, 2), padding="same")(layer_input)
        u = BatchNormalization()(u)
        u = Activation("relu")(u)
        return u

    inputs = Input(shape=(width, height, input_channels))

    conv1 = RRCNN_block(inputs, filters, conv_layers=conv_layers, rr_layers=rr_layers)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = RRCNN_block(
        pool1, filters * 2, conv_layers=conv_layers, rr_layers=rr_layers
    )
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = RRCNN_block(
        pool2, filters * 4, conv_layers=conv_layers, rr_layers=rr_layers
    )
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = RRCNN_block(
        pool3, filters * 8, conv_layers=conv_layers, rr_layers=rr_layers
    )
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = RRCNN_block(
        pool4, filters * 16, conv_layers=conv_layers, rr_layers=rr_layers
    )

    conv6 = deconv2d(conv5, filters * 8)
    up6 = concatenate([conv6, conv4])
    up6 = RRCNN_block(up6, filters * 8, conv_layers=conv_layers, rr_layers=rr_layers)

    conv7 = Conv2DTranspose(filters * 4, 3, strides=(2, 2), padding="same")(up6)
    up7 = concatenate([conv7, conv3])
    up7 = RRCNN_block(up7, filters * 4, conv_layers=conv_layers, rr_layers=rr_layers)

    conv8 = Conv2DTranspose(filters * 2, 3, strides=(2, 2), padding="same")(up7)
    up8 = concatenate([conv8, conv2])
    up8 = RRCNN_block(up8, filters * 2, conv_layers=conv_layers, rr_layers=rr_layers)

    conv9 = Conv2DTranspose(filters, 3, strides=(2, 2), padding="same")(up8)
    up9 = concatenate([conv9, conv1])
    up9 = RRCNN_block(up9, filters, conv_layers=conv_layers, rr_layers=rr_layers)

    output_layer_noActi = Conv2D(
        output_channels, (1, 1), padding="same", activation=None
    )(up9)
    outputs = Activation("sigmoid")(output_layer_noActi)

    model = Model(inputs=inputs, outputs=outputs)

    return model
