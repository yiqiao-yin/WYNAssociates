# Import Libraries
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Import Tensorflow
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, Dropout, Lambda
from keras import backend as K

# Import Other Libraries
from scipy import stats

# Import Libraries
import math

# define class
class YinsCV:
    
    print("---------------------------------------------------------------------")
    print(
        """
        Yin's Deep Learning Package 
        Copyright © W.Y.N. Associates, LLC, 2009 – Present
        For more information, please go to https://wyn-associates.com/
        """ )
    print("---------------------------------------------------------------------")

    # define jacard loss
    def jacard_coef(y_true, y_pred):
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        return (intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + 1.0)
        
    # define unet
    def multi_unet_model(n_classes=4, IMG_HEIGHT=256, IMG_WIDTH=256, IMG_CHANNELS=1, useSCALE=FALSE):
        
        # Build the model
        inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
        if useSCALE:
            s = Lambda(lambda x: x / 255)(inputs)   #No need for this if we normalize our inputs beforehand
        else:
            s = inputs

        # Contraction path
        c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
        c1 = Dropout(0.2)(c1)  # Original 0.1
        c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
        p1 = MaxPooling2D((2, 2))(c1)

        c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
        c2 = Dropout(0.2)(c2)  # Original 0.1
        c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
        p2 = MaxPooling2D((2, 2))(c2)

        c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
        c3 = Dropout(0.2)(c3)
        c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
        p3 = MaxPooling2D((2, 2))(c3)

        c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
        c4 = Dropout(0.2)(c4)
        c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
        p4 = MaxPooling2D(pool_size=(2, 2))(c4)

        c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
        c5 = Dropout(0.3)(c5)
        c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

        # Expansive path
        u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
        u6 = concatenate([u6, c4])
        c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
        c6 = Dropout(0.2)(c6)
        c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)

        u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
        u7 = concatenate([u7, c3])
        c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
        c7 = Dropout(0.2)(c7)
        c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)

        u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
        u8 = concatenate([u8, c2])
        c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
        c8 = Dropout(0.2)(c8)  # Original 0.1
        c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)

        u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
        u9 = concatenate([u9, c1], axis=3)
        c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
        c9 = Dropout(0.2)(c9)  # Original 0.1
        c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)

        outputs = Conv2D(n_classes, (1, 1), activation='softmax')(c9)

        model = Model(inputs=[inputs], outputs=[outputs])

        #NOTE: Compile the model in the main program to make it easy to test with various loss functions
        #model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        #model.summary()

        return model
    
    # define unet (inception style)
    def unet_inceptionstyle_model(
        x_train=None,
        y_train=None,
        x_val=None, 
        y_val=None,
        img_size = (128, 128, 1),
        num_classes = 2,
        ENC_PARAM = [2**i for i in range(5, 10)],
        optimizer="adam", 
        loss="sparse_categorical_crossentropy",
        epochs=400,
        figsize=(12,6),
        name_of_file = "model.png",
        name_of_model = "this_model",
        plotModel = True,
        useGPU = True,
        useCallback = False,
        augmentData = True,
        verbose = True,
        which_layer = None,
        X_for_internal_extraction = None,
        featurewise_center=True,
        featurewise_std_normalization=True,
        rescale=1,
        shear_range=0.3,
        zoom_range=0.2,
        rotation_range=90,
        horizontal_flip=True,
        vertical_flip=True
        ):

        # define unet
        def get_model(img_size, num_classes, ENC_PARAM):
            inputs = keras.Input(shape=img_size)

            ### [First half of the network: downsampling inputs] ###
            ENC_PARAM = ENC_PARAM

            # Entry block
            x = layers.Conv2D(ENC_PARAM[0], 3, strides=2, padding="same")(inputs)
            x = layers.BatchNormalization()(x)
            x = layers.Activation("relu")(x)

            previous_block_activation = x  # Set aside residual

            # Blocks 1, 2, 3 are identical apart from the feature depth.
            for filters in ENC_PARAM[1::]:
                x = layers.Activation("relu")(x)
                x = layers.SeparableConv2D(filters, 3, padding="same")(x)
                x = layers.BatchNormalization()(x)

                x = layers.Activation("relu")(x)
                x = layers.SeparableConv2D(filters, 3, padding="same")(x)
                x = layers.BatchNormalization()(x)

                x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

                # Project residual
                residual = layers.Conv2D(filters, 1, strides=2, padding="same")(
                    previous_block_activation
                )
                x = layers.add([x, residual])  # Add back residual
                previous_block_activation = x  # Set aside next residual

            ### [Second half of the network: upsampling inputs] ###
            DEC_PARAM = ENC_PARAM[::-1]

            for filters in DEC_PARAM:
                x = layers.Activation("relu")(x)
                x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
                x = layers.BatchNormalization()(x)

                x = layers.Activation("relu")(x)
                x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
                x = layers.BatchNormalization()(x)

                x = layers.UpSampling2D(2)(x)

                # Project residual
                residual = layers.UpSampling2D(2)(previous_block_activation)
                residual = layers.Conv2D(filters, 1, padding="same")(residual)
                x = layers.add([x, residual])  # Add back residual
                previous_block_activation = x  # Set aside next residual

            # Add a per-pixel classification layer
            outputs = layers.Conv2D(num_classes, 3, activation="softmax", padding="same")(x)

            # Define the model
            model = keras.Model(inputs, outputs)

            # return
            return model

        # Build model
        model = get_model(img_size=img_size, num_classes=2, ENC_PARAM=ENC_PARAM)

        # Plot Model
        if plotModel:
                # name_of_file = "model.png"
                tf.keras.utils.plot_model(model, to_file=name_of_file, show_shapes=True, expand_nested=True)

        # compile
        # Configure the model for training.
        # We use the "sparse" version of categorical_crossentropy
        # because our target data is integers.
        model.compile(
            # default:
            optimizer=optimizer, 
            loss=loss, 
            metrics=['accuracy']  )

        # callbacks
        callbacks = [ keras.callbacks.ModelCheckpoint(name_of_model+".h5", save_best_only=True) ]
        # note
        # https://www.tensorflow.org/guide/keras/save_and_serialize
        # when need to use the saved model, you can call it by using 
        # from tensorflow import keras
        # model = keras.models.load_model('path/to/location')
        
        # if we need data augmentation
        # from tf.keras.preprocessing.image import ImageDataGenerator
        
        # create generator for batches that centers mean and std deviation of training data
        # featurewise_center=True
        # featurewise_std_normalization=True
        # rescale=1
        # shear_range=0.3
        # zoom_range=0.2
        # rotation_range=90
        # horizontal_flip=True
        # vertical_flip=True
        datagen = ImageDataGenerator(
            featurewise_center=featurewise_center,
            featurewise_std_normalization=featurewise_std_normalization,
            rescale=rescale,
            shear_range=shear_range,
            zoom_range=zoom_range,
            rotation_range=rotation_range,
            horizontal_flip=horizontal_flip,
            vertical_flip=vertical_flip  )

        # fit data to the generator
        datagen.fit(x_train) # <= this should only be training data otherwise it is cheating!

        # Source:
        # Here are different ways of augmenting your training data
        # https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
        
        # training log for data augmentation
        class LossAndErrorPrintingCallback(keras.callbacks.Callback):
            def on_train_batch_end(self, batch, logs=None):
                print( "Up to batch {}, the average loss is {:7.2f}.".format(batch, logs["loss"]) )

            def on_test_batch_end(self, batch, logs=None):
                print( "Up to batch {}, the average loss is {:7.2f}.".format(batch, logs["loss"]) )

            def on_epoch_end(self, epoch, logs=None):
                print(
                    "The average loss for epoch {} is {:7.2f} "
                    "and mean absolute error is {:7.2f}.".format(
                        epoch, logs["loss"], logs["val_loss"] ) )
                
        # training early stop for data augmentation
        class EarlyStoppingAtMinLoss(keras.callbacks.Callback):
            """Stop training when the loss is at its min, i.e. the loss stops decreasing.

          Arguments:
              patience: Number of epochs to wait after min has been hit. After this
              number of no improvement, training stops.
          """

            def __init__(self, patience=0):
                super(EarlyStoppingAtMinLoss, self).__init__()
                self.patience = patience
                # best_weights to store the weights at which the minimum loss occurs.
                self.best_weights = None

            def on_train_begin(self, logs=None):
                # The number of epoch it has waited when loss is no longer minimum.
                self.wait = 0
                # The epoch the training stops at.
                self.stopped_epoch = 0
                # Initialize the best as infinity.
                self.best = np.Inf

            def on_epoch_end(self, epoch, logs=None):
                current = logs.get("loss")
                if np.less(current, self.best):
                    self.best = current
                    self.wait = 0
                    # Record the best weights if current results is better (less).
                    self.best_weights = self.model.get_weights()
                else:
                    self.wait += 1
                    if self.wait >= self.patience:
                        self.stopped_epoch = epoch
                        self.model.stop_training = True
                        print("Restoring model weights from the end of the best epoch.")
                        self.model.set_weights(self.best_weights)

            def on_train_end(self, logs=None):
                if self.stopped_epoch > 0:
                    print("Epoch %05d: early stopping" % (self.stopped_epoch + 1))

        # fit
        if useGPU:
                device_name = tf.test.gpu_device_name()
                if device_name != '/device:GPU:0':
                  raise SystemError('GPU device not found')
                print('Found GPU at: {}'.format(device_name))

                # use GPU
                with tf.device('/device:GPU:0'):
                    # Train the model, doing validation at the end of each epoch.
                    if augmentData:
                        if useCallback:
                            history = model.fit_generator(
                                datagen.flow(x_train, y_train),
                                epochs=epochs, 
                                validation_data=(x_val, y_val),
                                callbacks=callbacks )
                                # callbacks=[LossAndErrorPrintingCallback(), EarlyStoppingAtMinLoss(), tf.keras.callbacks.ModelCheckpoint("yin_segmentation.h5", save_best_only=True)] )
                        else:
                            history = model.fit_generator(
                                datagen.flow(x_train, y_train),
                                epochs=epochs, 
                                validation_data=(x_val, y_val),
                                # callbacks=callbacks 
                            )
                                # callbacks=[LossAndErrorPrintingCallback(), EarlyStoppingAtMinLoss(), tf.keras.callbacks.ModelCheckpoint("yin_segmentation.h5", save_best_only=True)] )
                    else:
                        if useCallback:
                            history = model.fit(
                                x_train, y_train, 
                                epochs=epochs, 
                                validation_data=(x_val, y_val), 
                                callbacks=callbacks)
                        else:
                            history = model.fit(
                                x_train, y_train, 
                                epochs=epochs, 
                                validation_data=(x_val, y_val) )
        else:         
                # Train the model, doing validation at the end of each epoch.
                if augmentData:                    
                    if useCallback:
                        history = model.fit_generator(
                            datagen.flow(x_train, y_train),
                            epochs=epochs, 
                            validation_data=(x_val, y_val),
                            callbacks=callbacks )
                            # callbacks=[LossAndErrorPrintingCallback(), EarlyStoppingAtMinLoss(), tf.keras.callbacks.ModelCheckpoint("yin_segmentation.h5", save_best_only=True)] )
                    else:
                        history = model.fit_generator(
                            datagen.flow(x_train, y_train),
                            epochs=epochs, 
                            validation_data=(x_val, y_val),
                            # callbacks=callbacks 
                        )
                                # callbacks=[LossAndErrorPrintingCallback(), EarlyStoppingAtMinLoss(), tf.keras.callbacks.ModelCheckpoint("yin_segmentation.h5", save_best_only=True)] )
                else:
                    if useCallback:
                        history = model.fit_generator(
                            datagen.flow(x_train, y_train),
                            epochs=epochs, 
                            validation_data=(x_val, y_val),
                            callbacks=callbacks )
                            # callbacks=[LossAndErrorPrintingCallback(), EarlyStoppingAtMinLoss(), tf.keras.callbacks.ModelCheckpoint("yin_segmentation.h5", save_best_only=True)] )
                    else:
                        history = model.fit_generator(
                            datagen.flow(x_train, y_train),
                            epochs=epochs, 
                            validation_data=(x_val, y_val),
                            # callbacks=callbacks 
                        )
                            # callbacks=[LossAndErrorPrintingCallback(), EarlyStoppingAtMinLoss(), tf.keras.callbacks.ModelCheckpoint("yin_segmentation.h5", save_best_only=True)] )
        
        # inference
        # with a Sequential model
        if verbose:
            print('Length of internal layers: ' + str(len(model.layers)))
            print('You can input an X and extract output but within any internal layer.')
            print('Please choose a positive interger up to ' + str(len(model.layers)-1))
        if which_layer != None:
            from tensorflow.keras import backend as K
            get_internal_layer_fct = K.function([model.layers[0].input], [model.layers[which_layer].output])
            internal_layer_output = get_internal_layer_fct([np.asarray(X_for_internal_extraction)])[0]
        else:
            internal_layer_output = "Please enter which_layer and X_for_internal_extraction to obtain this."

        # plot loss
        import matplotlib.pyplot as plt
        plt.figure(figsize=figsize)
        plt.plot(history.history['loss'], label = 'training loss')
        plt.plot(history.history['val_loss'], label = 'validating loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc='lower right') # specify location of the legend

        # prediction
        # make predictions using validating set
        y_hat_train_ = model.predict(x_train)
        y_hat_test_ = model.predict(x_val)

        # plt.figure(figsize=(28, 16))
        # for i in range(10):
        #     plt.subplot(1,25,i+1)
        #     plt.imshow(y_val[i][:, :, 0], cmap='gist_gray_r') # plt.cm.binary

        # plt.show()

        # plt.figure(figsize=(28, 16))
        # for i in range(10):
        #     plt.subplot(1,25,i+1)
        #     plt.imshow(x_val[i][:, :, 0], cmap='gist_gray_r') # plt.cm.binary

        # plt.show()

        # output
        return {
            'Data': {
                'x_train': x_train,
                'y_train': y_train,
                'x_val': x_val, 
                'y_val': y_val
            },
            'Model': model,
            'History': history,
            'Extracted Internal Layer': {
                    'internal_layer': internal_layer_output
            },
            'Prediction': {
                'y_hat_train_': y_hat_train_,
                'y_hat_train_': y_hat_train_
            }
        }

    # define
    def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
        # First, we create a model that maps the input image to the activations
        # of the last conv layer as well as the output predictions
        grad_model = tf.keras.models.Model(
            [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
        )

        # Then, we compute the gradient of the top predicted class for our input image
        # with respect to the activations of the last conv layer
        # The *pred_indx* points to the index of the class in the model. 
        # Note that for one object, we can select None so the algorithm
        # automatically points to the maximum probability class, e.g. argmax.
        # However, if there are more than one objects in the picture, we 
        # need to use pred_index to select the index of the actual class
        # in the model. In other words, this function is programmed to be able 
        # to detect any class in the model.
        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = grad_model(img_array)
            if pred_index is None:
                pred_index = tf.argmax(preds[0])
            class_channel = preds[:, pred_index]

        # This is the gradient of the output neuron (top predicted or chosen)
        # with regard to the output feature map of the last conv layer
        grads = tape.gradient(class_channel, last_conv_layer_output)

        # This is a vector where each entry is the mean intensity of the gradient
        # over a specific feature map channel. This is the usual empirical mean
        # that we do according。
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # We multiply each channel in the feature map array
        # by "how important this channel is" with regard to the top predicted class
        # then sum all the channels to obtain the heatmap class activation
        last_conv_layer_output = last_conv_layer_output[0]
        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
        # heatmap = tf.squeeze(heatmap) # Removes dimensions of size 1 from the shape of a tensor

        # # For visualization purpose, we will also normalize the heatmap between 0 & 1
        # heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        return heatmap

    # define
    def superimposedImages(
        img = None, # array_2D3D
        heatmap = None, # array_2D
        color_grad = "rainbow",
        alpha=.4,
        useOverlay=True):

        # Rescale heatmap to a range 0-255
        heatmap = np.round(np.multiply(heatmap, 255)).astype(int)

        # Use jet colormap to colorize heatmap
        # https://matplotlib.org/3.5.0/tutorials/colors/colormaps.html
        jet = cm.get_cmap("jet")

        # Use RGB values of the colormap
        jet_colors = jet(np.arange(256))[:, :3]
        jet_heatmap = jet_colors[heatmap]

        # Create an image with RGB colorized heatmap
        jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
        jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
        jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)

        # Superimpose the heatmap on original image
        if useOverlay:
            superimposed_img = jet_heatmap * alpha + img
        else:
            superimposed_img = jet_heatmap * alpha + img * (1 - alpha)
        superimposed_img_pil = tf.keras.preprocessing.image.array_to_img(superimposed_img)
        superimposed_img_ar = np.asarray(superimposed_img)/255

        # output
        return {
          'jet_heatmap': jet_heatmap,
          'pil_format': superimposed_img_pil,
          'array_format': superimposed_img_ar
        }
