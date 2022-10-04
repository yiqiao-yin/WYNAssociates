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

# Import Other Libraries
from scipy import stats

# Import Libraries
import math

# Import Libraries: Scalecast
# library
import boto3
import botocore 
import pandas as pd 
import sagemaker
from sagemaker import get_execution_role 
from sagemaker.session import Session

# import
import os
import numpy as np
import pandas as pd
import seaborn as sns
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm

# import tensorflow
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import EarlyStopping

# import scalecast
from scalecast.Forecaster import Forecaster
from sklearn.metrics import mean_absolute_percentage_error


# define class
class YinsDL:
    
    print("---------------------------------------------------------------------")
    print(
        """
        Yin's Deep Learning Package 
        Copyright © W.Y.N. Associates, LLC, 2009 – Present
        For more information, please go to https://wyn-associates.com/
        """ )
    print("---------------------------------------------------------------------")

   # Define function
    def NN3_Classifier(
        X_train, y_train, X_test, y_test, 
        l1_act='relu', l2_act='relu', l3_act='softmax',
        layer1size=128, layer2size=64, layer3size=2,
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
        num_of_epochs=10,
        plotROC=True,
        verbose=True):
        
        """
        MANUAL:
        
        # One can use the following example.
        house_sales = pd.read_csv('../data/kc_house_data.csv')
        house_sales.head(3)
        house_sales = house_sales.drop(['id', 'zipcode', 'lat', 'long', 'date'], axis=1)
        house_sales.info()

        X_all = house_sales.drop('price', axis=1)
        y = np.log(house_sales.price)
        y_binary = (y > y.mean()).astype(int)
        y_binary
        X_all.head(3), y_binary.head(3)

        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X_all, y_binary, test_size=0.3, random_state=0)
        print(X_train.shape, X_test.shape)
        print(y_train)

        testresult = NN3_Classifier(X_train, y_train, X_test, y_test, 
                                 l1_act='relu', l2_act='relu', l3_act='softmax',
                                 layer1size=128, layer2size=64, layer3size=2,
                                 num_of_epochs=50)
        """

        # TensorFlow and tf.keras
        import tensorflow as tf
        from tensorflow import keras

        # Helper libraries
        import numpy as np
        import matplotlib.pyplot as plt

        if verbose:
            print("Tensorflow Version:")
            print(tf.__version__)

        # Normalize
        # Helper Function
        def helpNormalize(X):
            return (X - X.mean()) / np.std(X)

        X_train = X_train.apply(helpNormalize, axis=1)
        X_test = X_test.apply(helpNormalize, axis=1)

        # Model
        model = tf.keras.Sequential([
            keras.layers.Dense(units=layer1size, input_shape=[X_train.shape[1]]),
            keras.layers.Dense(units=layer2size, activation=l2_act),
            keras.layers.Dense(units=layer3size, activation=l3_act)
        ])
        if verbose:
            print("Summary of Network Architecture:")
            model.summary()

        # Compile
        model.compile(optimizer=optimizer,
                      loss=loss,
                      metrics=metrics)

        # Model Fitting
        model.fit(X_train, y_train, epochs=num_of_epochs)

        # Prediction
        predictions = model.predict(X_test)

        # Performance
        from sklearn.metrics import confusion_matrix
        import numpy as np
        import pandas as pd
        y_test_hat = np.argmax(predictions, axis=1)
        confusion = confusion_matrix(y_test, y_test_hat)
        confusion = pd.DataFrame(confusion)
        test_acc = sum(np.diag(confusion)) / sum(sum(np.array(confusion)))
        
        # Print
        if verbose:
            print("Confusion Matrix:")
            print(confusion)
            print("Test Accuracy:", round(test_acc, 4))
            
        # ROCAUC
        if layer3size == 2:
            from sklearn.metrics import roc_curve, auc, roc_auc_score
            fpr, tpr, thresholds = roc_curve(y_test, y_test_hat)
            areaUnderROC = auc(fpr, tpr)
            resultsROC = {
                'false positive rate': fpr,
                'true positive rate': tpr,
                'thresholds': thresholds,
                'auc': round(areaUnderROC, 3)
            }
            if verbose:
                print(f'Test AUC: {areaUnderROC}')
            if plotROC:
                plt.figure()
                plt.plot(fpr, tpr, color='r', lw=2, label='ROC curve')
                plt.plot([0, 1], [0, 1], color='k', lw=2, linestyle='--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('Receiver operating characteristic: \
                          Area under the curve = {0:0.2f}'.format(areaUnderROC))
                plt.legend(loc="lower right")
                plt.show()
        else: 
            resultsROC = "Response not in two classes."
        
        # Output
        return {
            'Data': [X_train, y_train, X_test, y_test],
            'Shape': [X_train.shape, len(y_train), X_test.shape, len(y_test)],
            'Model Fitting': model,
            'Performance': {
                'response': {'response': y_test, 'estimated response': y_test_hat},
                'test_acc': test_acc, 
                'confusion': confusion
            },
            'Results of ROC': resultsROC
        }
    # End of function
    
    # Define function
    def plotOneImage(
            initialPosX = 1,
            initialPosY = 0,
            boxWidth    = 1,
            boxHeight   = 0,
            linewidth   = 2,
            edgecolor   = 'r',
            IMAGE       = 0):
        
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        from PIL import Image
        import numpy as np

        im = np.array(IMAGE, dtype=np.uint8)

        # Create figure and axes
        fig,ax = plt.subplots(1)

        # Display the image
        ax.imshow(im)

        # Create a Rectangle patch
        rect = patches.Rectangle(
            (initialPosX, initialPosY), boxWidth, boxHeight,
            linewidth=linewidth, edgecolor=edgecolor, facecolor='none')

        # Add the patch to the Axes
        ax.add_patch(rect)

        plt.show()
    # End of function
      
    # Define function
    def ConvOperationC1(
        X_train, y_train, X_test, y_test, 
        inputSHAPEwidth=10, inputSHAPElenth=3,
        filter1 = [[1,0], [0,1]], 
        verbose=True, printManual=True):
        
        if printManual:
            print("----------------------------------------------------------------------")
            print("Manual")
            print(
                """
                This script input X_train, y_train, X_test, y_test with selected input width and height 
                as well as a filter. Then the script executes convolutional operation to compute new 
                features from combination of original variables and the filter.

                Note: the filter plays crucial role which is why this function the filter is user-friendly
                      and can be updated as the user see fits.
                
                # Run
                newDataGenerated = YinsDL.ConvOperationC1(
                        X_train, y_train, X_test, y_test, 
                        inputSHAPEwidth=10, inputSHAPElenth=3,
                        filter1 = [[1,0], [0,1]], 
                        verbose=True, printManual=True)
                """ )
            print("----------------------------------------------------------------------")
        
        # TensorFlow and tf.keras
        import tensorflow as tf
        from tensorflow import keras

        # Helper libraries
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt

        if verbose:
            print("Tensorflow Version:")
            print(tf.__version__)

        # Normalize
        # Helper Function
        def helpNormalize(X):
            return (X - X.mean()) / np.std(X)

        X_train = X_train.apply(helpNormalize, axis=1)
        X_test = X_test.apply(helpNormalize, axis=1)
        
        # Convolutional Operation
        X_train = np.reshape(np.array(X_train), (X_train.shape[0], inputSHAPEwidth, inputSHAPElenth))
        X_test = np.reshape(np.array(X_test), (X_test.shape[0], inputSHAPEwidth, inputSHAPElenth))
        if verbose:
            print('Shapes of X in training set', X_train.shape, 'Shapes of X in test set:', X_test.shape)

        # Filter
        filter1 = pd.DataFrame(filter1)
        
        # Convolutional Operation (called Yins to make it different from default function)
        def YinsConvOp(incidence=0, X=X_train, unitFilter=filter1):
            filterHeight = unitFilter.shape[0]
            filterWidth = unitFilter.shape[1]
            unitSample = []
            for i in range(pd.DataFrame(X[incidence]).shape[0] - (filterHeight - 1)):
                for j in range(pd.DataFrame(X[incidence]).shape[1] - (filterWidth - 1)):
                    unitSample.append(
                        np.multiply(
                            pd.DataFrame(X[incidence]).iloc[i:(i + filterWidth), j:(j + filterHeight)],
                            unitFilter).sum(axis=1).sum())
            return unitSample

        # Apply Operation
        X_train_new = pd.DataFrame([YinsConvOp(incidence=0, X=X_train, unitFilter=filter1)])
        for i in range(1, X_train.shape[0]):
            X_train_new = pd.concat([
                X_train_new,
                pd.DataFrame([YinsConvOp(incidence=i, X=X_train, unitFilter=filter1)]) ])
            
        # For Prediction
        X_test_new = pd.DataFrame([YinsConvOp(incidence=0, X=X_test, unitFilter=filter1)])
        for i in range(1, X_test.shape[0]):
            X_test_new = pd.concat([
                X_test_new,
                pd.DataFrame([YinsConvOp(incidence=i, X=X_test, unitFilter=filter1)]) ])
            
        # Output
        return {
            'Data': [X_train, y_train, X_test, y_test, X_train_new, X_test_new],
            'Shape': [X_train.shape, len(y_train), X_test.shape, len(y_test)]
        }
    # End function

    # Define function
    def C1NN3_Classifier(
        X_train, y_train, X_test, y_test, 
        inputSHAPEwidth=10, inputSHAPElenth=3,
        filter1 = [[1,0], [0,1]],
        l1_act='relu', l2_act='relu', l3_act='softmax',
        layer1size=128, layer2size=64, layer3size=2,
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
        num_of_epochs=10,
        plotROC=True,
        verbose=True):
        
        """
        MANUAL:
        
        # One can use the following example.
        house_sales = pd.read_csv('../data/kc_house_data.csv')
        house_sales.head(3)
        house_sales = house_sales.drop(['id', 'zipcode', 'lat', 'long', 'date'], axis=1)
        house_sales.info()

        X_all = house_sales.drop('price', axis=1)
        y = np.log(house_sales.price)
        y_binary = (y > y.mean()).astype(int)
        y_binary
        X_all.head(3), y_binary.head(3)

        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X_all, y_binary, test_size=0.3, random_state=0)
        print(X_train.shape, X_test.shape)
        print(y_train)

        testresult = NN3_Classifier(X_train, y_train, X_test, y_test, 
                                 l1_act='relu', l2_act='relu', l3_act='softmax',
                                 layer1size=128, layer2size=64, layer3size=2,
                                 num_of_epochs=50)
        """

        # TensorFlow and tf.keras
        import tensorflow as tf
        from tensorflow import keras

        # Helper libraries
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt

        if verbose:
            print("Tensorflow Version:")
            print(tf.__version__)

        # Normalize
        # Helper Function
        def helpNormalize(X):
            return (X - X.mean()) / np.std(X)

        X_train = X_train.apply(helpNormalize, axis=1)
        X_test = X_test.apply(helpNormalize, axis=1)
        
        # Convolutional Operation
        X_train = np.reshape(np.array(X_train), (X_train.shape[0], inputSHAPEwidth, inputSHAPElenth))
        X_test = np.reshape(np.array(X_test), (X_test.shape[0], inputSHAPEwidth, inputSHAPElenth))
        if verbose:
            print('Shapes of X in training set', X_train.shape, 'Shapes of X in test set:', X_test.shape)

        # Filter
        filter1 = pd.DataFrame(filter1)
        
        # Convolutional Operation (called Yins to make it different from default function)
        def YinsConvOp(incidence=0, X=X_train, unitFilter=filter1):
            filterHeight = unitFilter.shape[0]
            filterWidth = unitFilter.shape[1]
            unitSample = []
            for i in range(pd.DataFrame(X[incidence]).shape[0] - (filterHeight - 1)):
                for j in range(pd.DataFrame(X[incidence]).shape[1] - (filterWidth - 1)):
                    unitSample.append(
                        np.multiply(
                            pd.DataFrame(X[incidence]).iloc[i:(i + filterWidth), j:(j + filterHeight)],
                            unitFilter).sum(axis=1).sum())
            return unitSample

        # Apply Operation
        X_train_new = pd.DataFrame([YinsConvOp(incidence=0, X=X_train, unitFilter=filter1)])
        for i in range(1, X_train.shape[0]):
            X_train_new = pd.concat([
                X_train_new,
                pd.DataFrame([YinsConvOp(incidence=i, X=X_train, unitFilter=filter1)]) ])

        # Model
        model = tf.keras.Sequential([
            keras.layers.Dense(units=layer1size, input_shape=[X_train_new.shape[1]]),
            keras.layers.Dense(units=layer2size, activation=l2_act),
            keras.layers.Dense(units=layer3size, activation=l3_act)
        ])
        if verbose:
            print("Summary of Network Architecture:")
            model.summary()

        # Compile
        model.compile(optimizer=optimizer,
                      loss=loss,
                      metrics=metrics)

        # Model Fitting
        model.fit(X_train_new, y_train, epochs=num_of_epochs)

        # Prediction
        X_test_new = pd.DataFrame([YinsConvOp(incidence=0, X=X_test, unitFilter=filter1)])
        for i in range(1, X_test.shape[0]):
            X_test_new = pd.concat([
                X_test_new,
                pd.DataFrame([YinsConvOp(incidence=i, X=X_test, unitFilter=filter1)]) ])
        predictions = model.predict(X_test_new)

        # Performance
        from sklearn.metrics import confusion_matrix
        import numpy as np
        import pandas as pd
        y_test_hat = np.argmax(predictions, axis=1)
        confusion = confusion_matrix(y_test, y_test_hat)
        confusion = pd.DataFrame(confusion)
        test_acc = sum(np.diag(confusion)) / sum(sum(np.array(confusion)))
        
        # Print
        if verbose:
            print("Confusion Matrix:")
            print(confusion)
            print("Test Accuracy:", round(test_acc, 4))
            
        # ROCAUC
        if layer3size == 2:
            from sklearn.metrics import roc_curve, auc, roc_auc_score
            fpr, tpr, thresholds = roc_curve(y_test, y_test_hat)
            areaUnderROC = auc(fpr, tpr)
            resultsROC = {
                'false positive rate': fpr,
                'true positive rate': tpr,
                'thresholds': thresholds,
                'auc': round(areaUnderROC, 3)
            }
            if verbose:
                print(f'Test AUC: {areaUnderROC}')
            if plotROC:
                plt.figure()
                plt.plot(fpr, tpr, color='r', lw=2, label='ROC curve')
                plt.plot([0, 1], [0, 1], color='k', lw=2, linestyle='--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('Receiver operating characteristic: \
                          Area under the curve = {0:0.2f}'.format(areaUnderROC))
                plt.legend(loc="lower right")
                plt.show()
        else: 
            resultsROC = "Response not in two classes."
        
        # Output
        return {
            'Data': [X_train, y_train, X_test, y_test, X_train_new, X_test_new],
            'Shape': [X_train.shape, len(y_train), X_test.shape, len(y_test)],
            'Model Fitting': model,
            'Performance': {
                'response': {'response': y_test, 'estimated response': y_test_hat},
                'test_acc': test_acc, 
                'confusion': confusion
            },
            'Results of ROC': resultsROC
        }
    # End of function     

   # Define function
    def C2NN3_Classifier(
        X_train, y_train, X_test, y_test, 
        inputSHAPEwidth1=10, inputSHAPElenth1=3,
        inputSHAPEwidth2=8, inputSHAPElenth2=9,
        filter1 = [[1,0], [0,1]],
        filter2 = [[1,0], [0,1]],
        l1_act='relu', l2_act='relu', l3_act='softmax',
        layer1size=128, layer2size=64, layer3size=2,
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
        num_of_epochs=10,
        plotROC=True,
        verbose=True,
        printManual=False):
        
        if printManual:
            print("--------------------------------------------------------------------")
            print("MANUAL:")
            print(
                """
                # One can use the following example.
                house_sales = pd.read_csv('../data/kc_house_data.csv')
                house_sales.head(3)
                house_sales = house_sales.drop(['id', 'zipcode', 'lat', 'long', 'date'], axis=1)
                house_sales.info()

                X_all = house_sales.drop('price', axis=1)
                y = np.log(house_sales.price)
                y_binary = (y > y.mean()).astype(int)
                y_binary
                X_all.head(3), y_binary.head(3)

                from sklearn.model_selection import train_test_split
                X_train, X_test, y_train, y_test = train_test_split(X_all, y_binary, test_size=0.3, random_state=0)
                print(X_train.shape, X_test.shape)
                print(y_train)

                testresult = C2NN3_Classifier(
                    X_train, y_train, X_test, y_test, 
                    inputSHAPEwidth1=10, inputSHAPElenth1=3,
                    inputSHAPEwidth2=8, inputSHAPElenth2=9,
                    filter1 = [[1,0], [0,1]],
                    filter2 = [[1,0], [0,1]],
                    l1_act='relu', l2_act='relu', l3_act='softmax',
                    layer1size=128, layer2size=64, layer3size=2,
                    optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'],
                    num_of_epochs=10,
                    plotROC=True,
                    verbose=True,
                    printManual=True
                """ )
            print("--------------------------------------------------------------------")

        # TensorFlow and tf.keras
        import tensorflow as tf
        from tensorflow import keras

        # Helper libraries
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import time
        if verbose:
            print("Tensorflow Version:")
            print(tf.__version__)

        # Normalize
        # Helper Function
        def helpNormalize(X):
            return (X - X.mean()) / np.std(X)

        X_train = X_train.apply(helpNormalize, axis=1)
        X_test = X_test.apply(helpNormalize, axis=1)
        
        # Convolutional Operation
        X_train = np.reshape(np.array(X_train), (X_train.shape[0], inputSHAPEwidth1, inputSHAPElenth1))
        X_test = np.reshape(np.array(X_test), (X_test.shape[0], inputSHAPEwidth1, inputSHAPElenth1))
        if verbose:
            print('Shapes of X in training set', X_train.shape, 'Shapes of X in test set:', X_test.shape)

        # Filter
        filter1 = pd.DataFrame(filter1)
        filter2 = pd.DataFrame(filter2)
        
        # Convolutional Operation (called Yins to make it different from default function)
        def YinsConvOp(incidence=0, X=X_train, unitFilter=filter1):
            filterHeight = unitFilter.shape[0]
            filterWidth = unitFilter.shape[1]
            unitSample = []
            for i in range(pd.DataFrame(X[incidence]).shape[0] - (filterHeight - 1)):
                for j in range(pd.DataFrame(X[incidence]).shape[1] - (filterWidth - 1)):
                    unitSample.append(
                        np.multiply(
                            pd.DataFrame(X[incidence]).iloc[i:(i + filterWidth), j:(j + filterHeight)],
                            unitFilter).sum(axis=1).sum())
            return unitSample

        # Apply Operation
        # Engineer the 1st convolutional layer
        start = time.time()
        X_train_new = pd.DataFrame([YinsConvOp(incidence=0, X=X_train, unitFilter=filter1)])
        for i in range(1, X_train.shape[0]):
            X_train_new = pd.concat([
                X_train_new,
                pd.DataFrame([YinsConvOp(incidence=i, X=X_train, unitFilter=filter1)]) ])
        end = time.time()
        # Time Check
        if verbose == True: 
            print('The 1st convolutional layer is done.')
            print('Time Consumption (in sec):', round(end - start, 2))
            print('Time Consumption (in min):', round((end - start)/60, 2))
            print('Time Consumption (in hr):', round((end - start)/60/60, 2))

        # Reshape
        start = time.time()
        X_train_new_copy = np.reshape(np.array(X_train_new), (X_train_new.shape[0], inputSHAPEwidth2, inputSHAPElenth2))
        if verbose:
            print("Shape of X in training set:", X_train_new_copy.shape)

        # Engineer the 2nd convolutional layer
        X_train_new = pd.DataFrame([YinsConvOp(incidence=0, X=X_train_new_copy, unitFilter=filter2)])
        for i in range(1, X_train_new_copy.shape[0]):
            X_train_new = pd.concat([
                X_train_new,
                pd.DataFrame([YinsConvOp(incidence=i, X=X_train_new_copy, unitFilter=filter2)]) ])
        end = time.time()
        # Time Check
        if verbose == True: 
            print("The 2nd convoluational layer is done. Shape of X in training set:", X_train_new_copy.shape)
            print('Time Consumption (in sec):', round(end - start, 2))
            print('Time Consumption (in min):', round((end - start)/60, 2))
            print('Time Consumption (in hr):', round((end - start)/60/60, 2))

        # Model
        start = time.time()
        model = tf.keras.Sequential([
            keras.layers.Dense(units=layer1size, input_shape=[X_train_new.shape[1]]),
            keras.layers.Dense(units=layer2size, activation=l2_act),
            keras.layers.Dense(units=layer3size, activation=l3_act)
        ])
        if verbose:
            print("Summary of Network Architecture:")
            model.summary()

        # Compile
        model.compile(optimizer=optimizer,
                      loss=loss,
                      metrics=metrics)

        # Model Fitting
        model.fit(X_train_new, y_train, epochs=num_of_epochs)
        end = time.time()
        # Time Check
        if verbose == True: 
            print('Training Completed.')
            print('Time Consumption (in sec):', round(end - start, 2))
            print('Time Consumption (in min):', round((end - start)/60, 2))
            print('Time Consumption (in hr):', round((end - start)/60/60, 2))

        # Prediction
        # Engineer the 1st convolutional layer
        X_test_new = pd.DataFrame([YinsConvOp(incidence=0, X=X_test, unitFilter=filter1)])
        for i in range(1, X_test.shape[0]):
            X_test_new = pd.concat([
                X_test_new,
                pd.DataFrame([YinsConvOp(incidence=i, X=X_test, unitFilter=filter1)]) ])
        # Reshape
        X_test_new_copy = np.reshape(np.array(X_test_new), (X_test_new.shape[0], inputSHAPEwidth2, inputSHAPElenth2))
        # Engineer the 2nd convolutional layer
        X_test_new = pd.DataFrame([YinsConvOp(incidence=0, X=X_test_new_copy, unitFilter=filter2)])
        for i in range(1, X_test_new_copy.shape[0]):
            X_test_new = pd.concat([
                X_test_new,
                pd.DataFrame([YinsConvOp(incidence=i, X=X_test_new_copy, unitFilter=filter2)]) ])
        # Predict
        predictions = model.predict(X_test_new)

        # Performance
        from sklearn.metrics import confusion_matrix
        import numpy as np
        import pandas as pd
        y_test_hat = np.argmax(predictions, axis=1)
        confusion = confusion_matrix(y_test, y_test_hat)
        confusion = pd.DataFrame(confusion)
        test_acc = sum(np.diag(confusion)) / sum(sum(np.array(confusion)))
        
        # Print
        if verbose:
            print("Confusion Matrix:")
            print(confusion)
            print("Test Accuracy:", round(test_acc, 4))
            
        # ROCAUC
        if layer3size == 2:
            from sklearn.metrics import roc_curve, auc, roc_auc_score
            fpr, tpr, thresholds = roc_curve(y_test, y_test_hat)
            areaUnderROC = auc(fpr, tpr)
            resultsROC = {
                'false positive rate': fpr,
                'true positive rate': tpr,
                'thresholds': thresholds,
                'auc': round(areaUnderROC, 3)
            }
            if verbose:
                print(f'Test AUC: {areaUnderROC}')
            if plotROC:
                plt.figure()
                plt.plot(fpr, tpr, color='r', lw=2, label='ROC curve')
                plt.plot([0, 1], [0, 1], color='k', lw=2, linestyle='--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('Receiver operating characteristic: \
                          Area under the curve = {0:0.2f}'.format(areaUnderROC))
                plt.legend(loc="lower right")
                plt.show()
        else: 
            resultsROC = "Response not in two classes."
        
        # Output
        return {
            'Data': [X_train, y_train, X_test, y_test, X_train_new, X_test_new],
            'Shape': [X_train.shape, len(y_train), X_test.shape, len(y_test)],
            'Model Fitting': model,
            'Performance': {
                'response': {'response': y_test, 'estimated response': y_test_hat},
                'test_acc': test_acc, 
                'confusion': confusion
            },
            'Results of ROC': resultsROC
        }
    # End of function
    
    # cnn with block structure:
    def cnn_blocked_design(
        input_shape=(64, 64, 3),
        conv_blocks=[[32, 64], [32, 64, 128], [32, 32, 64, 128]],
        kernel_size=[[(3,3), (3,3)], [(3,3), (3,3), (3,3)], [(3,3), (3,3), (3,3)]],
        hidden_layers=[1024, 512],
        output_dim=2,
        name="MODEL_JohnSmith"
    ):

        """
        input_shape: a tuple such as (64, 64, 3) | this is the input dimension for the image data, assume image data
        conv_blocks: a nested list such as [[32, 64], [32, 64, 128], [32, 32, 64, 128]] | each sublist is a block of convolutional layers
        kernel_size: a tuple of length 2 such as (2,2) | this is the kernel size
        hidden_layers: a list of integers such as [1024, 512] | this is the hidden dense layers
        output_dim: an integer such as 2 | this is the number of unit in the final output layers (must match number of classes in the given dataset)
        name: a string such as "MODEL_JohnSmith" | this is the name of the model
        """

        # args
        # input_shape=(64, 64, 3)
        # conv_blocks=[[32, 64], [32, 64, 128], [32, 32, 64, 128]]
        # kernel_size=[[(3,3), (3,3)], [(3,3), (3,3), (3,3)], [(3,3), (3,3), (3,3), (3,3)]]
        # hidden_layers=[1024, 512]
        # output_dim=2
        # name="MODEL_JohnSmith"

        # build a CNN (Convolutional Neural Network) model
        model = tf.keras.models.Sequential(name=name)
        ## Your Changes Start Here ##
        # starter
        first_conv_layers = conv_blocks[0]
        model.add(tf.keras.layers.Conv2D(filters=first_conv_layers[0], kernel_size=kernel_size[0][0], activation='relu', input_shape=input_shape, name="Conv_1"))
        i = 2
        m = 1
        k = 1
        for l_ in first_conv_layers[1::]:
            model.add(tf.keras.layers.Conv2D(filters=l_, kernel_size=kernel_size[0][k], activation='relu', name="Conv_"+str(i)))
            model.add(tf.keras.layers.BatchNormalization())
            i += 1
            k += 1
        model.add(tf.keras.layers.MaxPooling2D(name='Pool_'+str(m)))
        m += 1

        # conv blocks
        which_kernel = 1
        for conv_layers in conv_blocks[1::]:
            k = 0
            for l_ in conv_layers:
                model.add(tf.keras.layers.Conv2D(filters=l_, kernel_size=kernel_size[which_kernel][k], activation='relu', name="Conv_"+str(i)))
                model.add(tf.keras.layers.BatchNormalization())
                i += 1
                k += 1
            which_kernel += 1
            model.add(tf.keras.layers.MaxPooling2D(name='Pool_'+str(m))) 
            m += 1
        # You can have more CONVOLUTIONAL layers! # <===== TRY TO TUNE THIS!!!
        # Each convolutional layer can have arbitrary different number of units! # <===== TRY TO TUNE THIS!!!
        # ... you can have however many you want
        ## Your Changes Ends Here ##
        # up to here, we finish coding the convolutional layers, we have not done neural network layers

        # build neural network layers
        model.add(tf.keras.layers.Flatten(name='Flatten')) # neural network requires the input layer to be a vector instead of 2D array
        ## Your Changes Start Here ##
        d = 1
        for l in hidden_layers:
            model.add(tf.keras.layers.Dense(l, activation='relu', use_bias=True, name='Dense_'+str(d))) # input units (usually starts with 128) and activation (it's a choice, usually relu)
            d += 1
        # You can have more DENSE layers! # <===== TRY TO TUNE THIS!!!
        # Each dense layer can have arbitrary different number of units! # <===== TRY TO TUNE THIS!!!
        # ... you can have however many you want
        ## Your Changes Ends Here ##
        model.add(tf.keras.layers.Dense(output_dim, activation='softmax')) # output layer or end layer | you have to match the number of classes

        # output
        return model
    
    # transfer learning: from_vgg16
    def cnn_from_vgg16(input_shape, n_classes, hidden=[2048,1024], optimizer='rmsprop', fine_tune=0):
        """
        Compiles a model integrated with VGG16 pretrained layers

        input_shape: tuple  - the shape of input images (width, height, channels)
        n_classes:   int    - number of classes for the output layer
        optimizer:   string - instantiated optimizer to use for training. Defaults to 'RMSProp'
        hidden:      list of integers - a list of integers to indicate the number of units for each dense layer added in the middle
        fine_tune:   int    - The number of pre-trained layers to unfreeze.
                              If set to 0, all pretrained layers will freeze during training
        """

        # Pretrained convolutional layers are loaded using the Imagenet weights.
        # Include_top is set to False, in order to exclude the model's fully-connected layers.
        conv_base = tf.keras.applications.vgg16.VGG16(include_top=False,
                         weights='imagenet', 
                         input_shape=input_shape)

        # Defines how many layers to freeze during training.
        # Layers in the convolutional base are switched from trainable to non-trainable
        # depending on the size of the fine-tuning parameter.
        if fine_tune > 0:
            for layer in conv_base.layers[:-fine_tune]:
                layer.trainable = False
        else:
            for layer in conv_base.layers:
                layer.trainable = False

        # Create a new 'top' of the model (i.e. fully-connected layers).
        # This is 'bootstrapping' a new top_model onto the pretrained layers.
        top_model = conv_base.output
        top_model = tf.keras.layers.Flatten(name="flatten")(top_model)

        # add hidden layer
        for each_unit in hidden:
          top_model = tf.keras.layers.Dense(each_unit, activation='relu')(top_model)
          top_model = tf.keras.layers.Dropout(0.2)(top_model)
        output_layer = tf.keras.layers.Dense(n_classes, activation='softmax')(top_model)

        # Group the convolutional base and new fully-connected layers into a Model object.
        model = tf.keras.Model(inputs=conv_base.input, outputs=output_layer)

        # output
        return model
    
    # transfer learning: from_vgg16_unsampled
    def cnn_from_vgg16_upsampled(upsampling_multiplier, n_classes, hidden=[2048,1024], dropOutRate=0.2):
        """
        Compiles a model integrated with VGG16 pretrained layers

        input_shape: tuple  - the shape of input images (width, height, channels)
        n_classes:   int    - number of classes for the output layer
        hidden:      list of integers - a list of integers to indicate the number of units for each dense layer added in the middle
        """

        # Pretrained convolutional layers are loaded using the Imagenet weights.
        # Include_top is set to False, in order to exclude the model's fully-connected layers.
        conv_base = tf.keras.applications.vgg16.VGG16(include_top=False,
                                                weights='imagenet',
                                                pooling='avg')

        # design model
        model= tf.keras.Sequential()

        # upsample
        for i in range(upsampling_multiplier):
          model.add(tf.keras.layers.UpSampling2D())

        # add base model
        model.add(conv_base)
        model.add(tf.keras.layers.Flatten())

        # design hidden layers
        for curr_unit in hidden:
          model.add(tf.keras.layers.Dense(curr_unit, activation=('relu'))) 
          model.add(tf.keras.layers.Dropout(dropOutRate))

        # last layer
        model.add(tf.keras.layers.Dense(n_classes, activation=('softmax')))

        # output    
        return model
    
    # define NeuralNet_Regressor function:
    def NeuralNet_Regressor(
            X_train=None,
            y_train=None, 
            X_valid=None, 
            y_valid=None, 
            X_test=None, 
            y_test=None,
            name_of_architecture="ANN",
            input_shape=8,
            use_auxinput=True,
            num_of_res_style_block=None,
            hidden=[128,64,32,10],
            output_shape=1,
            activation="relu",
            last_activation="sigmoid",
            learning_rate=0.001,
            loss="mse",
            name_of_optimizer="adam",
            epochs=10,
            plotModelSummary=True,
            axis_font_size=20,
            which_layer=None,
            X_for_internal_extraction=None,
            useGPU=False,
            use_earlystopping=False,
            do_plot=False,
            verbose=True
        ):

        # library
        import numpy as np
        import pandas as pd
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers
        import time

        # define model
        def build_model(input_shape=input_shape, use_auxinput=use_auxinput, num_of_res_style_block=num_of_res_style_block,
                        hidden=hidden, output_shape=output_shape, learning_rate=learning_rate,
                        loss="mse", activation=activation, last_activation=last_activation, name_of_optimizer=name_of_optimizer):
            # model = tf.keras.models.Sequential(name=name_of_architecture)
            inputs = keras.Input(shape=(input_shape,), name="input_layer")
            if use_auxinput:
                aux_input = inputs

            # Set up the input layer or the 1st hidden layer
            dense = layers.Dense(hidden[0], activation=activation, name=str('dense1'))
            x = dense(inputs)

            # What type of API are we using for hidden layer?
            l = 2
            for layer in hidden[1::]:
                dense = layers.Dense(layer, activation=activation, name=str('dense'+str(l)))
                x = dense(x)
                l = l + 1

            # Merge all available features into a single large vector via concatenation
            if use_auxinput:
                x = layers.concatenate([x, aux_input])

            # Optional: design residual style block if num_of_res_style_block is an integer
            # else continue
            if num_of_res_style_block == None:
                pass
            else:
                for res_i in range(num_of_res_style_block):
                    aux_input = x
                    for layer in hidden:
                        dense = layers.Dense(layer, activation=activation, name=str('dense'+str(l)))
                        x = dense(x)
                        l = l + 1
                    x = layers.concatenate([x, aux_input])                

            # Why do we set number of neurons (or units) to be 1 for this following layer?
            outputs = layers.Dense(output_shape, name=str('dense'+str(l)))(x)

            # A gentle reminder question: What is the difference between 
            # stochastic gradient descent and gradient descent?
            if name_of_optimizer == "SGD" or name_of_optimizer == "sgd":
                optimizer = tf.keras.optimizers.SGD(lr=learning_rate)
            elif name_of_optimizer == "ADAM" or name_of_optimizer == "adam":
                optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
            elif name_of_optimizer == "RMSprop" or name_of_optimizer == "rmsprop":
                optimizer = tf.keras.optimizers.RMSprop(lr=learning_rate)

            # Design a model
            model = keras.Model(inputs=inputs, outputs=outputs, name=name_of_architecture)

            # Another gentle reminder question: Why do we use mse or mean squared error？
            model.compile(loss=loss, optimizer=optimizer)

            return model

        # create a KerasRegressor based on the model defined above
        # print("Checkpoint")
        # keras_reg_init = tf.keras.wrappers.scikit_learn.KerasRegressor(build_model)
        keras_reg = build_model()

        # plot model summary
        if plotModelSummary:
            import pydot
            import graphviz
            keras.utils.plot_model(keras_reg, name_of_architecture+".png", show_shapes=True)
            print(keras_reg.summary())

        # comment:
        # The KerasRegressor object is a think wrapper around the Keras model 
        # built using build_model(). Since we did not specify any hyperparameters 
        # when creating it, it will use the default hyperparameters we defined in 
        # build_model(). This makes things convenient because we can now use 
        # this object just like a regular Scikit-learn regressor. 
        # In other words, we can use .fit(), .predict(), and all these concepts
        # consistently as we discussed before.

        # checkpoint
        start = time.time()

        # fit the model: determine whether to use GPU
        if useGPU:
            # %tensorflow_version 2.x
            # import tensorflow as tf
            device_name = tf.test.gpu_device_name()
            if device_name != '/device:GPU:0':
                raise SystemError('GPU device not found. If you are in Colab, please go to Edit => Notebook Setting to select GPU as Hardware Accelerator.')
            print('Found GPU at: {}'.format(device_name))

            # train
            if verbose:
                vb=1
            else:
                vb=0
            print("Using GPU to compute...")
            if use_earlystopping:
                with tf.device('/device:GPU:0'):
                    history = keras_reg.fit(
                        X_train, y_train, epochs=epochs,
                        validation_data=(X_valid, y_valid),
                        callbacks=[tf.keras.callbacks.EarlyStopping(patience=10)],
                        verbose=vb)
            else:
                history = keras_reg.fit(
                    X_train, y_train, epochs=epochs,
                    validation_data=(X_valid, y_valid),
                    verbose=vb)
        else:        
            # X_train, y_train, X_valid, y_valid, X_test, y_test
            # print("Checkpoint")
            if verbose:
                vb=1
            else:
                vb=0
            if use_earlystopping:
                history = keras_reg.fit(
                    X_train, y_train, epochs=epochs,
                    validation_data=(X_valid, y_valid),
                    callbacks=[tf.keras.callbacks.EarlyStopping(patience=10)],
                    verbose=vb)
            else:
                history = keras_reg.fit(
                    X_train, y_train, epochs=epochs,
                    validation_data=(X_valid, y_valid),
                    verbose=vb)
            # print("Checkpoint")

        # checkpoint
        end = time.time()
        if verbose:
            print('Training time consumption ' + str(end-start) + ' seconds.')

        # prediction on train set
        y_train_hat_ = keras_reg.predict(X_train)

        # prediction on test set
        y_test_hat_ = keras_reg.predict(X_test)

        # library 
        import numpy as np
        # from sklearn.metrics import mean_absolute_percentage_error
        from sklearn.metrics import mean_squared_error
        
        def mean_absolute_percentage_error(y_true, y_pred): 
            y_true, y_pred = np.array(y_true), np.array(y_pred)
            return np.mean(np.abs((y_true - y_pred) / np.maximum(np.ones(len(y_true)), np.abs(y_true))))*100

        # mean square error on train set
        y_train_hat_ = y_train_hat_.reshape(-1)    
        y_train_hat_=pd.Series(y_train_hat_).fillna(0).tolist()
        if output_shape == 1:
            MAPE_train = mean_absolute_percentage_error(y_true=y_train, y_pred=y_train_hat_)
            RMSE_train = mean_squared_error(y_true=y_train, y_pred=y_train_hat_) ** (.5)
        else:
            MAPE_train = "Output layer has shape more than 1."
            RMSE_train = "Output layer has shape more than 1."

        # mean square error on test set
        y_test_hat_ = y_test_hat_.reshape(-1)
        y_test_hat_=pd.Series(y_test_hat_).fillna(0).tolist()
        if output_shape == 1:
            MAPE_test = mean_absolute_percentage_error(y_true=y_test, y_pred=y_test_hat_)
            RMSE_test = mean_squared_error(y_true=y_test, y_pred=y_test_hat_) ** (.5)
        else:
            MAPE_test = "Output layer has shape more than 1."
            RMSE_test = "Output layer has shape more than 1."

        # report: MAPE_train, RMSE_train, MAPE_test, RMSE_test

        # status
        if verbose:
            print("Display dimensions of the parameters for each of the layers:")
            for l in range(len(keras_reg.get_weights())):
                print("Shape of layer " + str(l) + ": " + str(keras_reg.get_weights()[l].shape))
            print("To access weights: use the syntax 'my_model['Model'].get_weights()'. ")

        # inference
        # with a Sequential model
        if verbose:
            print('Length of internal layers: ' + str(len(keras_reg.layers)))
            print('You can input an X and extract output but within any internal layer.')
            print('Please choose a positive interger up to ' + str(len(keras_reg.layers)-1))
        if which_layer != None:
            from tensorflow.keras import backend as K
            get_internal_layer_fct = K.function([keras_reg.layers[0].input], [keras_reg.layers[which_layer].output])
            internal_layer_output = get_internal_layer_fct([np.asarray(X_for_internal_extraction)])[0]
        else:
            internal_layer_output = "Please enter which_layer and X_for_internal_extraction to obtain this."

        # visualize
        if do_plot:
            import seaborn as sns
            residuals = y_test - y_test_hat_
            residuals = pd.Series(residuals, name='Residuials')
            fitted = pd.Series(y_test_hat_, name='Fitted Value')
            ax = sns.regplot(x=residuals, y=fitted, color='g').set(title='Residuals vs. Fitted Values (Test)')
            print("Reminder: A good fit leads to Gaussian-like residuals.")

        # Output
        return {
            'Data': {
                'X_train': X_train, 
                'y_train': y_train, 
                'X_test': X_test, 
                'y_test': y_test
            },
            'Model': keras_reg,
            'History': history,
            'Extracted Internal Layer': {
                'internal_layer': internal_layer_output
            },
            'Train Result': {
                'y_train_hat_': y_train_hat_,
                'RMSE_train': RMSE_train,
                'MAPE_train': MAPE_train
            },
            'Test Result': {
                'y_test_hat_': y_test_hat_,
                'RMSE_test': RMSE_test,
                'MAPE_test': MAPE_test
            }
        }
    # End of function

    # define function
    def NeuralNet_Classifier(
            X_train=None,
            y_train=None, 
            X_valid=None, 
            y_valid=None, 
            X_test=None, 
            y_test=None,
            input_shape=[8],
            hidden=[128,64,32,10],
            output_shape=2,
            activation="relu",
            final_activation="softmax",
            learning_rate=0.001,
            loss="sparse_categorical_crossentropy",
            epochs=10,
            plotModelSummary=True,
            useGPU=False,
            verbose=True,
            plotROC=False
        ):

        # library
        import tensorflow as tf
        import time

        # define model
        def build_model(input_shape=input_shape, hidden=hidden, output_shape=output_shape, learning_rate=learning_rate,
                        loss="mse", activation=activation):
            model = tf.keras.models.Sequential()

            # What type of API are we using for input layer?
            model.add(tf.keras.layers.InputLayer(input_shape=input_shape))

            # What type of API are we using for hidden layer?
            for layer in hidden:
                model.add(tf.keras.layers.Dense(layer, activation=activation))

            # Why do we set number of neurons (or units) to be 1 for this following layer?
            model.add(tf.keras.layers.Dense(output_shape, activation=final_activation))

            # A gentle reminder question: What is the difference between 
            # stochastic gradient descent and gradient descent?
            optimizer = tf.keras.optimizers.SGD(lr=learning_rate)

            # Another gentle reminder question: Why do we use mse or mean squared error？
            model.compile(loss=loss, optimizer=optimizer)

            return model

        # plot model summary
        if plotModelSummary:
            model = build_model()
            print(model.summary())

        # create a KerasRegressor based on the model defined above
        keras_reg = tf.keras.wrappers.scikit_learn.KerasRegressor(build_model)

        # comment:
        # The KerasRegressor object is a think wrapper around the Keras model 
        # built using build_model(). Since we did not specify any hyperparameters 
        # when creating it, it will use the default hyperparameters we defined in 
        # build_model(). This makes things convenient because we can now use 
        # this object just like a regular Scikit-learn regressor. 
        # In other words, we can use .fit(), .predict(), and all these concepts
        # consistently as we discussed before.

        # checkpoint
        start = time.time()

        # fit the model: determine whether to use GPU
        if useGPU:
            # %tensorflow_version 2.x
            # import tensorflow as tf
            device_name = tf.test.gpu_device_name()
            if device_name != '/device:GPU:0':
                raise SystemError('GPU device not found. If you are in Colab, please go to Edit => Notebook Setting to select GPU as Hardware Accelerator.')
            print('Found GPU at: {}'.format(device_name))

            print("Using GPU to compute...")
            with tf.device('/device:GPU:0'):
                keras_reg.fit(
                    X_train, y_train, epochs=epochs,
                    validation_data=(X_valid, y_valid),
                    callbacks=[tf.keras.callbacks.EarlyStopping(patience=10)])
        else:        
            # X_train, y_train, X_valid, y_valid, X_test, y_test
            keras_reg.fit(X_train, y_train, epochs=epochs,
                        validation_data=(X_valid, y_valid),
                        callbacks=[tf.keras.callbacks.EarlyStopping(patience=10)])

        # checkpoint
        end = time.time()
        if verbose:
            print('Training time consumption ' + str(end-start) + ' seconds.')

        # prediction on train set
        predictions = keras_reg.predict(X_test)

        # prediction on test set
        from sklearn.metrics import confusion_matrix
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt

        y_test_hat = np.argmax(predictions, axis=1)
        confusion = confusion_matrix(y_test, y_test_hat)
        confusion = pd.DataFrame(confusion)
        test_acc = sum(np.diag(confusion)) / sum(sum(np.array(confusion)))

        # Print
        if verbose:
            print("Confusion Matrix:")
            print(confusion)
            print("Test Accuracy:", round(test_acc, 4))
        # ROCAUC
        if output_shape == 2:
            from sklearn.metrics import roc_curve, auc, roc_auc_score
            fpr, tpr, thresholds = roc_curve(y_test, y_test_hat)
            areaUnderROC = auc(fpr, tpr)
            resultsROC = {
                'false positive rate': fpr,
                'true positive rate': tpr,
                'thresholds': thresholds,
                'auc': round(areaUnderROC, 3)
            }
            if verbose:
                print(f'Test AUC: {areaUnderROC}')
            if plotROC:
                plt.figure()
                plt.plot(fpr, tpr, color='r', lw=2, label='ROC curve')
                plt.plot([0, 1], [0, 1], color='k', lw=2, linestyle='--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('Receiver operating characteristic: \
                            Area under the curve = {0:0.2f}'.format(areaUnderROC))
                plt.legend(loc="lower right")
                plt.show()
        else: 
            resultsROC = "Response not in two classes."

        # Output
        return {
            'Data': {
                'X_train': X_train, 
                'y_train': y_train, 
                'X_test': X_test, 
                'y_test': y_test
            },
            'Model': keras_reg,
            'Performance': {
                'response': {'response': y_test, 'estimated response': y_test_hat},
                'test_acc': test_acc, 
                'confusion': confusion
            },
            'Results of ROC': resultsROC
        }        

    # define
    def LSTM_Regressor(
            X_train=None,
            y_train=None, 
            X_valid=None, 
            y_valid=None, 
            X_test=None, 
            y_test=None,
            name_of_architecture='MY_MODEL',
            hidden_range = [32, 10],
            dropOutRate = 0.1,
            output_layer = 1,
            optimizer = 'adam',
            loss = 'mean_squared_error',
            epochs = 10,
            batch_size = 1,
            verbose = True
        ):

        # import 
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense
        from tensorflow.keras.layers import LSTM
        from tensorflow.keras.layers import Dropout

        # import
        import time
        import numpy as np

        # initialize
        if verbose:
            begin_time = time.time()

        # model
        regressor = Sequential(name=name_of_architecture)
        regressor.add(LSTM(hidden_range[0], return_sequences = True, input_shape = (X_train.shape[1], 1), name='input_layer'))
        regressor.add(Dropout(dropOutRate))

        l = 1
        for hidden_layer in hidden_range[1::]:
            regressor.add(LSTM(units = hidden_layer, return_sequences = True, name=str('dense'+str(l))))
            regressor.add(Dropout(dropOutRate))
            regressor.add(tf.keras.layers.MaxPooling1D(pool_size=2, strides=1, padding='valid'))
            l = l + 1

        regressor.add(tf.keras.layers.Flatten())
        regressor.add(Dense(units=output_layer, name=str('dense'+str(l+1))))

        # verbose
        if verbose:
            end_time = time.time()
            print("Time Consumption (in sec): " + str((end_time - begin_time)/60))

        # compile
        regressor.compile(optimizer = optimizer, loss = loss)

        # print
        if verbose:
            # summary
            regressor.summary()

        # fit
        begin_time = time.time()
        regressor.fit(
            X_train, y_train, epochs = epochs, batch_size = batch_size,
            validation_data = (X_valid, y_valid))
        end_time = time.time()

        # verbose
        if verbose:
            print("Time Consumption (in sec): " + str((end_time - begin_time)/60))

        # prediction
        y_train_hat_ = regressor.predict(X_train)
        y_test_hat_ = regressor.predict(X_test)

        # errors
        RMSE_train = (np.sum((y_train_hat_ - y_train) ** 2) / len(y_train)) ** 0.5
        RMSE_test = (np.sum((y_test_hat_ - y_test) ** 2) / len(y_test)) ** 0.5

        # Output
        return {
            'Data': {
                'X_train': X_train, 
                'y_train': y_train, 
                'X_test': X_test, 
                'y_test': y_test
            },
            'Model': regressor,
            'Train Result': {
                'y_train_hat_': y_train_hat_,
                'RMSE_train': RMSE_train
            },
            'Test Result': {
                'y_test_hat_': y_test_hat_,
                'RMSE_test': RMSE_test
            }
        }
    # End of function

    # time-series forecast using scalecast
    # main function
    def run_rnn_scalecast(
        target_data_=None,
        args_dict_ = {
            'max_iteration': 3,
            'lags_range': [1, 2, 3, 4, 5, 6],
            'epochs_range': [800, 1000, 1200, 2000, 3000, 4000, 5000, 6000],
            'width_range': [2, 8, 12, 18, 22, 64, 128, 256, 512],
            'dropout_range': [0, 0.05, 0.1, 0.2],
            'depth_range': [1, 2, 3, 4, 5],
            'valsplit_range': [0, 0.05, 0.1, 0.2, 0.3],
            'learningrate_range': [0.00001, 0.0001, 0.001]
        }
    ):

        """
        This function uses the `forecaster` from the `scalecast` package as a wrapper to search for the optimal parameter set for any time-series data.
        Input argument:
            target_data_=None,                                                     | a dataframe with 2 columns: date and x (desire to be forecasted)
            args_dict_ = {                                                         | a dictionary of arguments
                'max_iteration': 3,                                                | an integer
                'lags_range': [1, 2, 3, 4, 5, 6],                                  | a list of integers
                'epochs_range': [800, 1000, 1200, 2000, 3000, 4000, 5000, 6000],   | a list of integers
                'width_range': [2, 8, 12, 18, 22, 64, 128, 256, 512],              | a list of integers
                'dropout_range': [0, 0.05, 0.1, 0.2],                              | a list of fractions (usually small number)
                'depth_range': [1, 2, 3, 4, 5],                                    | a list of integers
                'valsplit_range': [0, 0.05, 0.1, 0.2, 0.3],                        | a list of fractions
                'learningrate_range': [0.00001, 0.0001, 0.001]                     | a list of fractions (usually very small fraction)
            }
        """

        # args
        # target_data_ # a pd.DataFrame with two columns (Date and Kitsin)

        # duplicate from source
        data = target_data_
        data = data.iloc[0:-1, :]

        # display dim
        L = data.shape[0]

        # define model
        f = Forecaster(y=data['kitsin'], current_dates=data['Date'])

        # need these info
        f.set_test_length(10)       # 1. 12 observations to test the results
        f.generate_future_dates(10) # 2. 12 future points to forecast
        f.set_estimator('lstm')     # 3. LSTM neural network

        # tuning
        # this is where tuning steps start

        # initialize
        ii, jj, kk, ll, r_, ss_, lr_ = 2, 5, 12, 0.1, 1, 0.1, 0.00001

        # args
        # args_dict_ = {
        #     'lags_range': [1, 2, 3, 4, 5, 6],
        #     'epochs_range': [800, 1000, 1200, 2000, 3000, 4000, 5000, 6000],
        #     'width_range': [2, 8, 12, 18, 22, 64, 128, 256, 512],
        #     'dropout_range': [0, 0.05, 0.1, 0.2],
        #     'depth_range': [1, 2, 3, 4, 5],
        #     'valsplit_range': [0, 0.05, 0.1, 0.2, 0.3],
        #     'learningrate_range': [0.00001, 0.0001, 0.001]
        # }

        # set args
        max_iter = args_dict_['max_iteration']
        ii_range = args_dict_['lags_range']
        jj_range = args_dict_['epochs_range']
        kk_range = args_dict_['width_range']
        ll_range = args_dict_['dropout_range']
        r_range = args_dict_['depth_range']
        ss_range = args_dict_['valsplit_range']
        lr_range = args_dict_['learningrate_range']

        # global iterattions
        z = 0
        while z < max_iter:

            # tuning lags: ii
            args_ = []
            curr_range_ = []
            some_result_ = []
            for ii in ii_range:
                # name
                this_nom_ = '_'.join((str(ii), str(jj), str(kk), str(ll), str(r_), str(ss_)))
                curr_range_.append(ii)

                # model
                f.manual_forecast(call_me=str(this_nom_),
                                lags=ii,
                                batch_size=int(np.round(L/10)),
                                epochs=jj,
                                validation_split=ss_,
                                shuffle=True,
                                activation='tanh',
                                optimizer='Adam',
                                learning_rate=lr_,
                                lstm_layer_sizes=(kk,)*r_,
                                dropout=(ll,)*r_,
                                callbacks=EarlyStopping(monitor='loss', patience=200),
                                verbose=0,
                                plot_loss=True)
                f.plot_test_set(order_by='LevelTestSetMAPE', models='top_1', ci=True)
                plt.show()

                # this result
                tmp = f.export(
                    'model_summaries', determine_best_by='LevelTestSetMAPE')[
                    ['ModelNickname',
                    'LevelTestSetMAPE',
                    'LevelTestSetRMSE',
                    'LevelTestSetR2',
                    'best_model']
                ]

                print(tmp)

                # collect
                args_.append(this_nom_)
                some_result_.append(np.float(tmp.loc[tmp['ModelNickname'] == this_nom_, :]['LevelTestSetMAPE']))

                # checkpoint
                print('>>> currently, we are at this tuning args combo: ', this_nom_, '<<<')

            # pick the best
            ii = curr_range_[np.argmin(some_result_)]
            print('best lags: ', ii)

            # tuning epochs: jj
            args_ = []
            curr_range_ = []
            some_result_ = []
            for jj in jj_range:
                # name
                this_nom_ = '_'.join((str(ii), str(jj), str(kk), str(ll), str(r_), str(ss_)))
                curr_range_.append(jj)

                # model
                f.manual_forecast(call_me=str(this_nom_),
                                lags=ii,
                                batch_size=int(np.round(L/10)),
                                epochs=jj,
                                validation_split=ss_,
                                shuffle=True,
                                activation='tanh',
                                optimizer='Adam',
                                learning_rate=lr_,
                                lstm_layer_sizes=(kk,)*r_,
                                dropout=(ll,)*r_,
                                callbacks=EarlyStopping(monitor='loss', patience=200),
                                verbose=0,
                                plot_loss=True)
                f.plot_test_set(order_by='LevelTestSetMAPE', models='top_1', ci=True)
                plt.show()

                # this result
                tmp = f.export(
                    'model_summaries', determine_best_by='LevelTestSetMAPE')[
                    ['ModelNickname',
                    'LevelTestSetMAPE',
                    'LevelTestSetRMSE',
                    'LevelTestSetR2',
                    'best_model']
                ]

                print(tmp)

                # collect
                args_.append(this_nom_)
                some_result_.append(np.float(tmp.loc[tmp['ModelNickname'] == this_nom_, :]['LevelTestSetMAPE']))

                # checkpoint
                print('>>> currently, we are at this tuning args combo: ', this_nom_, '<<<')

            # pick the best
            jj = curr_range_[np.argmin(some_result_)]
            print('best epochs: ', jj)

            # tuning width: kk
            args_ = []
            curr_range_ = []
            some_result_ = []
            for kk in kk_range:
                # name
                this_nom_ = '_'.join((str(ii), str(jj), str(kk), str(ll), str(r_), str(ss_)))
                curr_range_.append(kk)

                # model
                f.manual_forecast(call_me=str(this_nom_),
                                lags=ii,
                                batch_size=int(np.round(L/10)),
                                epochs=jj,
                                validation_split=ss_,
                                shuffle=True,
                                activation='tanh',
                                optimizer='Adam',
                                learning_rate=lr_,
                                lstm_layer_sizes=(kk,)*r_,
                                dropout=(ll,)*r_,
                                callbacks=EarlyStopping(monitor='loss', patience=200),
                                verbose=0,
                                plot_loss=True)
                f.plot_test_set(order_by='LevelTestSetMAPE', models='top_1', ci=True)
                plt.show()

                # this result
                tmp = f.export(
                    'model_summaries', determine_best_by='LevelTestSetMAPE')[
                    ['ModelNickname',
                    'LevelTestSetMAPE',
                    'LevelTestSetRMSE',
                    'LevelTestSetR2',
                    'best_model']
                ]

                print(tmp)

                # collect
                args_.append(this_nom_)
                some_result_.append(np.float(tmp.loc[tmp['ModelNickname'] == this_nom_, :]['LevelTestSetMAPE']))

                # checkpoint
                print('>>> currently, we are at this tuning args combo: ', this_nom_, '<<<')

            # pick the best
            kk = curr_range_[np.argmin(some_result_)]
            print('best width: ', kk)

            # tuning dropout rate: ll
            args_ = []
            curr_range_ = []
            some_result_ = []
            for ll in ll_range:
                # name
                this_nom_ = '_'.join((str(ii), str(jj), str(kk), str(ll), str(r_), str(ss_)))
                curr_range_.append(ll)

                # model
                f.manual_forecast(call_me=str(this_nom_),
                                lags=ii,
                                batch_size=int(np.round(L/10)),
                                epochs=jj,
                                validation_split=ss_,
                                shuffle=True,
                                activation='tanh',
                                optimizer='Adam',
                                learning_rate=lr_,
                                lstm_layer_sizes=(kk,)*r_,
                                dropout=(ll,)*r_,
                                callbacks=EarlyStopping(monitor='loss', patience=200),
                                verbose=0,
                                plot_loss=True)
                f.plot_test_set(order_by='LevelTestSetMAPE', models='top_1', ci=True)
                plt.show()

                # this result
                tmp = f.export(
                    'model_summaries', determine_best_by='LevelTestSetMAPE')[
                    ['ModelNickname',
                    'LevelTestSetMAPE',
                    'LevelTestSetRMSE',
                    'LevelTestSetR2',
                    'best_model']
                ]

                print(tmp)

                # collect
                args_.append(this_nom_)
                some_result_.append(np.float(tmp.loc[tmp['ModelNickname'] == this_nom_, :]['LevelTestSetMAPE']))

                # checkpoint
                print('>>> currently, we are at this tuning args combo: ', this_nom_, '<<<')

            # pick the best
            ll = curr_range_[np.argmin(some_result_)]
            print('best dropout rate: ', ll)

            # tuning depth: r_
            args_ = []
            curr_range_ = []
            some_result_ = []
            for r_ in r_range:
                # name
                this_nom_ = '_'.join((str(ii), str(jj), str(kk), str(ll), str(r_), str(ss_)))
                curr_range_.append(r_)

                # model
                f.manual_forecast(call_me=str(this_nom_),
                                lags=ii,
                                batch_size=int(np.round(L/10)),
                                epochs=jj,
                                validation_split=ss_,
                                shuffle=True,
                                activation='tanh',
                                optimizer='Adam',
                                learning_rate=lr_,
                                lstm_layer_sizes=(kk,)*r_,
                                dropout=(ll,)*r_,
                                callbacks=EarlyStopping(monitor='loss', patience=200),
                                verbose=0,
                                plot_loss=True)
                f.plot_test_set(order_by='LevelTestSetMAPE', models='top_1', ci=True)
                plt.show()

                # this result
                tmp = f.export(
                    'model_summaries', determine_best_by='LevelTestSetMAPE')[
                    ['ModelNickname',
                    'LevelTestSetMAPE',
                    'LevelTestSetRMSE',
                    'LevelTestSetR2',
                    'best_model']
                ]

                print(tmp)

                # collect
                args_.append(this_nom_)
                some_result_.append(np.float(tmp.loc[tmp['ModelNickname'] == this_nom_, :]['LevelTestSetMAPE']))

                # checkpoint
                print('>>> currently, we are at this tuning args combo: ', this_nom_, '<<<')

            # pick the best
            r_ = curr_range_[np.argmin(some_result_)]
            print('best depth: ', r_)

            # tuning validation split: ss_
            args_ = []
            curr_range_ = []
            some_result_ = []
            for ss_ in ss_range:
                # name
                this_nom_ = '_'.join((str(ii), str(jj), str(kk), str(ll), str(r_), str(ss_)))
                curr_range_.append(ss_)

                # model
                f.manual_forecast(call_me=str(this_nom_),
                                lags=ii,
                                batch_size=int(np.round(L/10)),
                                epochs=jj,
                                validation_split=ss_,
                                shuffle=True,
                                activation='tanh',
                                optimizer='Adam',
                                learning_rate=lr_,
                                lstm_layer_sizes=(kk,)*r_,
                                dropout=(ll,)*r_,
                                callbacks=EarlyStopping(monitor='loss', patience=200),
                                verbose=0,
                                plot_loss=True)
                f.plot_test_set(order_by='LevelTestSetMAPE', models='top_1', ci=True)
                plt.show()

                # this result
                tmp = f.export(
                    'model_summaries', determine_best_by='LevelTestSetMAPE')[
                    ['ModelNickname',
                    'LevelTestSetMAPE',
                    'LevelTestSetRMSE',
                    'LevelTestSetR2',
                    'best_model']
                ]

                print(tmp)

                # collect
                args_.append(this_nom_)
                some_result_.append(np.float(tmp.loc[tmp['ModelNickname'] == this_nom_, :]['LevelTestSetMAPE']))

                # checkpoint
                print('>>> currently, we are at this tuning args combo: ', this_nom_, '<<<')

            # pick the best
            ss_ = curr_range_[np.argmin(some_result_)]
            print('best validation split: ', ss_)

            # tuning learning rate: lr_
            args_ = []
            curr_range_ = []
            some_result_ = []
            for lr_ in lr_range:
                # name
                this_nom_ = '_'.join((str(ii), str(jj), str(kk), str(ll), str(r_), str(ss_)))
                curr_range_.append(lr_)

                # model
                f.manual_forecast(call_me=str(this_nom_),
                                lags=ii,
                                batch_size=int(np.round(L/10)),
                                epochs=jj,
                                validation_split=ss_,
                                shuffle=True,
                                activation='tanh',
                                optimizer='Adam',
                                learning_rate=lr_,
                                lstm_layer_sizes=(kk,)*r_,
                                dropout=(ll,)*r_,
                                callbacks=EarlyStopping(monitor='loss', patience=200),
                                verbose=0,
                                plot_loss=True)
                f.plot_test_set(order_by='LevelTestSetMAPE', models='top_1', ci=True)
                plt.show()

                # this result
                tmp = f.export(
                    'model_summaries', determine_best_by='LevelTestSetMAPE')[
                    ['ModelNickname',
                    'LevelTestSetMAPE',
                    'LevelTestSetRMSE',
                    'LevelTestSetR2',
                    'best_model']
                ]

                print(tmp)

                # collect
                args_.append(this_nom_)
                some_result_.append(np.float(tmp.loc[tmp['ModelNickname'] == this_nom_, :]['LevelTestSetMAPE']))

                # checkpoint
                print('>>> currently, we are at this tuning args combo: ', this_nom_, '<<<')

            # pick the best
            lr_ = curr_range_[np.argmin(some_result_)]
            print('best learning rate: ', lr_)

            # finalize: build the best model
            f.manual_forecast(call_me=str(this_nom_),
                            lags=ii,
                            batch_size=int(np.round(L/10)),
                            epochs=jj,
                            validation_split=ss_,
                            shuffle=True,
                            activation='tanh',
                            optimizer='Adam',
                            learning_rate=lr_,
                            lstm_layer_sizes=(kk,)*r_,
                            dropout=(ll,)*r_,
                            callbacks=EarlyStopping(monitor='loss', patience=200),
                            verbose=0,
                            plot_loss=True)
            f.plot_test_set(order_by='LevelTestSetMAPE', models='top_1', ci=True)
            plt.show()

            # this result
            tmp = f.export(
                'model_summaries', determine_best_by='LevelTestSetMAPE')[
                ['ModelNickname',
                'LevelTestSetMAPE',
                'LevelTestSetRMSE',
                'LevelTestSetR2',
                'best_model']
            ]

            print('>>>>>>>>>> final model is here: <<<<<<<<')
            print(tmp)

            # view result
            this_tuning_result_ = f.export(
                'model_summaries',
                determine_best_by='LevelTestSetMAPE')[
                ['ModelNickname',
                'LevelTestSetMAPE',
                'LevelTestSetRMSE',
                'LevelTestSetR2',
                'best_model'] ]

            # directory
            os.chdir('/root/yiqiao/kit/data/results/')
            os.listdir(), nom_of_this_siteid_this_ta_data_.split('.')[0]+'_tuning_results_.csv'

            # display name
            print(nom_of_this_siteid_this_ta_data_.split('.')[0]+'_tuning_results_.csv')

            # save
            this_tuning_result_.to_csv(nom_of_this_siteid_this_ta_data_.split('.')[0]+'_tuning_results_.csv')

            # which model
            which_model = this_tuning_result_.iloc[0,0]
            which_model

            # data for plot
            df = pd.DataFrame()
            df['Date'] = data['Date']
            df['kitsin'] = data['kitsin']

            # to_be_added
            to_be_added = pd.DataFrame([
                ['2022-09', np.nan], ['2022-10', np.nan], ['2022-11', np.nan], ['2022-12', np.nan], ['2023-01', np.nan],
                ['2023-02', np.nan], ['2023-03', np.nan], ['2023-04', np.nan], ['2023-05', np.nan], ['2023-06', np.nan]])
            to_be_added.columns = df.columns
            to_be_added

            # update df
            df = pd.concat([df, to_be_added], axis=0)

            # data for plot
            some_length_ = df.shape[0] - len(f.history[which_model]['Forecast'])
            df['forecast'] = [np.nan for i in range(some_length_)] + f.history[which_model]['Forecast']
            df['ub'] = [np.nan for i in range(some_length_)] + f.history[which_model]['UpperCI']
            df['lb'] = [np.nan for i in range(some_length_)] + f.history[which_model]['LowerCI']
            df['fitted'] = [f.history[which_model]['LevelFittedVals'][0] for i in range(int(df.shape[0] - len(f.history[which_model]['LevelFittedVals'] + [np.nan for i in range(10)])))] + f.history[which_model]['LevelFittedVals'] + [np.nan for i in range(10)]
            df

            # save
            df.to_csv(nom_of_this_siteid_this_ta_data_.split('.')[0]+'_forecasting_results_.csv')

            # authorization
            role = get_execution_role() 

            # load data
            # sample path: s3://aws-lca-sandbox07-hipaa-users/yiqiao/sagemaker-output-kitsin/
            bucket = 'aws-lca-sandbox07-hipaa-users/yiqiao'
            data_key = 'sagemaker-output-kitsin'
            data_location = 's3://{}/{}'.format(bucket, data_key)

            # reindex
            df.index = df['Date']

            # save to s3
            new_s3_path = data_location+'/'+nom_of_this_siteid_this_ta_data_.split('.')[0]+'_forecasting_results_.csv'
            print(new_s3_path)
            df.to_csv(new_s3_path)

            # mape
            final_mape_ = this_tuning_result_
            final_mape_ = final_mape_.loc[final_mape_['ModelNickname'] == which_model, :]
            final_mape_ = final_mape_['LevelTestSetMAPE']
            final_mape_ = final_mape_.to_numpy()[0]

            # plotly
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=df['Date'],
                    y=df['forecast'],
                    name='forecast',
                    marker_color=px.colors.qualitative.Dark24[3]
                ))
            fig.add_trace(
                go.Scatter(
                    x=df['Date'],
                    y=df['fitted'],
                    name='fitted',
                    marker_color=px.colors.qualitative.Dark24[5]
                ))
            fig.add_trace(
                go.Scatter(
                    x=df['Date'][-10::],
                    y=df['ub'][-10::],
                    name='ub'
                ))
            fig.add_trace(
                go.Scatter(
                    x=df['Date'][-10::],
                    y=df['lb'][-10::],
                    name='lb',
                ))
            fig.add_trace(
                go.Bar(
                    x=df['Date'],
                    y=df['kitsin'],
                    name='truth',
                    marker_color=px.colors.qualitative.Dark24[0]
                ))
            fig.update_layout(
                autosize=False,
                width=1200, height=600,
                title='Kits In (by Month) | Data: ' + nom_of_this_siteid_this_ta_data_ +' | '+'<br>CI: Upper bound='+str(np.round(df['ub'].iloc[-3],2))+', Lower bound='+str(np.round(df['lb'].iloc[-3],2))+
                ', MAPE='+str(np.round(final_mape_, 3))+'; <br>*Next month prediction='+str(int(np.round(df['forecast'].iloc[-3]))),
                xaxis=dict(title='Date (by month)'),
                yaxis=dict(title='Number of Kits (in) <br>Data: ' + nom_of_this_siteid_this_ta_data_),
                hoverlabel=dict(
                    bgcolor="white",
                    font_size=19,
                    font_family="Rockwell"
                )
            )
            fig.show()

        # checkpoint
        print(">>>>>>>>>> finished with global iteration: ", z, '/', max_iter, " <<<<<<<<<<")
        z += 1
