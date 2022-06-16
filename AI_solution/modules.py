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
    def NN10_Classifier(
        X_train, y_train, X_test, y_test, 
        l1_act='relu', l2_act='relu', l3_act='relu', l4_act='relu', l5_act='relu',
        l6_act='relu', l7_act='relu', l8_act='relu', l9_act='relu', l10_act='softmax',
        layer1size=128, layer2size=64, layer3size=64, layer4size=64, layer5size=64, 
        layer6size=64, layer7size=64, layer8size=64, layer9size=64, layer10size=2,
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

        testresult = NN10_Classifier(
            X_train, y_train, X_test, y_test, 
            l1_act='relu', l2_act='relu', l3_act='relu', l4_act='relu', l5_act='relu',
            l6_act='relu', l7_act='relu', l8_act='relu', l9_act='relu', l10_act='softmax',
            layer1size=128, layer2size=64, layer3size=64, layer4size=64, layer5size=64, 
            layer6size=64, layer7size=64, layer8size=64, layer9size=64, layer10size=2,
            plotROC=True,
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
            keras.layers.Dense(units=layer3size, activation=l3_act),
            keras.layers.Dense(units=layer4size, activation=l4_act),
            keras.layers.Dense(units=layer5size, activation=l5_act),
            keras.layers.Dense(units=layer6size, activation=l6_act),
            keras.layers.Dense(units=layer7size, activation=l7_act),
            keras.layers.Dense(units=layer8size, activation=l8_act),
            keras.layers.Dense(units=layer9size, activation=l9_act),
            keras.layers.Dense(units=layer10size, activation=l10_act)
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
        if layer10size == 2:
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
    
    # transfer learning: from_vgg16
    def from_vgg16(input_shape, n_classes, hidden=[2048,1024], optimizer='rmsprop', fine_tune=0):
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
    def from_vgg16_upsampled(upsampling_multiplier, n_classes, hidden=[2048,1024], dropOutRate=0.2):
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
