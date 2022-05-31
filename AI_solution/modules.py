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
