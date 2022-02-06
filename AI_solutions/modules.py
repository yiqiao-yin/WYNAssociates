# Import Modules
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
from sklearn import tree


# Define function
def DecisionTree_Classifier(X_train, X_test, y_train, y_test, maxdepth = 3):

    # Train
    DCT = tree.DecisionTreeClassifier(max_depth=maxdepth)
    DCT = DCT.fit(X_train, y_train)

    # Report In-sample Estimators
    y_train_hat_ = DCT.predict(X_train)
    y_train_hat_score = DCT.predict_proba(X_train)

    from sklearn.metrics import confusion_matrix
    confusion_train = pd.DataFrame(confusion_matrix(y_train_hat_, y_train))
    confusion_train

    train_acc = sum(np.diag(confusion_train)) / sum(sum(np.array(confusion_train)))
    train_acc

    y_test_hat_ = DCT.predict(X_test)
    y_test_hat_score = DCT.predict_proba(X_test)
    confusion_test = pd.DataFrame(confusion_matrix(y_test_hat_, y_test))
    confusion_test

    test_acc = sum(np.diag(confusion_test)) / sum(sum(np.array(confusion_test)))
    test_acc

    # Output
    return {
        'Data': {
            'X_train': X_train, 
            'y_train': y_train, 
            'X_test': X_test, 
            'y_test': y_test
        },
        'Model': DCT,
        'Train Result': {
            'y_train_hat_': y_train_hat_,
            'y_train_hat_score': y_train_hat_score,
            'confusion_train': confusion_train,
            'train_acc': train_acc
        },
        'Test Result': {
            'y_test_hat_': y_test_hat_,
            'y_test_hat_score': y_test_hat_score,
            'confusion_test': confusion_test,
            'test_acc': test_acc
        }
    }
# End of function
