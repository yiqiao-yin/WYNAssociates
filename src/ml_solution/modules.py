# Import Libraries
import pandas as pd
import numpy as np
import yfinance as yf
import time

# Import Libraries
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Import Libraries
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator

# library
import boto3 
import botocore 
import pandas as pd 
from sagemaker import get_execution_role 

# math
import math

class YinSystem:

    # helper: save file to s3:
    def save_to_s3(data_location = None, df = None):
        
        if data_location == None or df == None:
            print('No information added')
        else:
            # authorization
            role = get_execution_role() 

            # load data
            # data_location = None # some s3 path

            # save
            df.to_csv(data_location)

            # checkpoint
            print('Done. Just saved data here:', data_location)
    
class YinsML:

    """
    Yin's Machine Learning Package 
    Copyright © W.Y.N. Associates, LLC, 2009 – Present
    """

    # Define function
    def LinearRegression_Classifier(X_train, X_test, y_train, y_test, random_state = 0):
        
        # Import Modules
        import numpy as np
        import matplotlib.pyplot as plt
        import pandas as pd
        import random
        from sklearn.linear_model import LinearRegression
        
        # Train
        LINEAR_Reg = LinearRegression( random_state=random_state )
        LINEAR_Reg = LINEAR_Reg.fit(X_train, y_train)
        
        # Report In-sample Estimators
        y_train_hat_ = LINEAR_Reg.predict(X_train)
        y_train_hat_score = LINEAR_Reg.predict_proba(X_train)

        # Predict
        y_test_hat_ = LINEAR_Reg.predict(X_test)
        
        # Output
        return {
            'Data': {
                'X_train': X_train, 
                'y_train': y_train, 
                'X_test': X_test, 
                'y_test': y_test
            },
            'Model': LINEAR_Reg,
            'Weights': {
                'coefficients', LINEAR_Reg.coef_,
                'intercept', LINEAR_Reg.intercept_
            },
            'Train Result': {
                'y_train_hat_': y_train_hat_,
                'y_train_hat_score': y_train_hat_score
            },
            'Test Result': {
                'y_test_hat_': y_test_hat_,
                'y_test_hat_score': y_test_hat_score
            }
        }
    # End of function
    
    # Define function
    def LogisticRegression_Classifier(X_train, X_test, y_train, y_test, random_state = 0):
        
        # Import Modules
        import numpy as np
        import matplotlib.pyplot as plt
        import pandas as pd
        import random
        from sklearn.linear_model import LogisticRegression
        
        # Train
        LOGIT_Clf = LogisticRegression( random_state=random_state )
        LOGIT_Clf = LOGIT_Clf.fit(X_train, y_train)
        
        # Report In-sample Estimators
        y_train_hat_ = LOGIT_Clf.predict(X_train)
        y_train_hat_score = LOGIT_Clf.predict_proba(X_train)

        from sklearn.metrics import confusion_matrix
        confusion_train = pd.DataFrame(confusion_matrix(y_train_hat_, y_train))
        confusion_train
        
        train_acc = sum(np.diag(confusion_train)) / sum(sum(np.array(confusion_train)))
        train_acc

        y_test_hat_ = LOGIT_Clf.predict(X_test)
        y_test_hat_score = LOGIT_Clf.predict_proba(X_test)
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
            'Model': LOGIT_Clf,
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
    
    # Define function
    def KNN_Classifier(X_train, X_test, y_train, y_test, n_neighbors = 3):
        
        # Import Modules
        import numpy as np
        import matplotlib.pyplot as plt
        import pandas as pd
        import random
        from sklearn.neighbors import KNeighborsClassifier
        
        # Train
        KNN_Clf = KNeighborsClassifier( n_neighbors=n_neighbors )
        KNN_Clf = KNN_Clf.fit(X_train, y_train)
        
        # Report In-sample Estimators
        y_train_hat_ = KNN_Clf.predict(X_train)
        y_train_hat_score = KNN_Clf.predict_proba(X_train)

        from sklearn.metrics import confusion_matrix
        confusion_train = pd.DataFrame(confusion_matrix(y_train_hat_, y_train))
        confusion_train
        
        train_acc = sum(np.diag(confusion_train)) / sum(sum(np.array(confusion_train)))
        train_acc

        y_test_hat_ = KNN_Clf.predict(X_test)
        y_test_hat_score = KNN_Clf.predict_proba(X_test)
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
            'Model': KNN_Clf,
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

    # Define function
    def DecisionTree_Classifier(
        X_train, X_test, y_train, y_test, maxdepth = 3,
        verbose=True,
        figsize=(12,6),
        fontsize=12):
        
        # Import Modules
        import numpy as np
        import matplotlib.pyplot as plt
        import pandas as pd
        import random
        from sklearn import tree
        
        # Train
        DCT = tree.DecisionTreeClassifier(max_depth=maxdepth)
        DCT = DCT.fit(X_train, y_train)
        
        # Plot
        if verbose:
            plt.figure(figsize=figsize)
            tree.plot_tree(DCT, feature_names=X_train.columns, fontsize=fontsize)
        
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
    
    # define function
    def DecisionTree_Regressor(
        X_train, X_test, y_train, y_test,
        maxdepth=3, 
        verbose=True,
        figsize=(12,6),
        fontsize=12):

        # Import Modules
        import numpy as np
        import matplotlib.pyplot as plt
        import pandas as pd
        import random
        from sklearn import tree

        # Train
        DCT = tree.DecisionTreeRegressor(max_depth=maxdepth)
        DCT = DCT.fit(X_train, y_train)

        # Report In-sample Estimators
        y_train_hat_ = DCT.predict(X_train)
        RMSE_train = np.sqrt(np.mean((y_train_hat_ - y_train)**2))

        # Report Out-of-sample Estimators
        y_test_hat_ = DCT.predict(X_test)
        RMSE_test = np.sqrt(np.mean((y_test_hat_ - y_test)**2))

        # Plot
        if verbose:
            plt.figure(figsize=figsize)
            tree.plot_tree(DCT, feature_names=X_train.columns, fontsize=fontsize)

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
                'RMSE_train': RMSE_train
            },
            'Test Result': {
                'y_test_hat_': y_test_hat_,
                'RMSE_test': RMSE_test
            }
        }
    # End of function
    
    # Define function
    def RandomForest_Classifier(X_train, X_test, y_train, y_test, maxdepth = 3):
        
        # Import Modules
        import numpy as np
        import matplotlib.pyplot as plt
        import pandas as pd
        import random
        from sklearn import ensemble
        
        # Train
        RF_Clf = ensemble.RandomForestClassifier(max_depth=maxdepth)
        RF_Clf = RF_Clf.fit(X_train, y_train)
        
        # Report In-sample Estimators
        y_train_hat_ = RF_Clf.predict(X_train)
        y_train_hat_score = RF_Clf.predict_proba(X_train)

        from sklearn.metrics import confusion_matrix
        confusion_train = pd.DataFrame(confusion_matrix(y_train_hat_, y_train))
        confusion_train
        
        train_acc = sum(np.diag(confusion_train)) / sum(sum(np.array(confusion_train)))
        train_acc

        y_test_hat_ = RF_Clf.predict(X_test)
        y_test_hat_score = RF_Clf.predict_proba(X_test)
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
            'Model': RF_Clf,
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
    
    # define function
    def RandomForest_Regressor(
        X_train, X_test, 
        y_train, y_test,
        n_trees=100,
        maxdepth=3,
        figsize=(4,4),
        dpi=800,
        font_size=12,
        verbose=True):

        # Import Modules
        import numpy as np
        import matplotlib.pyplot as plt
        import pandas as pd
        import random
        from sklearn import tree
        from sklearn.ensemble import RandomForestRegressor
        import time

        # Train
        RF = RandomForestRegressor(
            n_estimators=n_trees,
            max_depth=maxdepth)
        RF = RF.fit(X_train, y_train)

        # Visualization
        if verbose:
            cn=None
            fig, axes = plt.subplots(nrows=1, ncols=1, figsize=figsize, dpi=dpi)
            tree.plot_tree(RF.estimators_[0],
                           feature_names = X_train.columns,
                           class_names=cn,
                           filled = True);
            fig.savefig('rf_individualtree.png')

        # Feature Importance
        if verbose:
            start_time = time.time()
            importances = RF.feature_importances_
            std = np.std([tree.feature_importances_ for tree in RF.estimators_], axis=0)
            elapsed_time = time.time() - start_time
            print(f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")

            forest_importances = pd.Series(importances, index=X_train.columns)
            fig, ax = plt.subplots(figsize=figsize)
            forest_importances.plot.bar(yerr=std, ax=ax)
            plt.rc('font', size=font_size)
            ax.set_title("Feature importances using MDI")
            ax.set_ylabel("Mean Decrease in Impurity (MDI)")
            # fig.tight_layout()

        # Report In-sample Estimators
        y_train_hat_ = RF.predict(X_train)
        RMSE_train = np.sqrt(np.mean((y_train_hat_ - y_train)**2))

        # Report Out-of-sample Estimators
        y_test_hat_ = RF.predict(X_test)
        RMSE_test = np.sqrt(np.mean((y_test_hat_ - y_test)**2))

        # Output
        return {
            'Data': {
                'X_train': X_train, 
                'y_train': y_train, 
                'X_test': X_test, 
                'y_test': y_test
            },
            'Model': RF,
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
    
    # Define function
    def GradientBoosting_Classifier(X_train, X_test, y_train, y_test, 
                                n_estimators = 100, 
                                learning_rate = 0.2, 
                                maxdepth = 3,
                                random_state = 0):
        
        # Import Modules
        import numpy as np
        import matplotlib.pyplot as plt
        import pandas as pd
        import random
        from sklearn import ensemble
        
        # Train
        GB_Clf = ensemble.GradientBoostingClassifier(
            n_estimators=n_estimators, learning_rate=learning_rate, max_depth=maxdepth, random_state=random_state )
        GB_Clf = GB_Clf.fit(X_train, y_train)
        
        # Report In-sample Estimators
        y_train_hat_ = GB_Clf.predict(X_train)
        y_train_hat_score = GB_Clf.predict_proba(X_train)

        from sklearn.metrics import confusion_matrix
        confusion_train = pd.DataFrame(confusion_matrix(y_train_hat_, y_train))
        confusion_train
        
        train_acc = sum(np.diag(confusion_train)) / sum(sum(np.array(confusion_train)))
        train_acc

        y_test_hat_ = GB_Clf.predict(X_test)
        y_test_hat_score = GB_Clf.predict_proba(X_test)
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
            'Model': GB_Clf,
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
    
    # define function
    def GradientBoosting_Regressor(
        X_train, X_test, y_train, y_test, 
        n_estimators = 100, 
        learning_rate = 0.2, 
        maxdepth = 3,
        random_state = 0):

        # Import Modules
        import numpy as np
        import matplotlib.pyplot as plt
        import pandas as pd
        from sklearn import ensemble
        import time

        # Train
        GB_Reg = ensemble.GradientBoostingRegressor(
            n_estimators=n_estimators, learning_rate=learning_rate, max_depth=maxdepth, random_state=random_state )
        GB_Reg = GB_Reg.fit(X_train, y_train)

        # Features
        feature_importance = pd.DataFrame([GB_Reg.feature_importances_], columns=X_train.columns)

        # Report In-sample Estimators
        y_train_hat_ = GB_Reg.predict(X_train)
        RMSE_train = np.sqrt(np.mean((y_train_hat_ - y_train)**2))

        # Report Out-of-sample Estimators
        y_test_hat_ = GB_Reg.predict(X_test)
        RMSE_test = np.sqrt(np.mean((y_test_hat_ - y_test)**2))

        # Output
        return {
            'Data': {
                'X_train': X_train, 
                'y_train': y_train, 
                'X_test': X_test, 
                'y_test': y_test
            },
            'Model': GB_Reg,
            'Feature Importance': feature_importance,
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
    
    # define SVM_Regressor function:
    def SVM_Regressor(
            X_train=None,
            y_train=None, 
            X_valid=None, 
            y_valid=None, 
            X_test=None, 
            y_test=None,
            useStandardScaler=True,
            kernel='rbf', gamma='auto', 
            C=1.0, epsilon=0.2,
            axis_font_size=20,
            verbose=True
        ):

        # library
        import pandas as pd
        import time

        # checkpoint
        start = time.time()

        # build model
        from sklearn.svm import SVR
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        import numpy as np

        # source: https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html
        if useStandardScaler:
            regr = make_pipeline(StandardScaler(), SVR(kernel=kernel, gamma=gamma, C=C, epsilon=epsilon, verbose=verbose))
        else:
            # kernel='rbf', gamma='auto', C=1.0, epsilon=0.2
            regr = SVR(kernel=kernel, gamma=gamma, C=C, epsilon=epsilon, verbose=verbose)

        # fit model
        regr.fit(X_train, y_train)

        # checkpoint
        end = time.time()
        if verbose:
            print('Training time consumption ' + str(end-start) + ' seconds.')

        # prediction on train set
        y_train_hat_ = regr.predict(X_train)

        # prediction on test set
        y_test_hat_ = regr.predict(X_test)

        # library 
        import numpy as np

        # mean square error on train set
        y_train_hat_ = y_train_hat_.reshape(-1)
        RMSE_train = (np.sum((y_train_hat_ - y_train) ** 2) / len(y_train)) ** 0.5

        # mean square error on test set
        y_test_hat_ = y_test_hat_.reshape(-1)
        RMSE_test = (np.sum((y_test_hat_ - y_test) ** 2) / len(y_test)) ** 0.5            

        # visualize
        if verbose:
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
            'Model': regr,
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
        
    # Define function
    def Adam_Regressor(Xadam, y, batch_size = 10, lr = 0.01, epochs = 200, period = 20, verbose=True):
        
        # Library
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt

        # Adam
        def adam(params, vs, sqrs, lr, batch_size, t):
            beta1 = 0.1
            beta2 = 0.111
            eps_stable = 1e-9

            for param, v, sqr in zip(params, vs, sqrs):
                g = param.grad / batch_size

                v[:] = beta1 * v + (1. - beta1) * g
                sqr[:] = beta2 * sqr + (1. - beta2) * nd.square(g)

                v_bias_corr = v / (1. - beta1 ** t)
                sqr_bias_corr = sqr / (1. - beta2 ** t)

                div = lr * v_bias_corr / (nd.sqrt(sqr_bias_corr) + eps_stable)
                param[:] = param - div

        # Library
        import mxnet as mx
        from mxnet import autograd
        from mxnet import ndarray as nd
        from mxnet import gluon
        import random

        mx.random.seed(1)
        random.seed(1)

        # Generate data.
        # Xadam = covid19_confirmed_china_rolling_data.iloc[:, [1,2,3,5]] <=== this is input
        num_inputs = pd.DataFrame(Xadam).shape[1]
        num_examples = pd.DataFrame(Xadam).shape[0]
        X = nd.array(Xadam)
        # y = nd.array(covid19_confirmed_china_rolling_data['Y']) <=== this is input
        dataset = gluon.data.ArrayDataset(X, y)

        # Construct data iterator.
        def data_iter(batch_size):
            idx = list(range(num_examples))
            random.shuffle(idx)
            for batch_i, i in enumerate(range(0, num_examples, batch_size)):
                j = nd.array(idx[i: min(i + batch_size, num_examples)])
                yield batch_i, X.take(j), y.take(j)

        # Initialize model parameters.
        def init_params():
            w = nd.random_normal(scale=1, shape=(num_inputs, 1))
            b = nd.zeros(shape=(1,))
            params = [w, b]
            vs = []
            sqrs = []
            for param in params:
                param.attach_grad()
                vs.append(param.zeros_like())
                sqrs.append(param.zeros_like())
            return params, vs, sqrs

        # Linear regression.
        def net(X, w, b):
            return nd.dot(X, w) + b

        # Loss function.
        def square_loss(yhat, y):
            return (yhat - y.reshape(yhat.shape)) ** 2 / 2

        # %matplotlib inline
        import matplotlib as mpl
        mpl.rcParams['figure.dpi']= 120
        import matplotlib.pyplot as plt
        import numpy as np

        def train(batch_size, lr, epochs, period):
            assert period >= batch_size and period % batch_size == 0
            [w, b], vs, sqrs = init_params()
            total_loss = [np.mean(square_loss(net(X, w, b), y).asnumpy())]

            t = 0
            # Epoch starts from 1.
            for epoch in range(1, epochs + 1):
                for batch_i, data, label in data_iter(batch_size):
                    with autograd.record():
                        output = net(data, w, b)
                        loss = square_loss(output, label)
                    loss.backward()
                    # Increment t before invoking adam.
                    t += 1
                    adam([w, b], vs, sqrs, lr, batch_size, t)
                    if batch_i * batch_size % period == 0:
                        total_loss.append(np.mean(square_loss(net(X, w, b), y).asnumpy()))
                print("Batch size %d, Learning rate %f, Epoch %d =========================> loss %.4e" %
                      (batch_size, lr, epoch, total_loss[-1]))
            print('w:', np.reshape(w.asnumpy(), (1, -1)),
                  'b:', b.asnumpy()[0], '\n')
            x_axis = np.linspace(0, epochs, len(total_loss), endpoint=True)
            plt.semilogy(x_axis, total_loss)
            plt.xlabel('epoch')
            plt.ylabel('loss')
            plt.show()

            return w, b

        w, b = train(batch_size = batch_size, lr = lr, epochs = epochs, period = period)

        w_adam = []
        for w_i in range(len(list(w.asnumpy()))):
            w_adam.append(list(w.asnumpy())[w_i][0])
        if verbose: 
            print('Weight:', w_adam)

        b_adam = list(b.asnumpy())[0]
        if verbose:
            print('Bias:', b_adam)

        y_hat_adam = np.dot(Xadam, w_adam) + b_adam

        return {
            'parameters': {'w': w, 'b': b},
            'y_estimate': y_hat_adam
        }
    # End of function
    
    # Define function
    def multi_rocauc_plot(
        model_names,
        y_test,
        y_pred,
        figsize=(5, 5),
        linewidth=3
    ):

      """
      model_names: a list of strings for model names such as ['model1', 'model2']
      y_test: a list of integers of 1's and 0's such as [1,1,0,0]
      y_pred: a nested list of predicted probabilities such as [[1,1,0,0],[.9,.8,.7,.6]]
      figsize: a tuple of two integers indicating figure size such as (10, 10)
      linewidth: an integer indicating line width such as 3
      """

      # set up plotting area
      plt.figure(figsize=figsize)

      # models
      i = 0
      for this_y_pred_ in y_pred:
        fpr, tpr, _ = metrics.roc_curve(y_test, this_y_pred_)
        auc = np.round(metrics.roc_auc_score(y_test, this_y_pred_), 4)
        plt.plot(fpr, tpr, label=model_names[i]+", AUC="+str(auc), linewidth=linewidth)
        i += 1

      # decorate
      plt.plot([0, 1], [0, 1], color='k', lw=2, linestyle='--')
      plt.xlabel('False Positive Rate')
      plt.ylabel('True Positive Rate')
      plt.title('Receiver operating characteristic')
      plt.legend(loc="lower right")

    # define function
    def AutoMachineLearningClassifier(
        X = None,
        y = None,
        cutoff = 0.1,
        random_state = 123,
        useMinMaxScale = False,
        selected_algorithm = ['AdaBoostClassifier', 'BaggingClassifier', 'BernoulliNB', 'CalibratedClassifierCV', 'DecisionTreeClassifier', 'DummyClassifier', 'ExtraTreeClassifier', 'ExtraTreesClassifier', 'GaussianNB', 'KNeighborsClassifier', 'LabelPropagation', 'LabelSpreading', 'LinearDiscriminantAnalysis', 'LinearSVC', 'LogisticRegression', 'NearestCentroid', 'NuSVC', 'PassiveAggressiveClassifier', 'Perceptron', 'QuadraticDiscriminantAnalysis', 'RandomForestClassifier', 'RidgeClassifier', 'RidgeClassifierCV', 'SGDClassifier', 'SVC', 'XGBClassifier', 'LGBMClassifier']
    ):
        
        # library
        import lazypredict
        from lazypredict.Supervised import LazyClassifier
        
        # min-max scale
        if useMinMaxScale:
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
            scaler.fit(X)
            X = scaler.transform(X)

        # split train test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=cutoff, random_state=random_state)
        
        # fit
        clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
        results, predictions = clf.fit(X_train, X_test, y_train, y_test)
        models_ = clf.provide_models(X_train, X_test, y_train, y_test)
        
        # prediction
        y_train_hat_mat_ = []
        y_test_hat_mat_ = []
        for some_algo in selected_algorithm:
            y_train_hat_mat_.append(models_[some_algo].predict(X_train))
            y_test_hat_mat_.append(models_[some_algo].predict(X_test))
            
        # convert
        y_train_hat_mat_ = pd.DataFrame(np.asarray(y_train_hat_mat_)).transpose().values
        y_test_hat_mat_ = pd.DataFrame(np.asarray(y_test_hat_mat_)).transpose().values
        
        # output
        return {
            'Data': {
                'X_train': X_train,
                'y_train': y_train,
                'X_test': X_test,
                'y_test': y_test
            },
            'Model': models_,
            'List of Algorithms': models_.keys(),
            'Results': results,
            'Predictions': {
                'y_train_hat_mat_': y_train_hat_mat_,
                'y_test_hat_mat_': y_test_hat_mat_
            }
        }

    # define function
    def AutoMachineLearningRegressor(
        X = None,
        y = None,
        cutoff = 0.1,
        random_state = 123,
        useMinMaxScale = False,
        selected_algorithm = ['AdaBoostRegressor', 'BaggingRegressor', 'BayesianRidge', 'DecisionTreeRegressor', 'DummyRegressor', 'ElasticNet', 'ElasticNetCV', 'ExtraTreeRegressor', 'ExtraTreesRegressor', 'GammaRegressor', 'GaussianProcessRegressor', 'GeneralizedLinearRegressor', 'GradientBoostingRegressor', 'HistGradientBoostingRegressor', 'HuberRegressor', 'KNeighborsRegressor', 'KernelRidge', 'Lars', 'LarsCV', 'Lasso', 'LassoCV', 'LassoLars', 'LassoLarsCV', 'LassoLarsIC', 'LinearRegression', 'LinearSVR', 'MLPRegressor', 'NuSVR', 'OrthogonalMatchingPursuit', 'OrthogonalMatchingPursuitCV', 'PassiveAggressiveRegressor', 'PoissonRegressor', 'RANSACRegressor', 'RandomForestRegressor', 'Ridge', 'RidgeCV', 'SGDRegressor', 'SVR', 'TransformedTargetRegressor', 'TweedieRegressor', 'XGBRegressor', 'LGBMRegressor']
    ):
        
        # library
        import lazypredict
        from lazypredict.Supervised import LazyRegressor
        
        # min-max scale
        if useMinMaxScale:
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
            scaler.fit(X)
            X = scaler.transform(X)

        # split train test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=cutoff, random_state=random_state)
        
        # fit
        reg = LazyRegressor(verbose=0, ignore_warnings=True, custom_metric=None)
        results, predictions = reg.fit(X_train, X_test, y_train, y_test)
        models_ = reg.provide_models(X_train, X_test, y_train, y_test)
        
        # prediction
        y_train_hat_mat_ = []
        y_test_hat_mat_ = []
        for some_algo in selected_algorithm:
            y_train_hat_mat_.append(models_[some_algo].predict(X_train))
            y_test_hat_mat_.append(models_[some_algo].predict(X_test))
        
        # convert
        y_train_hat_mat_ = pd.DataFrame(np.asarray(y_train_hat_mat_)).transpose().values
        y_test_hat_mat_ = pd.DataFrame(np.asarray(y_test_hat_mat_)).transpose().values

        # output
        return {
            'Data': {
                'X_train': X_train,
                'y_train': y_train,
                'X_test': X_test,
                'y_test': y_test
            },
            'Model': models_,
            'List of Algorithms': models_.keys(),
            'Results': results,
            'Predictions': {
                'y_train_hat_mat_': y_train_hat_mat_,
                'y_test_hat_mat_': y_test_hat_mat_
            }
        }
