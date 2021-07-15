# Define Function
def RNN3_Regressor(
    start_date =   '2013-01-01',
    end_date   =   '2019-12-6',
    tickers    =   'AAPL',
    cutoff     =   0.8,
    l1_units   =   50,
    l2_units   =   50,
    l3_units   =   50,
    dropOutRate =  0.2,
    optimizer  =   'adam',
    loss       =   'mean_squared_error',
    epochs     =   50,
    batch_size =   64,
    plotGraph  =   True,
    verbose    =   True ):

    if verbose:
        print("------------------------------------------------------------------------------")
        print(
            """
            MANUAL: To install this python package, please use the following code.

            # In a python notebook:
            # !pip install git+https://github.com/yiqiao-yin/YinPortfolioManagement.git
            # In a command line:
            # pip install git+https://github.com/yiqiao-yin/YinPortfolioManagement.git

            # Run
            tmp = RNN3_Regressor(
                    start_date =   '2013-01-01',
                    end_date   =   '2019-12-6',
                    tickers    =   'AAPL',
                    cutoff     =   0.8,
                    l1_units   =   50,
                    l2_units   =   50,
                    l3_units   =   50,
                    dropOutRate =  0.2,
                    optimizer  =   'adam',
                    loss       =   'mean_squared_error',
                    epochs     =   50,
                    batch_size =   64,
                    plotGraph  =   True,
                    verbose    =   True )
                    
            # Cite
            # All Rights Reserved. © Yiqiao Yin
            """ )
        print("------------------------------------------------------------------------------")

    # Initiate Environment
    from scipy import stats
    import pandas as pd
    import numpy as np
    import yfinance as yf
    import matplotlib.pyplot as plt
    import time

    # Define function
    def getDatafromYF(ticker, start_date, end_date):
        stockData = yf.download(ticker, start_date, end_date)
        return stockData
    # End function

    start_date = pd.to_datetime(start_date)
    end_date   = pd.to_datetime(end_date)
    tickers    = [tickers]

    # Start with Dictionary (this is where data is saved)
    stockData = {}
    for i in tickers:
        stockData[i] = pd.DataFrame(getDatafromYF(str(i), start_date, end_date))
        close = stockData[i]['Adj Close']
        stockData[i]['Normalize Return'] = close / close.shift() - 1

    # Take a look
    # print(stockData[tickers[0]].head(2)) # this is desired stock
    # print(stockData[tickers[1]].head(2)) # this is benchmark (in this case, it is S&P 500 SPDR Index Fund: SPY)

    # Feature Scaling
    from sklearn.preprocessing import MinMaxScaler

    stockData[tickers[0]].iloc[:, 4].head(3)

    data = stockData[tickers[0]].iloc[:, 4:5].values
    sc = MinMaxScaler(feature_range = (0, 1))
    scaled_dta = sc.fit_transform(data)
    scaled_dta = pd.DataFrame(scaled_dta)

    training_set = scaled_dta.iloc[0:round(scaled_dta.shape[0] * cutoff), :]
    testing_set = scaled_dta.iloc[round(cutoff * scaled_dta.shape[0] + 1):scaled_dta.shape[0], :]

    # print(training_set.shape, testing_set.shape)

    X_train = []
    y_train = []

    for i in range(100, training_set.shape[0]):
        X_train.append(np.array(training_set)[i-100:i, 0])
        y_train.append(np.array(training_set)[i, 0])

    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    if verbose:
        print('--------------------------------------------------------------------')
        print('Shape for data frame in training set:')
        print('Shape of X:', X_train.shape, '; Shape of Y:', len(y_train))
        print('--------------------------------------------------------------------')

    X_test = []
    y_test = []

    for i in range(100, testing_set.shape[0]):
        X_test.append(np.array(testing_set)[i-100:i, 0])
        y_test.append(np.array(testing_set)[i, 0])

    X_test, y_test = np.array(X_test), np.array(y_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    if verbose:
        print('--------------------------------------------------------------------')
        print('Shape for data frame in testing set:')
        print('Shape of X:', X_test.shape, ': Shape of Y:', len(y_test))
        print('--------------------------------------------------------------------')

    ### Build RNN
    # Importing the Keras libraries and packages
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import LSTM
    from keras.layers import Dropout
    import time

    # Initialize RNN
    begintime = time.time()
    regressor = Sequential()

    # Adding the first LSTM layer and some Dropout regularisation
    regressor.add(LSTM(units = l1_units, return_sequences = True, input_shape = (X_train.shape[1], 1)))
    regressor.add(Dropout(dropOutRate))

    # Adding a second LSTM layer and some Dropout regularisation
    regressor.add(LSTM(units = l2_units, return_sequences = True))
    regressor.add(Dropout(dropOutRate))

    # Adding a third LSTM layer and some Dropout regularisation
    regressor.add(LSTM(units = l3_units))
    regressor.add(Dropout(dropOutRate))

    # Adding the output layer
    regressor.add(Dense(units = 1))
    endtime = time.time()

    # Summary
    if verbose:
        print("--------------------------------------------")
        print('Let us investigate the sequential models.')
        regressor.summary()
        print("--------------------------------------------")
        print("Time Consumption (in sec):", endtime - begintime)
        print("Time Consumption (in min):", round((endtime - begintime)/60, 2))
        print("Time Consumption (in hr):", round((endtime - begintime)/60)/60, 2)
        print("--------------------------------------------")

    ### Train RNN
    # Compiling the RNN
    start = time.time()
    regressor.compile(optimizer = optimizer, loss = loss)

    # Fitting the RNN to the Training set
    regressor.fit(X_train, y_train, epochs = epochs, batch_size = batch_size)
    end = time.time()

    # Time Check
    if verbose == True: 
        print('Time Consumption:', end - start)

    ### Predictions
    predicted_stock_price = regressor.predict(X_test)
    predicted_stock_price = sc.inverse_transform(predicted_stock_price)

    real_stock_price = np.reshape(y_test, (y_test.shape[0], 1))
    real_stock_price = sc.inverse_transform(real_stock_price)

    ### Performance Visualization

    # Visualising the results
    import matplotlib.pyplot as plt
    if plotGraph:
        plt.plot(real_stock_price, color = 'red', label = f'Real {tickers[0]} Stock Price')
        plt.plot(predicted_stock_price, color = 'blue', label = f'Predicted {tickers[0]} Stock Price')
        plt.title(f'{tickers[0]} Stock Price Prediction')
        plt.xlabel('Time')
        plt.ylabel(f'{tickers[0]} Stock Price')
        plt.legend()
        plt.show()

    import math
    from sklearn.metrics import mean_squared_error
    rmse = np.sqrt(mean_squared_error(real_stock_price, predicted_stock_price))
    if verbose:
        print(f'---------------------------------------------------------------------------------')
        print(f'Root Mean Square Error is {round(rmse,2)} for test set.')
        print(f'------------------')
        print(f'Interpretation:')
        print(f'------------------')
        print(f'On the test set, the performance of this LSTM architecture guesses ')
        print(f'{tickers[0]} stock price on average within the error of ${round(rmse,2)} dollars.')
        print(f'---------------------------------------------------------------------------------')

    # Output
    return {
        'Information': [training_set.shape, testing_set.shape],
        'Data': [X_train, y_train, X_test, y_test],
        'Test Response': [predicted_stock_price, real_stock_price],
        'Test Error': rmse
    }
# End function
