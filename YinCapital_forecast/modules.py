# Import Libraries
import pandas as pd
import numpy as np
import yfinance as yf
import time

# Import Libraries
from scipy import stats
# import pandas as pd
# import numpy as np
# import yfinance as yf
import matplotlib.pyplot as plt
# import time

# Import Libraries
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import yfinance as yf
import math

# Define function: Yins Timer Algorithm
def Yin_Timer(
    start_date       =   '2015-01-01',
    end_date         =   '2021-01-01',
    ticker           =   'FB',
    rescale          =   True,
    figsize          =   (15,6),
    LB               =   -1,
    UB               =   +1, 
    pick_SMA         =   1,
    sma_threshold_1  =   10,
    sma_threshold_2  =   30,
    sma_threshold_3  =   100,
    plotGraph        =   True,
    verbose          =   True,
    printManual      =   True,
    gotoSEC          =   True):
    if printManual:
        print("------------------------------------------------------------------------------")
        print("MANUAL: ")
        print("Try run the following line by line in a Python Notebook.")
        print(
        """
        MANUAL: To install this python package, please use the following code.

        # In a python notebook:
        # !pip install git+https://github.com/yiqiao-yin/YinPortfolioManagement.git
        # In a command line:
        # pip install git+https://github.com/yiqiao-yin/YinPortfolioManagement.git
        
        # Run
        temp = YinsTimer(
            start_date       =   '2015-01-01',
            end_date         =   '2021-01-01',
            ticker           =   'FB',
            figsize          =   (15,6),
            LB               =   -1,
            UB               =   +1, 
            pick_SMA         =   1,
            sma_threshold_1  =   10,
            sma_threshold_2  =   30,
            sma_threshold_3  =   100,
            plotGraph        =   True,
            verbose          =   True,
            printManual      =   True,
            gotoSEC          =   True)
        """ )
        print("Manual ends here.")
        print("------------------------------------------------------------------------------")

#     # Initiate Environment
#     import pandas as pd
#     import numpy as np
#     import yfinance as yf
#     import time

    # Time
    start = time.time()

    # Get Data
    dta = yf.download(ticker, start_date, end_date)
    dta_stock = pd.DataFrame(dta)

    # Define Checking Functions:
    if LB > 0:
        print('Lower Bound (LB) for Signal is not in threshold and is set to default value: -0.01')
        LB = -0.01
    if UB < 0:
        print('Upper Bound (UB) for Signal is not in threshold and is set to default value: +0.01')
        UB = +0.01
    def chk(row):
        if row['aveDIST'] < LB or row['aveDIST'] > UB:
            val = row['aveDIST']
        else:
            val = 0
        return val

    # Generate Data
    df_stock = dta_stock
    close = df_stock['Adj Close']
    df_stock['Normalize Return'] = close / close.shift() - 1

    # Generate Signal:
    if len(dta_stock) < 200:
        data_for_plot = []
        basicStats = []
        print('Stock went IPO within a year.')
    else:
        # Create Features
        df_stock['SMA12'] = close.rolling(window=12).mean()
        df_stock['SMA20'] = close.rolling(window=20).mean()
        df_stock['SMA50'] = close.rolling(window=50).mean()
        df_stock['SMA100'] = close.rolling(window=100).mean()
        df_stock['SMA200'] = close.rolling(window=200).mean()
        df_stock['DIST12'] = close / df_stock['SMA12'] - 1
        df_stock['DIST20'] = close / df_stock['SMA20'] - 1
        df_stock['DIST50'] = close / df_stock['SMA50'] - 1
        df_stock['DIST100'] = close / df_stock['SMA100'] - 1
        df_stock['DIST200'] = close / df_stock['SMA200'] - 1
        df_stock['aveDIST'] = (df_stock['DIST12'] + df_stock['DIST20'] + 
                               df_stock['DIST50'] + df_stock['DIST100'] + df_stock['DIST200'])/5
        df_stock['Signal'] = df_stock.apply(chk, axis = 1)

    # Plot
    import matplotlib.pyplot as plt
    from ta.trend import sma_indicator
    if plotGraph:
        tickers    =   ticker
        buy_threshold = LB
        sell_threshold = UB 

        # Get Data
        stock = dta

        # Scale Data
        if rescale == False:
            smaData1 = stock['Close'] - sma_indicator(stock['Close'], sma_threshold_1, True)
            smaData2 = stock['Close'] - sma_indicator(stock['Close'], sma_threshold_2, True)
            smaData3 = stock['Close'] - sma_indicator(stock['Close'], sma_threshold_3, True)
        else:
            smaData1 = stock['Close'] - sma_indicator(stock['Close'], sma_threshold_1, True)
            smaData2 = stock['Close'] - sma_indicator(stock['Close'], sma_threshold_2, True)
            smaData3 = stock['Close'] - sma_indicator(stock['Close'], sma_threshold_3, True)
            maxDist = max(abs(stock['Close'] - sma_indicator(stock['Close'], sma_threshold_3, True)))
            smaData1 = (stock['Close'] - sma_indicator(stock['Close'], sma_threshold_1, True)) / maxDist
            smaData2 = (stock['Close'] - sma_indicator(stock['Close'], sma_threshold_2, True)) / maxDist
            smaData3 = (stock['Close'] - sma_indicator(stock['Close'], sma_threshold_3, True)) / maxDist
            
        # Conditional Buy/Sell => Signals
        conditionalBuy1 = np.where(smaData1 < buy_threshold, stock['Close'], np.nan)
        conditionalSell1 = np.where(smaData1 > sell_threshold, stock['Close'], np.nan)
        conditionalBuy2 = np.where(smaData2 < buy_threshold, stock['Close'], np.nan)
        conditionalSell2 = np.where(smaData2 > sell_threshold, stock['Close'], np.nan)
        conditionalBuy3 = np.where(smaData3 < buy_threshold, stock['Close'], np.nan)
        conditionalSell3 = np.where(smaData3 > sell_threshold, stock['Close'], np.nan)

        # SMA Construction
        stock['SMA1'] = smaData1
        stock['SMA2'] = smaData2
        stock['SMA3'] = smaData3
        stock['SMA1_Buy'] = conditionalBuy1
        stock['SMA1_Sell'] = conditionalSell1
        stock['SMA2_Buy'] = conditionalBuy2
        stock['SMA2_Sell'] = conditionalSell2
        stock['SMA3_Buy'] = conditionalBuy3
        stock['SMA3_Sell'] = conditionalSell3

        strategy = "SMA"
        title = f'Close Price Buy/Sell Signals using {strategy} {pick_SMA}'

        fig, axs = plt.subplots(2, sharex=True, figsize=figsize)

        # fig.suptitle(f'Top: {tickers} Stock Price. Bottom: {strategy}')

        if pick_SMA == 1:
            if not stock['SMA1_Buy'].isnull().all():
                axs[0].scatter(stock.index, stock['SMA1_Buy'], color='green', label='Buy Signal', marker='^', alpha=1)
            if not stock['SMA1_Sell'].isnull().all():
                axs[0].scatter(stock.index, stock['SMA1_Sell'], color='red', label='Sell Signal', marker='v', alpha=1)
            axs[0].plot(stock['Close'], label='Close Price', color='blue', alpha=0.35)
        elif pick_SMA == 2:
            if not stock['SMA2_Buy'].isnull().all():
                axs[0].scatter(stock.index, stock['SMA2_Buy'], color='green', label='Buy Signal', marker='^', alpha=1)
            if not stock['SMA2_Sell'].isnull().all():
                axs[0].scatter(stock.index, stock['SMA2_Sell'], color='red', label='Sell Signal', marker='v', alpha=1)
            axs[0].plot(stock['Close'], label='Close Price', color='blue', alpha=0.35)
        elif pick_SMA == 3:
            if not stock['SMA3_Buy'].isnull().all():
                axs[0].scatter(stock.index, stock['SMA3_Buy'], color='green', label='Buy Signal', marker='^', alpha=1)
            if not stock['SMA3_Sell'].isnull().all():
                axs[0].scatter(stock.index, stock['SMA3_Sell'], color='red', label='Sell Signal', marker='v', alpha=1)
            axs[0].plot(stock['Close'], label='Close Price', color='blue', alpha=0.35)
        else:
            if not stock['SMA1_Buy'].isnull().all():
                axs[0].scatter(stock.index, stock['SMA1_Buy'], color='green', label='Buy Signal', marker='^', alpha=1)
            if not stock['SMA1_Sell'].isnull().all():
                axs[0].scatter(stock.index, stock['SMA1_Sell'], color='red', label='Sell Signal', marker='v', alpha=1)
            axs[0].plot(stock['Close'], label='Close Price', color='blue', alpha=0.35)

        # plt.xticks(rotation=45)
        axs[0].set_title(title)
        axs[0].set_ylabel('Close Price', fontsize=10)
        axs[0].legend(loc='upper left')
        axs[0].grid()

        axs[1].plot(stock['SMA1'], label='SMA', color = 'green')
        axs[1].plot(stock['SMA2'], label='SMA', color = 'blue')
        axs[1].plot(stock['SMA3'], label='SMA', color = 'red')
        axs[1].set_ylabel('Price Minus SMA (Rescaled to Max=1)', fontsize=10)
        axs[1].set_xlabel('Date', fontsize=18)
        axs[1].grid()

        # Check Statistics:
        SIGNAL      = df_stock['Signal']
        LENGTH      = len(SIGNAL)
        count_plus  = 0
        count_minus = 0
        for i in range(LENGTH):
            if float(SIGNAL.iloc[i,]) > 0:
                count_plus += 1
        for i in range(LENGTH):
            if float(SIGNAL.iloc[i,]) < 0:
                count_minus += 1
        basicStats = {'AVE_BUY': round(np.sum(count_minus)/LENGTH, 4),
                      'AVE_SELL': round(np.sum(count_plus)/LENGTH, 4) }

        # Print
        if verbose:
            print("----------------------------------------------------------------------------------------------------")
            print(f"Entered Stock has the following information:")
            print(f'Ticker: {ticker}')
            print("---")
            print(f"Expted Return: {round(np.mean(dta_stock['Normalize Return']), 4)}")
            print(f"Expted Risk (Volatility): {round(np.std(dta_stock['Normalize Return']), 4)}")
            print(f"Reward-Risk Ratio (Daily Data): {round(np.mean(dta_stock['Normalize Return']) / np.std(dta_stock['Normalize Return']), 4)}")
            print("---")
            print("Tail of the 'Buy/Sell Signal' dataframe:")
            print(pd.DataFrame(stock).tail(3))
            print("Note: positive values indicate 'sell' and negative values indicate 'buy'.")
            print("---")
            print(f"Basic Statistics for Buy Sell Signals: {basicStats}")
            print("Note: Change LB and UB to ensure average buy sell signals fall beneath 2%.")
            print("---")
            url_front = "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK="
            url_back = "&type=10-K&dateb=&owner=exclude&count=40"
            url_all = str(url_front + ticker + url_back)
            print("For annual report on SEC site, please go to: ")
            print(url_all)
            if gotoSEC:
                import webbrowser
                webbrowser.open(url_all)
            print("----------------------------------------------------------------------------------------------------")

    # Get More Data:
    tck = yf.Ticker(ticker)
    ALL_DATA = {
        'get stock info': tck.info,
        'get historical market data': tck.history(period="max"),
        'show actions (dividends, splits)': tck.actions,
        'show dividends': tck.dividends,
        'show splits': tck.splits,
        'show financials': [tck.financials, tck.quarterly_financials],
        'show balance sheet': [tck.balance_sheet, tck.quarterly_balance_sheet],
        'show cashflow': [tck.cashflow, tck.quarterly_cashflow],
        'show earnings': [tck.earnings, tck.quarterly_earnings],
        'show sustainability': tck.sustainability,
        'show analysts recommendations': tck.recommendations,
        'show next event (earnings, etc)': tck.calendar
    }

    # Time
    end = time.time()
    if verbose == True: 
        print('Time Consumption (in sec):', round(end - start, 2))
        print('Time Consumption (in min):', round((end - start)/60, 2))
        print('Time Consumption (in hr):', round((end - start)/60/60, 2))

    # Return
    return {
        'data': dta_stock, 
        'resulting matrix': stock,
        'basic statistics': basicStats,
        'estimatedReturn': np.mean(dta_stock['Normalize Return']), 
        'estimatedRisk': np.std(dta_stock['Normalize Return']),
        'ALL_DATA': ALL_DATA 
    } 
# End function

# Define Function: RSI Timer
def RSI_Timer(
    start_date =   '2013-01-01',
    end_date   =   '2019-12-6',
    tickers    =   'AAPL',
    pick_RSI   = 1,
    rsi_threshold_1 = 10,
    rsi_threshold_2 = 30,
    rsi_threshold_3 = 100,
    buy_threshold = 20,
    sell_threshold = 80 ):
    
    print("------------------------------------------------------------------------------")
    print("MANUAL: ")
    print("Try run the following line by line in a Python Notebook.")
    print(
    """
    MANUAL: To install this python package, please use the following code.

    # In a python notebook:
    # !pip install git+https://github.com/yiqiao-yin/YinPortfolioManagement.git
    # In a command line:
    # pip install git+https://github.com/yiqiao-yin/YinPortfolioManagement.git

    # Run
    start_date = '2010-01-01'
    end_date   = '2020-01-18'
    ticker = 'FB'
    temp = YinsTimer(
            start_date, end_date, ticker, figsize=(15,6), LB=-0.01, UB=0.01, 
            plotGraph=True, verbose=True, printManual=True, gotoSEC=True)
    """ )
    print("Manual ends here.")
    print("------------------------------------------------------------------------------")

    # Get Data
    stock = yf.download(tickers, start_date, end_date)
    rsiData1 = RSIIndicator(stock['Close'], rsi_threshold_1, True)
    rsiData2 = RSIIndicator(stock['Close'], rsi_threshold_2, True)
    rsiData3 = RSIIndicator(stock['Close'], rsi_threshold_3, True)

    # Conditional Buy/Sell => Signals
    conditionalBuy1 = np.where(rsiData1.rsi() < buy_threshold, stock['Close'], np.nan)
    conditionalSell1 = np.where(rsiData1.rsi() > sell_threshold, stock['Close'], np.nan)
    conditionalBuy2 = np.where(rsiData2.rsi() < buy_threshold, stock['Close'], np.nan)
    conditionalSell2 = np.where(rsiData2.rsi() > sell_threshold, stock['Close'], np.nan)
    conditionalBuy3 = np.where(rsiData3.rsi() < buy_threshold, stock['Close'], np.nan)
    conditionalSell3 = np.where(rsiData3.rsi() > sell_threshold, stock['Close'], np.nan)

    # RSI Construction
    stock['RSI1'] = rsiData1.rsi()
    stock['RSI2'] = rsiData2.rsi()
    stock['RSI3'] = rsiData3.rsi()
    stock['RSI1_Buy'] = conditionalBuy1
    stock['RSI1_Sell'] = conditionalSell1
    stock['RSI2_Buy'] = conditionalBuy2
    stock['RSI2_Sell'] = conditionalSell2
    stock['RSI3_Buy'] = conditionalBuy3
    stock['RSI3_Sell'] = conditionalSell3

    strategy = "RSI"
    title = f'Close Price Buy/Sell Signals using {strategy}'

    fig, axs = plt.subplots(2, sharex=True, figsize=(13,9))

    # fig.suptitle(f'Top: {tickers} Stock Price. Bottom: {strategy}')

    if pick_RSI == 1:
        if not stock['RSI1_Buy'].isnull().all():
            axs[0].scatter(stock.index, stock['RSI1_Buy'], color='green', label='Buy Signal', marker='^', alpha=1)
        if not stock['RSI1_Sell'].isnull().all():
            axs[0].scatter(stock.index, stock['RSI1_Sell'], color='red', label='Sell Signal', marker='v', alpha=1)
        axs[0].plot(stock['Close'], label='Close Price', color='blue', alpha=0.35)
    elif pick_RSI == 2:
        if not stock['RSI2_Buy'].isnull().all():
            axs[0].scatter(stock.index, stock['RSI2_Buy'], color='green', label='Buy Signal', marker='^', alpha=1)
        if not stock['RSI2_Sell'].isnull().all():
            axs[0].scatter(stock.index, stock['RSI2_Sell'], color='red', label='Sell Signal', marker='v', alpha=1)
        axs[0].plot(stock['Close'], label='Close Price', color='blue', alpha=0.35)
    elif pick_RSI == 3:
        if not stock['RSI3_Buy'].isnull().all():
            axs[0].scatter(stock.index, stock['RSI3_Buy'], color='green', label='Buy Signal', marker='^', alpha=1)
        if not stock['RSI3_Sell'].isnull().all():
            axs[0].scatter(stock.index, stock['RSI3_Sell'], color='red', label='Sell Signal', marker='v', alpha=1)
        axs[0].plot(stock['Close'], label='Close Price', color='blue', alpha=0.35)
    else:
        if not stock['RSI1_Buy'].isnull().all():
            axs[0].scatter(stock.index, stock['RSI1_Buy'], color='green', label='Buy Signal', marker='^', alpha=1)
        if not stock['RSI1_Sell'].isnull().all():
            axs[0].scatter(stock.index, stock['RSI1_Sell'], color='red', label='Sell Signal', marker='v', alpha=1)
        axs[0].plot(stock['Close'], label='Close Price', color='blue', alpha=0.35)

    # plt.xticks(rotation=45)
    axs[0].set_title(title)
    axs[0].set_xlabel('Date', fontsize=18)
    axs[0].set_ylabel('Close Price', fontsize=18)
    axs[0].legend(loc='upper left')
    axs[0].grid()

    axs[1].plot(stock['RSI1'], label='RSI', color = 'green')
    axs[1].plot(stock['RSI2'], label='RSI', color = 'blue')
    axs[1].plot(stock['RSI3'], label='RSI', color = 'red')
    
    return {
        "data": stock
    }

# Define Function: Recurrent Neural Network Regressor
def RNN_Regressor(
    start_date       =   '2013-01-01',
    end_date         =   '2019-12-6',
    tickers          =   'AAPL',
    numberOfPastDays = 100,
    cutoff           =   0.8,
    numOfHiddenLayer = 3,
    l1_units         =   50,
    l2_units         =   50,
    l3_units         =   50,
    l4_units         =   30,
    l5_units         =   10,
    dropOutRate       =  0.2,
    optimizer        =   'adam',
    loss             =   'mean_squared_error',
    epochs           =   50,
    batch_size       =   64,
    plotGraph        =   True,
    verbose          =   True ):

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
            tmp = RNN_Regressor(
                start_date       =   '2013-01-01',
                end_date         =   '2019-12-6',
                tickers          =   'AAPL',
                numberOfPastDays = 100,
                cutoff           =   0.8,
                numOfHiddenLayer = 3,
                l1_units         =   50,
                l2_units         =   50,
                l3_units         =   50,
                l4_units         =   30,
                l5_units         =   10,
                dropOutRate       =  0.2,
                optimizer        =   'adam',
                loss             =   'mean_squared_error',
                epochs           =   50,
                batch_size       =   64,
                plotGraph        =   True,
                verbose          =   True )
                    
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

    for i in range(numberOfPastDays, training_set.shape[0]):
        X_train.append(np.array(training_set)[i-numberOfPastDays:i, 0])
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

    for i in range(numberOfPastDays, testing_set.shape[0]):
        X_test.append(np.array(testing_set)[i-numberOfPastDays:i, 0])
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

    # Design hidden layers
    if numOfHiddenLayer == 2:
        # Adding the first LSTM layer and some Dropout regularisation
        regressor.add(LSTM(units = l1_units, return_sequences = True, input_shape = (X_train.shape[1], 1)))
        regressor.add(Dropout(dropOutRate))

        # Adding a second LSTM layer and some Dropout regularisation
        regressor.add(LSTM(units = l2_units))
        regressor.add(Dropout(dropOutRate))

    elif numOfHiddenLayer == 3:
        # Adding the first LSTM layer and some Dropout regularisation
        regressor.add(LSTM(units = l1_units, return_sequences = True, input_shape = (X_train.shape[1], 1)))
        regressor.add(Dropout(dropOutRate))

        # Adding a second LSTM layer and some Dropout regularisation
        regressor.add(LSTM(units = l2_units, return_sequences = True))
        regressor.add(Dropout(dropOutRate))

        # Adding a third LSTM layer and some Dropout regularisation
        regressor.add(LSTM(units = l3_units))
        regressor.add(Dropout(dropOutRate))
        
    elif numOfHiddenLayer == 4:
        # Adding the first LSTM layer and some Dropout regularisation
        regressor.add(LSTM(units = l1_units, return_sequences = True, input_shape = (X_train.shape[1], 1)))
        regressor.add(Dropout(dropOutRate))

        # Adding a second LSTM layer and some Dropout regularisation
        regressor.add(LSTM(units = l2_units, return_sequences = True))
        regressor.add(Dropout(dropOutRate))

        # Adding a third LSTM layer and some Dropout regularisation
        regressor.add(LSTM(units = l3_units, return_sequences = True))
        regressor.add(Dropout(dropOutRate))
        
        # Adding a fourth LSTM layer and some Dropout regularisation
        regressor.add(LSTM(units = l4_units))
        regressor.add(Dropout(dropOutRate))
        
    elif numOfHiddenLayer == 5:
        # Adding the first LSTM layer and some Dropout regularisation
        regressor.add(LSTM(units = l1_units, return_sequences = True, input_shape = (X_train.shape[1], 1)))
        regressor.add(Dropout(dropOutRate))

        # Adding a second LSTM layer and some Dropout regularisation
        regressor.add(LSTM(units = l2_units, return_sequences = True))
        regressor.add(Dropout(dropOutRate))

        # Adding a third LSTM layer and some Dropout regularisation
        regressor.add(LSTM(units = l3_units, return_sequences = True))
        regressor.add(Dropout(dropOutRate))
        
        # Adding a fourth LSTM layer and some Dropout regularisation
        regressor.add(LSTM(units = l4_units, return_sequences = True))
        regressor.add(Dropout(dropOutRate))
        
        # Adding a fifth LSTM layer and some Dropout regularisation
        regressor.add(LSTM(units = l5_units))
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
        'Information': {
            'train set shape': training_set.shape, 
            'test set shape': testing_set.shape
        },
        'Data': {
            'X_train': X_train, 
            'y_train': y_train,
            'X_test': X_test,
            'y_test': y_test
        },
        'Test Response': {
            'predicted_stock_price': predicted_stock_price, 
            'real_stock_price': real_stock_price
        },
        'Test Error': rmse
    }
# End function
