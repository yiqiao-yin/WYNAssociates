class YinsFinancialTools:

    """
    Yin's Machine Learning Package for Financial Tools
    Copyright © W.Y.N. Associates, LLC, 2009 – Present
    """
    
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
        import time
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
        temp = RSI_Timer(
            start_date =   '2013-01-01',
            end_date   =   '2019-12-6',
            tickers    =   'AAPL',
            pick_RSI   = 1,
            rsi_threshold_1 = 10,
            rsi_threshold_2 = 30,
            rsi_threshold_3 = 100,
            buy_threshold = 20,
            sell_threshold = 80 )
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
        numberOfPastDays =   100,
        cutoff           =   0.8,
        numOfHiddenLayer =   3,
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
                    end_date         =   '2021-01-01',
                    tickers          =   'AAPL',
                    numberOfPastDays =   100,
                    cutoff           =   0.8,
                    numOfHiddenLayer =   3,
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

    # Define Function: Recurrent Neural Network: Neural Sequence Translation
    def Neural_Sequence_Translation(
            start_date       =   '2013-01-01',
            end_date         =   '2021-01-01',
            ticker           =   'AAPL',
            w                =   1,
            h                =   5,
            cutoff           =   0.8,
            numOfHiddenLayer =   3,
            numOfDense       =   2,
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
            useMPLFinancePlot=   True,
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
                tmp = Neural_Sequence_Translation(
                    start_date       =   '2013-01-01',
                    end_date         =   '2021-01-01',
                    ticker           =   'AAPL',
                    w                =   1,
                    h                =   5,
                    cutoff           =   0.8,
                    numOfHiddenLayer =   3,
                    numOfDense       =   2,
                    l1_units         =   50,
                    l2_units         =   50,
                    l2_units         =   50,
                    l3_units         =   50,
                    l4_units         =   30,
                    l5_units         =   10,
                    dropOutRate       =  0.2,
                    optimizer        =   'adam',
                    loss             =   'mean_squared_error',
                    useDice          =   True,
                    epochs           =   50,
                    batch_size       =   64,
                    plotGraph        =   True,
                    useMPLFinancePlot=   True,
                    verbose          =   True )

                # Cite
                # All Rights Reserved. © Yiqiao Yin
                """ )
            print("------------------------------------------------------------------------------")

            # libraries
            import pandas as pd
            import numpy as np
            import yfinance as yf

            # get data
            stockData = yf.download(ticker, start_date, end_date)
            stockData = stockData.iloc[:,:5] # omit volume

            # create data
            Y = stockData.iloc[w::, ]
            X = np.arange(0, Y.shape[0]*w*h, 1).reshape(Y.shape[0], w*h)
            for i in range(0,int(stockData.shape[0]-w)):
                X[i,] = np.array(stockData.iloc[i:(i+w),]).reshape(1, w*h)

            X_train = X[0:round(X.shape[0]*cutoff), ]
            X_test = X[round(X.shape[0]*cutoff):X.shape[0], ]

            y_train = Y.iloc[0:round(Y.shape[0]*cutoff), ]
            y_test = Y.iloc[round(Y.shape[0]*cutoff):Y.shape[0], ]

            X_train = np.array(X_train).reshape(X_train.shape[0], w, h)
            X_test = np.array(X_test).reshape(X_test.shape[0], w, h)

            if verbose:
                print(X_train.shape)
                print(X_test.shape)
                print(y_train.shape)
                print(y_test.shape)

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
                regressor.add(LSTM(units = l1_units, return_sequences = True, input_shape = (w, h)))
                regressor.add(Dropout(dropOutRate))

                # Adding a second LSTM layer and some Dropout regularisation
                regressor.add(LSTM(units = l2_units))
                regressor.add(Dropout(dropOutRate))

            elif numOfHiddenLayer == 3:
                # Adding the first LSTM layer and some Dropout regularisation
                regressor.add(LSTM(units = l1_units, return_sequences = True, input_shape = (w, h)))
                regressor.add(Dropout(dropOutRate))

                # Adding a second LSTM layer and some Dropout regularisation
                regressor.add(LSTM(units = l2_units, return_sequences = True))
                regressor.add(Dropout(dropOutRate))

                # Adding a third LSTM layer and some Dropout regularisation
                regressor.add(LSTM(units = l3_units))
                regressor.add(Dropout(dropOutRate))

            elif numOfHiddenLayer == 4:
                # Adding the first LSTM layer and some Dropout regularisation
                regressor.add(LSTM(units = l1_units, return_sequences = True, input_shape = (w, h)))
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
                regressor.add(LSTM(units = l1_units, return_sequences = True, input_shape = (w, h)))
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

            # Design dense layers
            if numOfDense == 1:
                regressor.add(Dense(units = l1_units))
            elif numOfDense == 2:
                regressor.add(Dense(units = l1_units))
                regressor.add(Dense(units = l2_units))
            elif numOfDense == 3:
                regressor.add(Dense(units = l1_units))
                regressor.add(Dense(units = l2_units))
                regressor.add(Dense(units = l3_units))
            else:
                if verbose:
                    print("Options are 1, 2, or 3. Reset to one dense layer.")
                regressor.add(Dense(units = l1_units))

            # Adding the output layer
            regressor.add(Dense(units = y_train.shape[1]))
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
            real_stock_price = y_test

            # Visualising the results
            import matplotlib.pyplot as plt
            if plotGraph:
                fig, axs = plt.subplots(2, figsize = (10,6))
                fig.suptitle(f'Real (Up) vs. Estimate (Down) {ticker} Stock Price')
                axs[0].plot(real_stock_price, color = 'red', label = f'Real {ticker} Stock Price')
                axs[1].plot(predicted_stock_price, color = 'blue', label = f'Predicted {ticker} Stock Price')
            if useMPLFinancePlot:
                import pandas as pd
                import mplfinance as mpf

                predicted_stock_price = pd.DataFrame(predicted_stock_price)
                predicted_stock_price.columns = real_stock_price.columns
                predicted_stock_price.index = real_stock_price.index

                s = mpf.make_mpf_style(base_mpf_style='charles', rc={'font.size': 6})
                fig = mpf.figure(figsize=(10, 7), style=s) # pass in the self defined style to the whole canvas
                ax = fig.add_subplot(2,1,1) # main candle stick chart subplot, you can also pass in the self defined style here only for this subplot
                av = fig.add_subplot(2,1,2, sharex=ax)  # volume chart subplot

                df1 = real_stock_price
                mpf.plot(df1, type='candle', style='yahoo', ax=ax, volume=False)

                df2 = predicted_stock_price
                mpf.plot(df2, type='candle', style='yahoo', ax=av)

        # Output
        return {
            'Information': {
                'explanatory matrix X shape': X.shape, 
                'response matrix Y shape': Y.shape
            },
            'Data': {
                'X_train': X_train, 
                'y_train': y_train,
                'X_test': X_test,
                'y_test': y_test
            },
            'Model': {
                'neural sequence translation model': regressor
            },
            'Test Response': {
                'predicted_stock_price': predicted_stock_price, 
                'real_stock_price': real_stock_price
            }
        }
    # End function

    # Define Function: Recurrent Neural Network: Neural Sequence Translation
    def Autonomous_Neural_Sequence_Translation(
            X                 =   0,
            Y                 =   0,
            w                 =   1,
            h                 =   5,
            cutoff            =   0.8,
            numOfHiddenLayer  =   3,
            numOfDense        =   2,
            l1_units          =   128,
            l2_units          =   64,
            l3_units          =   32,
            l4_units          =   16,
            l5_units          =   10,
            dropOutRate       =   0.2,
            layer_activation  =   'relu',
            final_activation  =   'softmax',
            optimizer         =   'adam',
            loss              =   'mean_squared_error',
            epochs            =   50,
            batch_size        =   64,
            plotGraph         =   False,
            useMPLFinancePlot =   True,
            verbose           =   True ):

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
                tmp = Autonomous_Neural_Sequence_Translation(
                    X                 =   X,   # explanatory data matrix
                    Y                 =   Y,   # response data matrix
                    w                 =   1,
                    h                 =   5,
                    cutoff            =   0.8, # take a fraction between 0 and 1
                    numOfHiddenLayer  =   3,   # take an integer from 1, 2, 3, 4, or 5
                    numOfDense        =   2,   # take an integer from 1, 2, or 3
                    l1_units          =   128,
                    l2_units          =   64,
                    l3_units          =   32,
                    l4_units          =   16,
                    l5_units          =   10,
                    dropOutRate       =   0.2,
                    optimizer         =   'adam',
                    loss              =   'mean_squared_error',
                    epochs            =   50,
                    batch_size        =   64,
                    plotGraph         =   False,
                    useMPLFinancePlot =   True,
                    verbose           =   True )

                # Cite
                # All Rights Reserved. © Yiqiao Yin
                """ )
            print("------------------------------------------------------------------------------")

            # libraries
            import pandas as pd
            import numpy as np
            import yfinance as yf

            # get data
            X_train = X[0:round(X.shape[0]*cutoff), ]
            X_test = X[round(X.shape[0]*cutoff):X.shape[0], ]

            y_train = Y.iloc[0:round(Y.shape[0]*cutoff), ]
            y_test = Y.iloc[round(Y.shape[0]*cutoff):Y.shape[0], ]

            X_train = np.array(X_train).reshape(X_train.shape[0], w, h)
            X_test = np.array(X_test).reshape(X_test.shape[0], w, h)

            if verbose:
                print(X_train.shape)
                print(X_test.shape)
                print(y_train.shape)
                print(y_test.shape)

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
                regressor.add(LSTM(units = l1_units, return_sequences = True, input_shape = (w, h)))
                regressor.add(Dropout(dropOutRate))

                # Adding a second LSTM layer and some Dropout regularisation
                regressor.add(LSTM(units = l2_units))
                regressor.add(Dropout(dropOutRate))

            elif numOfHiddenLayer == 3:
                # Adding the first LSTM layer and some Dropout regularisation
                regressor.add(LSTM(units = l1_units, return_sequences = True, input_shape = (w, h)))
                regressor.add(Dropout(dropOutRate))

                # Adding a second LSTM layer and some Dropout regularisation
                regressor.add(LSTM(units = l2_units, return_sequences = True))
                regressor.add(Dropout(dropOutRate))

                # Adding a third LSTM layer and some Dropout regularisation
                regressor.add(LSTM(units = l3_units))
                regressor.add(Dropout(dropOutRate))

            elif numOfHiddenLayer == 4:
                # Adding the first LSTM layer and some Dropout regularisation
                regressor.add(LSTM(units = l1_units, return_sequences = True, input_shape = (w, h)))
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
                regressor.add(LSTM(units = l1_units, return_sequences = True, input_shape = (w, h)))
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

            # Design dense layers
            if numOfDense == 1:
                regressor.add(Dense(units = l1_units, activation = layer_activation))
            elif numOfDense == 2:
                regressor.add(Dense(units = l1_units, activation = layer_activation))
                regressor.add(Dense(units = l2_units, activation = layer_activation))
            elif numOfDense == 3:
                regressor.add(Dense(units = l1_units, activation = layer_activation))
                regressor.add(Dense(units = l2_units, activation = layer_activation))
                regressor.add(Dense(units = l3_units, activation = layer_activation))
            elif numOfDense == 4:
                regressor.add(Dense(units = l1_units, activation = layer_activation))
                regressor.add(Dense(units = l2_units, activation = layer_activation))
                regressor.add(Dense(units = l3_units, activation = layer_activation))
            elif numOfDense == 5:
                regressor.add(Dense(units = l1_units, activation = layer_activation))
                regressor.add(Dense(units = l2_units, activation = layer_activation))
                regressor.add(Dense(units = l3_units, activation = layer_activation))
                regressor.add(Dense(units = l4_units, activation = layer_activation))
                regressor.add(Dense(units = l5_units, activation = layer_activation))
            else:
                if verbose:
                    print("Options are 1, 2, 3, 4, or 5. Reset to one dense layer.")
                regressor.add(Dense(units = l1_units, activation = final_activation))

            # Adding the output layer
            regressor.add(Dense(units = y_train.shape[1]))
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
            real_stock_price = y_test

            # Visualising the results
            import matplotlib.pyplot as plt
            if plotGraph:
                fig, axs = plt.subplots(2, figsize = (10,6))
                fig.suptitle(f'Real (Up) vs. Estimate (Down) {ticker} Stock Price')
                axs[0].plot(real_stock_price, color = 'red', label = f'Real {ticker} Stock Price')
                axs[1].plot(predicted_stock_price, color = 'blue', label = f'Predicted {ticker} Stock Price')
            if useMPLFinancePlot:
                import pandas as pd
                import mplfinance as mpf

                predicted_stock_price = pd.DataFrame(predicted_stock_price)
                predicted_stock_price.columns = real_stock_price.columns
                predicted_stock_price.index = real_stock_price.index

                s = mpf.make_mpf_style(base_mpf_style='charles', rc={'font.size': 6})
                fig = mpf.figure(figsize=(10, 7), style=s) # pass in the self defined style to the whole canvas
                ax = fig.add_subplot(2,1,1) # main candle stick chart subplot, you can also pass in the self defined style here only for this subplot
                av = fig.add_subplot(2,1,2, sharex=ax)  # volume chart subplot

                df1 = real_stock_price
                mpf.plot(df1, type='candle', style='yahoo', ax=ax, volume=False)

                df2 = predicted_stock_price
                mpf.plot(df2, type='candle', style='yahoo', ax=av)

        # Output
        return {
            'Information': {
                'explanatory matrix X shape': X.shape, 
                'response matrix Y shape': Y.shape
            },
            'Data': {
                'X_train': X_train, 
                'y_train': y_train,
                'X_test': X_test,
                'y_test': y_test
            },
            'Model': {
                'neural sequence translation model': regressor
            },
            'Test Response': {
                'predicted_stock_price': predicted_stock_price, 
                'real_stock_price': real_stock_price
            }
        }
    # End function

    # Define Function: Recurrent Neural Network: Neural Sequence Translation
    def Embedding_Neural_Sequence_Translation(
            X                 =   0,
            Y                 =   0,
            w                 =   1,
            h                 =   5,
            cutoff            =   0.8,
            max_len           =   1000,
            output_dim        =   5,
            numOfHiddenLayer  =   3,
            numOfDense        =   2,
            l1_units          =   128,
            l2_units          =   64,
            l3_units          =   32,
            l4_units          =   16,
            l5_units          =   10,
            dropOutRate       =   0.2,
            layer_activation  =   'relu',
            final_activation  =   'softmax',
            optimizer         =   'adam',
            loss              =   'mean_squared_error',
            epochs            =   50,
            batch_size        =   64,
            plotGraph         =   False,
            useMPLFinancePlot =   True,
            verbose           =   True ):

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
                tmp = Embedding_Neural_Sequence_Translation(
                    X                 =   X,   # explanatory data matrix
                    Y                 =   Y,   # response data matrix
                    w                 =   1,
                    h                 =   5,
                    cutoff            =   0.8, # take a fraction between 0 and 1
                    max_len           =   1000,
                    output_dim        =   5,
                    numOfHiddenLayer  =   3,   # take an integer from 1, 2, 3, 4, or 5
                    numOfDense        =   2,   # take an integer from 1, 2, or 3
                    l1_units          =   128,
                    l2_units          =   64,
                    l3_units          =   32,
                    l4_units          =   16,
                    l5_units          =   10,
                    dropOutRate       =   0.2,
                    optimizer         =   'adam',
                    loss              =   'mean_squared_error',
                    epochs            =   50,
                    batch_size        =   64,
                    plotGraph         =   False,
                    useMPLFinancePlot =   True,
                    verbose           =   True )

                # Cite
                # All Rights Reserved. © Yiqiao Yin
                """ )
            print("------------------------------------------------------------------------------")

            # libraries
            import pandas as pd
            import numpy as np
            import yfinance as yf

            # get data
            X_train = X[0:round(X.shape[0]*cutoff), ]
            X_test = X[round(X.shape[0]*cutoff):X.shape[0], ]

            y_train = Y.iloc[0:round(Y.shape[0]*cutoff), ]
            y_test = Y.iloc[round(Y.shape[0]*cutoff):Y.shape[0], ]

            X_train = np.array(X_train).reshape(X_train.shape[0], w*h) # dim would be 1, w*h if Embedding is used
            X_test = np.array(X_test).reshape(X_test.shape[0], w*h)

            if verbose:
                print(X_train.shape)
                print(X_test.shape)
                print(y_train.shape)
                print(y_test.shape)

            ### Build RNN
            # Importing the Keras libraries and packages
            from keras.models import Sequential
            from keras.layers import Dense
            from keras.layers import LSTM
            from keras.layers import Dropout
            from keras.layers import Embedding
            from keras.layers import LayerNormalization
            import time

            # Initialize RNN
            begintime = time.time()
            regressor = Sequential()

            # Embedding
            regressor.add(Embedding(input_dim=max_len, output_dim=output_dim, input_length=w*h))

            # Design hidden layers
            if numOfHiddenLayer == 2:
                # Adding the first LSTM layer and some Dropout regularisation
                regressor.add(LSTM(units = l1_units, return_sequences = True, input_shape = (w*h, output_dim)))
                regressor.add(LayerNormalization(axis=1))
                regressor.add(Dropout(dropOutRate))

                # Adding a second LSTM layer and some Dropout regularisation
                regressor.add(LSTM(units = l2_units))
                regressor.add(LayerNormalization(axis=1))
                regressor.add(Dropout(dropOutRate))

            elif numOfHiddenLayer == 3:
                # Adding the first LSTM layer and some Dropout regularisation
                regressor.add(LSTM(units = l1_units, return_sequences = True, input_shape = (w*h, output_dim)))
                regressor.add(LayerNormalization(axis=1))
                regressor.add(Dropout(dropOutRate))

                # Adding a second LSTM layer and some Dropout regularisation
                regressor.add(LSTM(units = l2_units, return_sequences = True))
                regressor.add(LayerNormalization(axis=1))
                regressor.add(Dropout(dropOutRate))

                # Adding a third LSTM layer and some Dropout regularisation
                regressor.add(LSTM(units = l3_units))
                regressor.add(LayerNormalization(axis=1))
                regressor.add(Dropout(dropOutRate))

            elif numOfHiddenLayer == 4:
                # Adding the first LSTM layer and some Dropout regularisation
                regressor.add(LSTM(units = l1_units, return_sequences = True, input_shape = (w*h, output_dim)))
                regressor.add(LayerNormalization(axis=1))
                regressor.add(Dropout(dropOutRate))

                # Adding a second LSTM layer and some Dropout regularisation
                regressor.add(LSTM(units = l2_units, return_sequences = True))
                regressor.add(LayerNormalization(axis=1))
                regressor.add(Dropout(dropOutRate))

                # Adding a third LSTM layer and some Dropout regularisation
                regressor.add(LSTM(units = l3_units, return_sequences = True))
                regressor.add(LayerNormalization(axis=1))
                regressor.add(Dropout(dropOutRate))

                # Adding a fourth LSTM layer and some Dropout regularisation
                regressor.add(LSTM(units = l4_units))
                regressor.add(LayerNormalization(axis=1))
                regressor.add(Dropout(dropOutRate))

            elif numOfHiddenLayer == 5:
                # Adding the first LSTM layer and some Dropout regularisation
                regressor.add(LSTM(units = l1_units, return_sequences = True, input_shape = (w*h, output_dim)))
                regressor.add(LayerNormalization(axis=1))
                regressor.add(Dropout(dropOutRate))

                # Adding a second LSTM layer and some Dropout regularisation
                regressor.add(LSTM(units = l2_units, return_sequences = True))
                regressor.add(LayerNormalization(axis=1))
                regressor.add(Dropout(dropOutRate))

                # Adding a third LSTM layer and some Dropout regularisation
                regressor.add(LSTM(units = l3_units, return_sequences = True))
                regressor.add(LayerNormalization(axis=1))
                regressor.add(Dropout(dropOutRate))

                # Adding a fourth LSTM layer and some Dropout regularisation
                regressor.add(LSTM(units = l4_units, return_sequences = True))
                regressor.add(LayerNormalization(axis=1))
                regressor.add(Dropout(dropOutRate))

                # Adding a fifth LSTM layer and some Dropout regularisation
                regressor.add(LSTM(units = l5_units))
                regressor.add(LayerNormalization(axis=1))
                regressor.add(Dropout(dropOutRate))

            # Design dense layers
            if numOfDense == 1:
                regressor.add(Dense(units = l1_units, activation = layer_activation))
            elif numOfDense == 2:
                regressor.add(Dense(units = l1_units, activation = layer_activation))
                regressor.add(Dense(units = l2_units, activation = layer_activation))
            elif numOfDense == 3:
                regressor.add(Dense(units = l1_units, activation = layer_activation))
                regressor.add(Dense(units = l2_units, activation = layer_activation))
                regressor.add(Dense(units = l3_units, activation = layer_activation))
            elif numOfDense == 4:
                regressor.add(Dense(units = l1_units, activation = layer_activation))
                regressor.add(Dense(units = l2_units, activation = layer_activation))
                regressor.add(Dense(units = l3_units, activation = layer_activation))
            elif numOfDense == 5:
                regressor.add(Dense(units = l1_units, activation = layer_activation))
                regressor.add(Dense(units = l2_units, activation = layer_activation))
                regressor.add(Dense(units = l3_units, activation = layer_activation))
                regressor.add(Dense(units = l4_units, activation = layer_activation))
                regressor.add(Dense(units = l5_units, activation = layer_activation))
            else:
                if verbose:
                    print("Options are 1, 2, 3, 4, or 5. Reset to one dense layer.")
                regressor.add(Dense(units = l1_units, activation = final_activation))

            # Adding the output layer
            regressor.add(Dense(units = y_train.shape[1]))
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
            real_stock_price = y_test

            # Visualising the results
            import matplotlib.pyplot as plt
            if plotGraph:
                fig, axs = plt.subplots(2, figsize = (10,6))
                fig.suptitle(f'Real (Up) vs. Estimate (Down) {ticker} Stock Price')
                axs[0].plot(real_stock_price, color = 'red', label = f'Real {ticker} Stock Price')
                axs[1].plot(predicted_stock_price, color = 'blue', label = f'Predicted {ticker} Stock Price')
            if useMPLFinancePlot:
                import pandas as pd
                import mplfinance as mpf

                predicted_stock_price = pd.DataFrame(predicted_stock_price)
                predicted_stock_price.columns = real_stock_price.columns
                predicted_stock_price.index = real_stock_price.index

                s = mpf.make_mpf_style(base_mpf_style='charles', rc={'font.size': 6})
                fig = mpf.figure(figsize=(10, 7), style=s) # pass in the self defined style to the whole canvas
                ax = fig.add_subplot(2,1,1) # main candle stick chart subplot, you can also pass in the self defined style here only for this subplot
                av = fig.add_subplot(2,1,2, sharex=ax)  # volume chart subplot

                df1 = real_stock_price
                mpf.plot(df1, type='candle', style='yahoo', ax=ax, volume=False)

                df2 = predicted_stock_price
                mpf.plot(df2, type='candle', style='yahoo', ax=av)

        # Output
        return {
            'Information': {
                'explanatory matrix X shape': X.shape, 
                'response matrix Y shape': Y.shape
            },
            'Data': {
                'X_train': X_train, 
                'y_train': y_train,
                'X_test': X_test,
                'y_test': y_test
            },
            'Model': {
                'neural sequence translation model': regressor
            },
            'Test Response': {
                'predicted_stock_price': predicted_stock_price, 
                'real_stock_price': real_stock_price
            }
        }
    # End function


class YinsML:

    """
    Yin's Machine Learning Package 
    Copyright © W.Y.N. Associates, LLC, 2009 – Present
    """

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
    def ResultAUCROC(y_test, y_test_hat):
        from sklearn.metrics import roc_curve, auc, roc_auc_score
        fpr, tpr, thresholds = roc_curve(y_test, y_test_hat)
        areaUnderROC = auc(fpr, tpr)
        resultsROC = {
            'false positive rate': fpr,
            'true positive rate': tpr,
            'thresholds': thresholds,
            'auc': round(areaUnderROC, 3) }

        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(fpr, tpr, color='r', lw=2, label='ROC curve')
        plt.plot([0, 1], [0, 1], color='k', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic: \
                  Area under the curve = {0:0.3f}'.format(areaUnderROC))
        plt.legend(loc="lower right")
        plt.show()

        
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

    # Define Function
    def RNN4_Regressor(
        start_date = '2013-01-01',
        end_date   = '2019-12-6',
        tickers    = 'AAPL', cutoff = 0.8,
        l1_units = 50, l2_units = 50, l3_units = 50, l4_units = 50,
        optimizer = 'adam', loss = 'mean_squared_error',
        epochs = 50, batch_size = 64,
        plotGraph = True,
        verbatim = True
    ):
        """
        MANUAL
        
        # Load Package
        %run "../scripts/YinsMM.py"
        
        # Run
        tmp = YinsDL.RNN4_Regressor(
                start_date = '2013-01-01',
                end_date   = '2019-12-6',
                tickers    = 'AMD', cutoff = 0.8,
                l1_units = 50, l2_units = 50, l3_units = 50, l4_units = 50,
                optimizer = 'adam', loss = 'mean_squared_error',
                epochs = 50, batch_size = 64,
                plotGraph = True,
                verbatim = True )
        """
        
        # Initiate Environment
        from scipy import stats
        import pandas as pd
        import numpy as np
        import yfinance as yf
        import matplotlib.pyplot as plt

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

        print(X_train.shape, y_train.shape)

        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        print(X_train.shape)

        X_test = []
        y_test = []

        for i in range(100, testing_set.shape[0]):
            X_test.append(np.array(testing_set)[i-100:i, 0])
            y_test.append(np.array(testing_set)[i, 0])

        X_test, y_test = np.array(X_test), np.array(y_test)

        print(X_test.shape, y_test.shape)

        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        print(X_test.shape)

        ### Build RNN

        # Importing the Keras libraries and packages
        from keras.models import Sequential
        from keras.layers import Dense
        from keras.layers import LSTM
        from keras.layers import Dropout

        # Initialize RNN
        regressor = Sequential()

        # Adding the first LSTM layer and some Dropout regularisation
        regressor.add(LSTM(units = l1_units, return_sequences = True, input_shape = (X_train.shape[1], 1)))
        regressor.add(Dropout(0.2))

        # Adding a second LSTM layer and some Dropout regularisation
        regressor.add(LSTM(units = l2_units, return_sequences = True))
        regressor.add(Dropout(0.2))

        # Adding a third LSTM layer and some Dropout regularisation
        regressor.add(LSTM(units = l3_units, return_sequences = True))
        regressor.add(Dropout(0.2))

        # Adding a fourth LSTM layer and some Dropout regularisation
        regressor.add(LSTM(units = l4_units))
        regressor.add(Dropout(0.2))

        # Adding the output layer
        regressor.add(Dense(units = 1))

        regressor.summary()

        ### Train RNN

        # Compiling the RNN
        regressor.compile(optimizer = optimizer, loss = loss)

        # Fitting the RNN to the Training set
        regressor.fit(X_train, y_train, epochs = epochs, batch_size = batch_size)

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
        if verbatim:
            print(f'Root Mean Square Error is {round(rmse,2)} for test set.')
            print(f'Interpretation: ---------------')
            print(f'On the test set, the performance of this LSTM architecture guesses ')
            print(f'{tickers[0]} stock price on average within the error of ${round(rmse,2)} dollars.')

        # Output
        return {
            'Information': [training_set.shape, testing_set.shape],
            'Data': [X_train, y_train, X_test, y_test],
            'Test Response': [predicted_stock_price, real_stock_price],
            'Test Error': rmse
        }
    # End function
    
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

            print("Using GPU to compute...")
            with tf.device('/device:GPU:0'):
                history = keras_reg.fit(
                    X_train, y_train, epochs=epochs,
                    validation_data=(X_valid, y_valid),
                    callbacks=[tf.keras.callbacks.EarlyStopping(patience=10)])
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
        from sklearn.metrics import mean_absolute_percentage_error
        from sklearn.metrics import mean_squared_error

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

        # Performance

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
