import os
import time
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
import mplfinance as mpf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

class NST:
    """
    This class object takes care of (1) training, (2) saving, and (3)
      plotting the results. List of functions:
    Function: create_autoregressive_df
    Function: interactive_ts_plot_
    """

    def create_autoregressive_df(
        ar_terms = int,
        df = pd.DataFrame()
    ) -> dict:

        """create autoregressive data frame"""

        final_df = df
        for t in range(ar_terms):
            final_df = pd.concat([final_df, df.shift(t+1)], axis=1)
        final_df = final_df.fillna(0)

        X = final_df.iloc[:, 7::]
        Y = final_df.iloc[:, 0:7]

        return {'X': X, 'Y': Y}

    def Autonomous_Neural_Sequence_Translation(
        X                 =   0,
        Y                 =   0,
        w                 =   1,
        h                 =   5,
        cutoff            =   0.8,
        val_split         =   0.2,
        hiddens           =   [128, 128, 128],
        hiddens_dense     =   [128, 128],
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

        """Recurrent Neural Network: Neural Sequence Translation"""

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
                    val_split         =   0.2, # take a fraction between 0 and 1
                    hiddens           =   [128, 128, 128],
                    hiddens_dense     =   [128, 128],
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

        # get data
        X_train = X.iloc[0:round(X.shape[0]*cutoff), :]
        X_test = X.iloc[round(X.shape[0]*cutoff):X.shape[0], :]

        y_train = Y.iloc[0:round(Y.shape[0]*cutoff), :]
        y_test = Y.iloc[round(Y.shape[0]*cutoff):Y.shape[0], :]

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
        # Adding the first LSTM layer and some Dropout regularisation
        regressor.add(LSTM(units = hiddens[0], return_sequences = True, input_shape = (w, h), name='input_layer'))
        regressor.add(Dropout(dropOutRate, name='dropout_layer_1'))

        l_, d_ = 1, 2
        for l in hiddens[1:-1]:
            # Adding a second LSTM layer and some Dropout regularisation
            regressor.add(LSTM(units = l, return_sequences = True, name='hidden_lstm_layer'+str(l_)))
            regressor.add(Dropout(dropOutRate, name='dropout_layer_'+str(d_)))
            l_ += 1
            d_ += 1

        # Adding a third LSTM layer and some Dropout regularisation
        regressor.add(LSTM(units = hiddens[-1], name='hidden_lstm_layer'+str(l_+1))
        regressor.add(Dropout(dropOutRate))

        # Design dense layers
        for d in range(len(hiddens_dense)):
            regressor.add(Dense(units = hiddens_dense[d], activation = layer_activation, name='dense_'+str(d+1)))

        # Adding the output layer
        regressor.add(Dense(units = y_train.shape[1], name='output_layer'))
        endtime = time.time()

        # Summary
        if verbose:
            print("--------------------------------------------")
            print('Let us investigate the sequential models.')
            regressor.summary()
            print("--------------------------------------------")
            print("Time Consumption (in sec):", endtime - begintime)
            print("Time Consumption (in min):", round((endtime - begintime)/60, 2))
            print("--------------------------------------------")

        ### Train RNN
        # Compiling the RNN
        start = time.time()
        regressor.compile(optimizer = optimizer, loss = loss)

        # Fitting the RNN to the Training set
        regressor.fit(
            X_train,
            y_train,
            epochs = epochs,
            validation_split = val_split,
            batch_size = batch_size,
            verbose=verbose)
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
            fig.suptitle(f'Real (Up) vs. Estimate (Down) Stock Price')
            axs[0].plot(real_stock_price, color = 'red', label = f'Real Stock Price')
            axs[1].plot(predicted_stock_price, color = 'blue', label = f'Predicted Stock Price')
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


    def interactive_ts_plot_(
        data = pd.DataFrame,
        args = dict
    ):
        """interactive time-series plot
            data: this is a pandas dataframe
            args: this is a dictionary
              example: 
                args = {
                    'secondary y': ['x1'],
                    'bars': ['x2']
                    'width': 1200,
                    'height': 800,
                    'xlabel': 'Date',
                    'ylabel': 'Number',
                    'title': 'Kitsin (daily)',
                    'font size': 30
                }
        """

        # data
        df = data
        width = args['width']
        height = args['height']

        # figure
        # fig = go.Figure()
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        for j in range(df.shape[1]):
            if df.columns[j] not in args['bars']:
                if df.columns[j] in args['secondary y']:
                    fig.add_trace(
                        go.Scatter(
                            x=df.index,
                            y=df.iloc[:, j],
                            name=df.columns[j],
                            marker_color=px.colors.qualitative.Dark24[j]
                        ), secondary_y=True)
                else:
                    fig.add_trace(
                        go.Scatter(
                            x=df.index,
                            y=df.iloc[:, j],
                            name=df.columns[j],
                            marker_color=px.colors.qualitative.Dark24[j]
                        ))
            if df.columns[j] in args['bars']:
                fig.add_trace(
                    go.Bar(
                        x=df.index,
                        y=df.iloc[:, j],
                        name=df.columns[j],
                        marker_color=px.colors.qualitative.Dark24[j]
                    ))
        fig.update_layout(
            autosize=False,
            width=width,
            height=height,
            xaxis=dict(title=args['xlabel']),
            yaxis=dict(title=args['ylabel']),
            title=args['title'],
            hoverlabel=dict(
                bgcolor="white",
                font_size=args['font size'],
                font_family="Rockwell"
            )
        )

        # show plot
        fig.show()
