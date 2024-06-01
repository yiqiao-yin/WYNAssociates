import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scalecast.Forecaster import Forecaster
from tensorflow.keras.callbacks import EarlyStopping


class TSForecast:
    """
    This class object takes care of (1) training, (2) saving, and (3)
      plotting the results. List of functions:
    Function: tune_scalecast_
    Function: get_best_model_
    Function: plot_trained_model
    """

    def __init__(self, target_data_, target_name, n_forecast):
        self.target_data_ = target_data_
        self.target_name = target_name
        self.n_forecast = n_forecast

    def tune_scalecast_(
        target_data_ = pd.DataFrame(),
        target_name = str,
        n_forecast = float,
        args_dict_ = dict,
        file_args_dict_ = dict,
        determine_best_by = 'LevelTestSetMAPE',
        nom_of_this_siteid_this_ta_data_ = "name_of_data_you_desired_to_use",
        PATH_TO_SAVE_DATA = '/root/yiqiao/kit/data/results/',
        plot_test=False
    ):
        """tune_scalecast_ tune the parameter one by one"""

        # duplicate from source
        data = target_data_
        data = data.iloc[0:-1, :]

        # display dim
        ll_ = data.shape[0]

        # define model:
        forecaster_model = Forecaster(y=data[target_name], current_dates=data['Date'])

        # need these info
        forecaster_model.set_test_length(n_forecast)
        forecaster_model.generate_future_dates(n_forecast)
        forecaster_model.set_estimator(file_args_dict_['model_name'])

        # initialize
        ii_, jj_, kk_, ll_, rr_, ss_, lr_ = 2, 5, 12, 0.1, 1, 0.1, 0.00001

        # set args
        max_iter = args_dict_['max_iteration']
        ii_range = args_dict_['lags_range']
        jj_range = args_dict_['epochs_range']
        kk_range = args_dict_['width_range']
        ll_range = args_dict_['dropout_range']
        r_range = args_dict_['depth_range']
        ss_range = args_dict_['valsplit_range']
        lr_range = args_dict_['learningrate_range']

        # earlystopping
        early_stopping_rule = EarlyStopping(monitor='loss', patience=200)

        # global iterattion
        zz_ = 0
        while zz_ <= max_iter:

            # tuning lags: ii_
            args_ = []
            curr_range_ = []
            some_result_ = []
            if ii_range is not None:
                for ii_ in ii_range:
                    # name
                    this_nom_ = '_'.join((str(ii_), str(jj_),
                                          str(kk_), str(ll_), str(rr_), str(ss_), str(lr_)))
                    curr_range_.append(ii_)

                    # model
                    forecaster_model.manual_forecast(call_me=str(this_nom_),
                                    lags=ii_,
                                    batch_size=int(np.round(ll_/10)),
                                    epochs=jj_,
                                    validation_split=ss_,
                                    shuffle=True,
                                    activation='tanh',
                                    optimizer='Adam',
                                    learning_rate=lr_,
                                    lstm_layer_sizes=(kk_,)*rr_,
                                    dropout=(ll_,)*rr_,
                                    callbacks=early_stopping_rule,
                                    verbose=0,
                                    plot_loss=True)
                    if plot_test is True:
                        forecaster_model.plot_test_set(order_by='LevelTestSetMAPE',
                                                       models='top_1', ci=True)
                        plt.show()

                    # this result
                    tmp = forecaster_model.export(
                        'model_summaries', determine_best_by=determine_best_by)[
                        ['ModelNickname',
                        'LevelTestSetMAPE',
                        'LevelTestSetRMSE',
                        'LevelTestSetR2',
                        'best_model']
                    ]

                    print(tmp)

                    # collect
                    args_.append(this_nom_)
                    some_result_.append(np.float(tmp.loc[tmp['ModelNickname'] == this_nom_,
                                                         :]['LevelTestSetMAPE']))

                    # checkpoint
                    print('>>> currently, we are at this tuning args combo: ', this_nom_, '<<<')

                # pick the best
                ii_ = curr_range_[np.argmin(some_result_)]
                print('best lags: ', ii_)

            # tuning epochs: jj_
            args_ = []
            curr_range_ = []
            some_result_ = []
            if jj_range is not None:
                for jj_ in jj_range:
                    # name
                    this_nom_ = '_'.join((str(ii_), str(jj_),
                                          str(kk_), str(ll_), str(rr_), str(ss_), str(lr_)))
                    curr_range_.append(jj_)

                    # model
                    forecaster_model.manual_forecast(call_me=str(this_nom_),
                                    lags=ii_,
                                    batch_size=int(np.round(ll_/10)),
                                    epochs=jj_,
                                    validation_split=ss_,
                                    shuffle=True,
                                    activation='tanh',
                                    optimizer='Adam',
                                    learning_rate=lr_,
                                    lstm_layer_sizes=(kk_,)*rr_,
                                    dropout=(ll_,)*rr_,
                                    callbacks=early_stopping_rule,
                                    verbose=0,
                                    plot_loss=True)
                    if plot_test is True:
                        forecaster_model.plot_test_set(order_by='LevelTestSetMAPE',
                                                       models='top_1', ci=True)
                        plt.show()

                    # this result
                    tmp = forecaster_model.export(
                        'model_summaries', determine_best_by=determine_best_by)[
                        ['ModelNickname',
                        'LevelTestSetMAPE',
                        'LevelTestSetRMSE',
                        'LevelTestSetR2',
                        'best_model']
                    ]

                    print(tmp)

                    # collect
                    args_.append(this_nom_)
                    some_result_.append(np.float(tmp.loc[tmp['ModelNickname'] == this_nom_,
                                                         :]['LevelTestSetMAPE']))

                    # checkpoint
                    print('>>> currently, we are at this tuning args combo: ', this_nom_, '<<<')

                # pick the best
                jj_ = curr_range_[np.argmin(some_result_)]
                print('best epochs: ', jj_)

            # tuning width: kk_
            args_ = []
            curr_range_ = []
            some_result_ = []
            if kk_range is not None:
                for kk_ in kk_range:
                    # name
                    this_nom_ = '_'.join((str(ii_), str(jj_),
                                          str(kk_), str(ll_), str(rr_), str(ss_), str(lr_)))
                    curr_range_.append(kk_)

                    # model
                    forecaster_model.manual_forecast(call_me=str(this_nom_),
                                    lags=ii_,
                                    batch_size=int(np.round(ll_/10)),
                                    epochs=jj_,
                                    validation_split=ss_,
                                    shuffle=True,
                                    activation='tanh',
                                    optimizer='Adam',
                                    learning_rate=lr_,
                                    lstm_layer_sizes=(kk_,)*rr_,
                                    dropout=(ll_,)*rr_,
                                    callbacks=early_stopping_rule,
                                    verbose=0,
                                    plot_loss=True)
                    if plot_test is True:
                        forecaster_model.plot_test_set(order_by='LevelTestSetMAPE',
                                                       models='top_1', ci=True)
                        plt.show()

                    # this result
                    tmp = forecaster_model.export(
                        'model_summaries', determine_best_by=determine_best_by)[
                        ['ModelNickname',
                        'LevelTestSetMAPE',
                        'LevelTestSetRMSE',
                        'LevelTestSetR2',
                        'best_model']
                    ]

                    print(tmp)

                    # collect
                    args_.append(this_nom_)
                    some_result_.append(np.float(tmp.loc[tmp['ModelNickname'] == this_nom_,
                                                         :]['LevelTestSetMAPE']))

                    # checkpoint
                    print('>>> currently, we are at this tuning args combo: ', this_nom_, '<<<')

                # pick the best
                kk_ = curr_range_[np.argmin(some_result_)]
                print('best width: ', kk_)

            # tuning dropout rate: ll_
            args_ = []
            curr_range_ = []
            some_result_ = []
            if ll_range is not None:
                for ll_ in ll_range:
                    # name
                    this_nom_ = '_'.join((str(ii_), str(jj_),
                                          str(kk_), str(ll_), str(rr_), str(ss_), str(lr_)))
                    curr_range_.append(ll_)

                    # model
                    forecaster_model.manual_forecast(call_me=str(this_nom_),
                                    lags=ii_,
                                    batch_size=int(np.round(ll_/10)),
                                    epochs=jj_,
                                    validation_split=ss_,
                                    shuffle=True,
                                    activation='tanh',
                                    optimizer='Adam',
                                    learning_rate=lr_,
                                    lstm_layer_sizes=(kk_,)*rr_,
                                    dropout=(ll_,)*rr_,
                                    callbacks=early_stopping_rule,
                                    verbose=0,
                                    plot_loss=True)
                    if plot_test is True:
                        forecaster_model.plot_test_set(order_by='LevelTestSetMAPE',
                                                       models='top_1', ci=True)
                        plt.show()

                    # this result
                    tmp = forecaster_model.export(
                        'model_summaries', determine_best_by=determine_best_by)[
                        ['ModelNickname',
                        'LevelTestSetMAPE',
                        'LevelTestSetRMSE',
                        'LevelTestSetR2',
                        'best_model']
                    ]

                    print(tmp)

                    # collect
                    args_.append(this_nom_)
                    some_result_.append(np.float(tmp.loc[tmp['ModelNickname'] == this_nom_,
                                                         :]['LevelTestSetMAPE']))

                    # checkpoint
                    print('>>> currently, we are at this tuning args combo: ', this_nom_, '<<<')

                # pick the best
                ll_ = curr_range_[np.argmin(some_result_)]
                print('best dropout rate: ', ll_)

            # tuning depth: r_
            args_ = []
            curr_range_ = []
            some_result_ = []
            if r_range is not None:
                for rr_ in r_range:
                    # name
                    this_nom_ = '_'.join((str(ii_), str(jj_),
                                          str(kk_), str(ll_), str(rr_), str(ss_), str(lr_)))
                    curr_range_.append(rr_)

                    # model
                    forecaster_model.manual_forecast(call_me=str(this_nom_),
                                    lags=ii_,
                                    batch_size=int(np.round(ll_/10)),
                                    epochs=jj_,
                                    validation_split=ss_,
                                    shuffle=True,
                                    activation='tanh',
                                    optimizer='Adam',
                                    learning_rate=lr_,
                                    lstm_layer_sizes=(kk_,)*rr_,
                                    dropout=(ll_,)*rr_,
                                    callbacks=early_stopping_rule,
                                    verbose=0,
                                    plot_loss=True)
                    if plot_test is True:
                        forecaster_model.plot_test_set(order_by='LevelTestSetMAPE',
                                                       models='top_1', ci=True)
                        plt.show()

                    # this result
                    tmp = forecaster_model.export(
                        'model_summaries', determine_best_by=determine_best_by)[
                        ['ModelNickname',
                        'LevelTestSetMAPE',
                        'LevelTestSetRMSE',
                        'LevelTestSetR2',
                        'best_model']
                    ]

                    print(tmp)

                    # collect
                    args_.append(this_nom_)
                    some_result_.append(np.float(tmp.loc[tmp['ModelNickname'] == this_nom_,
                                                         :]['LevelTestSetMAPE']))

                    # checkpoint
                    print('>>> currently, we are at this tuning args combo: ', this_nom_, '<<<')

                # pick the best
                rr_ = curr_range_[np.argmin(some_result_)]
                print('best depth: ', rr_)

            # tuning validation split: ss_
            args_ = []
            curr_range_ = []
            some_result_ = []
            if ss_range is not None:
                for ss_ in ss_range:
                    # name
                    this_nom_ = '_'.join((str(ii_), str(jj_),
                                          str(kk_), str(ll_), str(rr_), str(ss_), str(lr_)))
                    curr_range_.append(ss_)

                    # model
                    forecaster_model.manual_forecast(call_me=str(this_nom_),
                                    lags=ii_,
                                    batch_size=int(np.round(ll_/10)),
                                    epochs=jj_,
                                    validation_split=ss_,
                                    shuffle=True,
                                    activation='tanh',
                                    optimizer='Adam',
                                    learning_rate=lr_,
                                    lstm_layer_sizes=(kk_,)*rr_,
                                    dropout=(ll_,)*rr_,
                                    callbacks=early_stopping_rule,
                                    verbose=0,
                                    plot_loss=True)
                    if plot_test is True:
                        forecaster_model.plot_test_set(order_by='LevelTestSetMAPE',
                                                       models='top_1', ci=True)
                        plt.show()

                    # this result
                    tmp = forecaster_model.export(
                        'model_summaries', determine_best_by=determine_best_by)[
                        ['ModelNickname',
                        'LevelTestSetMAPE',
                        'LevelTestSetRMSE',
                        'LevelTestSetR2',
                        'best_model']
                    ]

                    print(tmp)

                    # collect
                    args_.append(this_nom_)
                    some_result_.append(np.float(tmp.loc[tmp['ModelNickname'] == this_nom_,
                                                         :]['LevelTestSetMAPE']))

                    # checkpoint
                    print('>>> currently, we are at this tuning args combo: ', this_nom_, '<<<')

                # pick the best
                ss_ = curr_range_[np.argmin(some_result_)]
                print('best validation split: ', ss_)

            # tuning learning rate: lr_
            args_ = []
            curr_range_ = []
            some_result_ = []
            if lr_range is not None:
                for lr_ in lr_range:
                    # name
                    this_nom_ = '_'.join((str(ii_), str(jj_),
                                          str(kk_), str(ll_), str(rr_), str(ss_), str(lr_)))
                    curr_range_.append(lr_)

                    # model
                    forecaster_model.manual_forecast(call_me=str(this_nom_),
                                    lags=ii_,
                                    batch_size=int(np.round(ll_/10)),
                                    epochs=jj_,
                                    validation_split=ss_,
                                    shuffle=True,
                                    activation='tanh',
                                    optimizer='Adam',
                                    learning_rate=lr_,
                                    lstm_layer_sizes=(kk_,)*rr_,
                                    dropout=(ll_,)*rr_,
                                    callbacks=early_stopping_rule,
                                    verbose=0,
                                    plot_loss=True)
                    if plot_test is True:
                        forecaster_model.plot_test_set(order_by='LevelTestSetMAPE',
                                                       models='top_1', ci=True)
                        plt.show()

                    # this result
                    tmp = forecaster_model.export(
                        'model_summaries', determine_best_by=determine_best_by)[
                        ['ModelNickname',
                        'LevelTestSetMAPE',
                        'LevelTestSetRMSE',
                        'LevelTestSetR2',
                        'best_model']
                    ]

                    print(tmp)

                    # collect
                    args_.append(this_nom_)
                    some_result_.append(np.float(tmp.loc[tmp['ModelNickname'] == this_nom_,
                                                         :]['LevelTestSetMAPE']))

                    # checkpoint
                    print('>>> currently, we are at this tuning args combo: ', this_nom_, '<<<')

                # pick the best
                lr_ = curr_range_[np.argmin(some_result_)]
                print('best learning rate: ', lr_)

            # finalize: build the best model
            forecaster_model.manual_forecast(call_me=str(this_nom_),
                            lags=ii_,
                            batch_size=int(np.round(ll_/10)),
                            epochs=jj_,
                            validation_split=ss_,
                            shuffle=True,
                            activation='tanh',
                            optimizer='Adam',
                            learning_rate=lr_,
                            lstm_layer_sizes=(kk_,)*rr_,
                            dropout=(ll_,)*rr_,
                            callbacks=early_stopping_rule,
                            verbose=0,
                            plot_loss=True)
            if plot_test is True:
                forecaster_model.plot_test_set(order_by='LevelTestSetMAPE',
                                               models='top_1', ci=True)
                plt.show()

            # this result
            tmp = forecaster_model.export(
                'model_summaries', determine_best_by=determine_best_by)[
                ['ModelNickname',
                'LevelTestSetMAPE',
                'LevelTestSetRMSE',
                'LevelTestSetR2',
                'best_model']
            ]

            print('>>>>>>>>>> final model is here: <<<<<<<<')
            print(tmp)

            # view result
            this_tuning_result_ = forecaster_model.export(
                'model_summaries',
                determine_best_by='LevelTestSetMAPE')[
                ['ModelNickname',
                'LevelTestSetMAPE',
                'LevelTestSetRMSE',
                'LevelTestSetR2',
                'best_model'] ]

            # directory
            # os.chdir(PATH_TO_SAVE_DATA)
            # os.listdir(), nom_of_this_siteid_this_ta_data_.split('.')[0]+'_tuning_results_.csv'

            # display name
            print(nom_of_this_siteid_this_ta_data_.split('.')[0]+'_tuning_results_.csv')

            # save
            this_tuning_result_.to_csv(nom_of_this_siteid_this_ta_data_.split('.')[0]+
                                       '_tuning_results_.csv')

            # checkpoint
            print("############################################################################")
            print(">>>>>>>>>> finished with global iteration: ", zz_, '/', max_iter, " <<<<<<<<<<")
            print("############################################################################")

            zz_ += 1

        # output
        return {
            'n_forecast': n_forecast,
            'data': target_data_,
            'model': forecaster_model,
            'target_name': target_name,
            'tuning_result': this_tuning_result_ }

    def get_best_model_(
        data = pd.DataFrame(),
        this_tuning_result_ = pd.DataFrame(),
        forecaster_model = None,
        n_forecast = float,
        target_name = str,
        nom_of_this_siteid_this_ta_data_ = float,
        file_args_dict_ = dict,
        partitions = dict,
        save_to_s3 = True,
        go_live = False,
        TOY_PATH = str,
        LIVE_PATH = str
    ):

        """get_best_model_: get the best model from tuning results"""

        # which model
        which_model = this_tuning_result_.iloc[0,0]

        # data for plot
        new_data_frame = pd.DataFrame()
        new_data_frame['Date'] = data['Date']
        new_data_frame[target_name] = data[target_name]

        # to_be_added
        to_be_added = pd.DataFrame([[pd.date_range(
            start=pd.to_datetime('today').date(),
            freq='M',
            periods=n_forecast)[i].date(), np.nan] for i in range(n_forecast)])
        to_be_added.columns = new_data_frame.columns

        # update new_data_frame
        new_data_frame = pd.concat([new_data_frame, to_be_added], axis=0)

        # data for plot
        some_length_ = new_data_frame.shape[0] - len(
            forecaster_model.history[which_model]['Forecast'])
        new_data_frame['forecast'] = [
            np.nan for i in range(some_length_)] + forecaster_model.history[which_model]['Forecast']
        new_data_frame['ub'] = [
            np.nan for i in range(some_length_)] + forecaster_model.history[which_model]['UpperCI']
        new_data_frame['lb'] = [
            np.nan for i in range(some_length_)] + forecaster_model.history[which_model]['LowerCI']
        new_data_frame['fitted'] = [
            forecaster_model.history[which_model]['LevelFittedVals'][0] for i in range(
                int(new_data_frame.shape[0] - len(
                    forecaster_model.history[which_model]['LevelFittedVals'] + [
                        np.nan for i in range(n_forecast)])))] + forecaster_model.history[
            which_model]['LevelFittedVals'] + [np.nan for i in range(n_forecast)]

        # save
        print('Saving locally to Sagemaker...')
        new_data_frame.to_csv(nom_of_this_siteid_this_ta_data_.split('.')[0]+
                              '_forecasting_results_.csv')
        print('Saved the forecasting results to sagemaker locally.')

        # reindex
        new_data_frame.index = new_data_frame['Date']

        # redefine using agreed upon data contract
        print('Getting data ready for s3...')
        tmp = new_data_frame
        tmp['Date'] = [pd.to_datetime(
            new_data_frame['Date'].to_numpy()[i]).date() for i in range(len(new_data_frame))]
        tmp = tmp.iloc[-n_forecast::, :][['Date', target_name, 'forecast', 'lb', 'ub']]
        tmp['model_name'] = file_args_dict_['model_name']
        tmp['model_publish_date'] = partitions['model_publish_date']
        tmp['by_field'] = file_args_dict_['by_field']
        tmp['by_value'] = file_args_dict_['by_value']
        tmp['frequency'] = file_args_dict_['frequency']
        tmp['enum'] = file_args_dict_['enum']
        tmp['forecast_date'] = tmp['Date']
        tmp = tmp[[
            'frequency',
            'by_field',
            'by_value',
            'forecast_date',
            'forecast',
            'lb',
            'ub']]
        tmp.columns = [
            "frequency",
            "by_field",
            "by_value",
            "forecast_date",
            "forecast",
            "forecast_low",
            "forecast_high"
        ]
        tmp.index = np.arange(0, len(tmp))
        df_after_datacontract = tmp

        # save to s3
        if save_to_s3:
            print("Saving data to S3...")
            if go_live:
                main_path=TOY_PATH
            else:
                main_path=LIVE_PATH
            main_path=main_path+'/model_name='
            this_s3_path_for_inference_ = main_path+partitions['model_name']
            this_s3_path_for_inference_ = this_s3_path_for_inference_+'/model_publish_date='
            this_s3_path_for_inference_ = this_s3_path_for_inference_+str(partitions['model_publish_date'])+'/forecasting_results_'
            this_s3_path_for_inference_ = this_s3_path_for_inference_+nom_of_this_siteid_this_ta_data_+'.snappy.parquet'
            df_after_datacontract.to_parquet(this_s3_path_for_inference_, compression='gzip', index=False)
            print('Saved to:', this_s3_path_for_inference_)
            print('>>>>>>>>>> Just saved to s3! <<<<<<<<<<')

        # output
        return {
            'which_model': which_model,
            'new_data_frame': new_data_frame,
            'data_saved_as_parquet': df_after_datacontract
        }

    def plot_trained_model(
        new_data_frame = pd.DataFrame,
        which_model = str,
        target_name = str,
        n_forecast = float,
        this_tuning_result_ = pd.DataFrame,
        nom_of_this_siteid_this_ta_data_=str,,
        width = int,
        height = int
    ):

        """plot_trained_model: plot the graph"""

        # mape
        final_mape_ = this_tuning_result_
        final_mape_ = final_mape_.loc[final_mape_['ModelNickname'] == which_model, :]
        final_mape_ = final_mape_['LevelTestSetMAPE']
        final_mape_ = final_mape_.to_numpy()[0]

        # plotly
        print('Prepare for visualization using plotly ... ')
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=new_data_frame['Date'],
                y=new_data_frame['forecast'],
                name='forecast',
                marker_color=px.colors.qualitative.Dark24[3]
            ))
        fig.add_trace(
            go.Scatter(
                x=new_data_frame['Date'],
                y=new_data_frame['fitted'],
                name='fitted',
                marker_color=px.colors.qualitative.Dark24[5]
            ))
        new_data = new_data_frame['Date'][-n_forecast::]
        new_data_upper = new_data_frame['ub'][-n_forecast::]
        new_data_lower = new_data_frame['lb'][-n_forecast::]
        fig.add_trace(
            go.Scatter(
                x=new_data,
                y=new_data_upper,
                name='ub'
            ))
        fig.add_trace(
            go.Scatter(
                x=new_data,
                y=new_data_lower,
                name='lb',
            ))
        fig.add_trace(
            go.Bar(
                x=new_data_frame['Date'],
                y=new_data_frame[target_name],
                name='truth',
                marker_color=px.colors.qualitative.Dark24[0]
            ))
        fig.update_layout(
            autosize=False,
            width=width,
            height=height,
            # title='Kits In (by Month) | Data: ' + nom_of_this_siteid_this_ta_data_ +' | '+'<br>CI: Upper bound='+str(np.round(new_data_frame['ub'].iloc[-n_forecast], 2))+', Lower bound='+str(np.round(new_data_frame['lb'].iloc[-n_forecast], 2))+', MAPE='+str(np.round(final_mape_, 3))+'; <br>*Next month prediction='+str(int(np.round(new_data_frame['forecast'].iloc[-n_forecast]))),
            xaxis=dict(title='Date (by month)'),
            yaxis=dict(title='Number of Kits (in) <br>Data: ' + nom_of_this_siteid_this_ta_data_),
            hoverlabel=dict(
                bgcolor="white",
                font_size=19,
                font_family="Rockwell"
            )
        )
        fig.show()

    def interactive_ts_plot_(
        data = pd.DataFrame,
        args = dict
    ):
        """interactive time-series plot"""

        # data
        df = data
        width = args['width']
        height = args['height']

        # figure
        fig = go.Figure()
        for j in range(df.shape[1]):
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df.iloc[:, j],
                    name='site '+str(j),
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
