import tensorflow
# ensure results are reproducible
tensorflow.keras.utils.set_random_seed(1234)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay

import numpy as np
import math
import pandas as pd
import ssl

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from statsmodels.tsa.stattools import adfuller

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

from BinanceDataRequest import getUnixTimestamp, OHLC_binance


def difference_data(data):
    differenced = data.copy()

    for col in differenced.columns:
        differenced[col] = differenced[col] - differenced.shift(1)[col]
        
    # drop first row with nan values
    differenced = differenced.iloc[1:, :]

    return differenced


# Dickey-Fuller test to make sure data is stationary
def dicky_fuller(differenced):
    for col in differenced.columns:
        p_test = adfuller(differenced[col])
        print(col)
        print('Dickey–Fuller test: %f' % p_test[0])
        print('p-value: %f' % p_test[1])

        if p_test[1] < 0.05:
            print('Stationary data!')

            
# reverse difference method for predictions
def reverse_difference(prev_value, predictions):
    rev_diff = [prev_value + predictions[0]]
    for i in range(1, len(predictions)):
        diff = rev_diff[i - 1] + predictions[i]
        rev_diff.append(diff)
        
    return rev_diff


def create_dataset(data, time_step):
    x_set = []
    y_set = []
        
    for i in range(len(data) - time_step - 1):
        x1 = data[i:(i + time_step), 0]    
        x_set.append(x1)
        y_set.append(data[i + time_step, 0])
            
    return np.array(x_set), np.array(y_set)


def build_LSTM(hidden1, hidden2, kernel_initializer, activation1, activation2, dropout):
    model = Sequential()

    # first LSTM layer
    model.add(LSTM(units=hidden1, return_sequences=True, input_shape=(10, 1), 
                   kernel_initializer=kernel_initializer, activation=activation1))
    model.add(Dropout(dropout))

    # second LSTM layer
    model.add(LSTM(units=hidden1, return_sequences=True, 
                   kernel_initializer=kernel_initializer, activation=activation1))
    model.add(Dropout(dropout))

    # third LSTM layer
    model.add(LSTM(units=hidden1, return_sequences=True, 
                   kernel_initializer=kernel_initializer, activation=activation1))
    model.add(Dropout(dropout))

    # fourth LSTM layer
    model.add(LSTM(units=hidden2, kernel_initializer=
                   kernel_initializer, activation=activation1))
    model.add(Dropout(dropout))

    # output layer
    model.add(Dense(units=1, activation=activation2))
    
    return model
    
        
if __name__=="__main__":
    
    ssl._create_default_https_context = ssl._create_unverified_context
    
    # can pull various time frames for testing
    data_pd = OHLC_binance(pair='ETH/USDT', starttime='2017-08-17T00:00:00', 
                       endtime='2021-02-27T00:00:00', interval='1h')

    # convert raw data to CSV file if needed
    # filename = "ETH_Binance.csv"  
    # data_pd.to_csv(filename, index=False) 
    
    # data_pd = data_pd.set_index(['date'])
    data_pd.reset_index(inplace=True)
    # data_pd.head()

    # get rid of unnecessary features
    data = data_pd.drop(axis=1, labels=['unix', 'close_unix', 'volume_from', 'marketorder_volume', 
                                        'marketorder_volume_from', 'tradecount', 'date'])

    data.head()
            
    # determine if any nan or infinity values need to be handled
    data_pd.isin([np.nan, np.inf, -np.inf]).sum()
    
    # Differencing actually yielded worse results so we took it out

    # use difference method to remove seasonality
    # assumes interval of 1 (subtracting adjacent rows)
    # data['differenced'] = data['close'].diff().values
    # data.head()

    # make sure there are no null values after differencing
    # data['differenced_checked'] = np.where(pd.isnull(data['differenced']), data['close'], data['differenced'])
    # check to make sure differencing is correct
    # data['reverse_difference'] = data['differenced_checked'].cumsum()
    
    # dicky_fuller(data['differenced_checked'])
    
    # visualize ETH price, if desired
    %matplotlib inline
    FIGURE_SIZE = (20, 10)
    plt.rcParams['axes.grid'] = True
    %matplotlib inline

    data.set_index('date')['close'].plot(figsize=FIGURE_SIZE)
    # change as needed! 
    plt.title('ETH price from 08/17/2017 - 02/27/2021')
    plt.show()
    
    # data['close_norm'] = scaler.fit_transform(np.array(data['close']).reshape(-1, 1))
    # train_close = data['close_norm'][0]
    # test_close = data['close_norm'][val_end]

    # drop first row with nan values but keep the initial closing price for reverse differencing
    # data = data.iloc[1:, :]

    # scaler_y = MinMaxScaler()
    # scaled_data_y = scaler_y.fit_transform(data[['differenced']])
    
    y = data['close']

    training_size = int(len(y) * 0.7)
    val_end = int(len(y) * 0.8)

    # normalize
    scaler = MinMaxScaler()
    y = scaler.fit_transform(np.array(y).reshape(-1, 1))

    train_data, val_data, test_data = y[0:training_size, :]
    val_data = y[training_size:val_end, :]
    test_data =  y[val_end:len(y), :]
    
    time_step = 10

    X_train, y_train = create_dataset(train_data, time_step)
    X_val, y_val = create_dataset(val_data, time_step)
    X_test, y_test = create_dataset(test_data, time_step)
    
    # if not using selu, can put kernel_initializer=None
    model = build_LSTM(hidden1=75, hidden2=50, kernel_initializer='lecun_normal', 
                       activation1='selu', activation2='linear', dropout=0.2):

    # lr_schedule = ExponentialDecay(
    #    0.001,decay_steps=100000,
    #    decay_rate=0.96,
    #    staircase=True)

    # opt = Adam(learning_rate=lr_schedule)

    model.compile(loss='mean_squared_error', optimizer='adam')
    model.summary()
    
    # use early stopping
    monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=23, 
                        verbose=1, mode='auto', restore_best_weights=False)

    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), 
                        verbose=1, epochs=300, callbacks=[monitor])
    
    # saves the model architecture, weights, and 
    # the traced Tensorflow subgraphs of the call functions in current directory
    # model.save('ethereum_model')
    
    # predictions 
    train_predict = model.predict(X_train)
    tpredict = model.predict(X_test)

    # undo MinMax normalization
    train_predict = scaler.inverse_transform(train_predict)
    test_predict = scaler.inverse_transform(tpredict)
    
    # reverse differencing 
    # train_predict = reverse_difference(train_close, train_predict)
    # test_predict = reverse_difference(test_close, tpredict)

    # visualization
    # shift train predictions 
    fig, ax = plt.subplots(figsize=(20, 10))
    train_plot = np.empty_like(y)
    train_plot[:, :] = np.nan
    train_plot[time_step:len(train_predict) + time_step, :] = train_predict

    # shift test predictions 
    test_plot = np.empty_like(y)
    test_plot[:, :] = np.nan
    test_plot[val_end + (time_step) + 1:len(y), :] = test_predict

    # plot ground truth + predictions
    plt.plot(data['close'])
    plt.plot(train_plot)
    plt.plot(test_plot)
    plt.legend(['Actual', 'Train', 'Test'])
    plt.xlabel('Time Steps')
    plt.ylabel('ETH Price')
    plt.title('ETH price from 08/17/2017 - 02/27/2021') 
    plt.show()
    
    # test set only plot for easier visualization
    fig, ax = plt.subplots(figsize=(20, 10))
    plt.plot(scaler.inverse_transform(y_test.reshape(-1, 1)))
    plt.plot(test_predict)
    plt.legend(['Actual', 'Test'])
    plt.xlabel('Time Steps')
    plt.ylabel('ETH Price')
    plt.title('ETH price (test set)')
    plt.show()
    
    # loss plot
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.show()

    # determine MSE
    print('the mean squared error is: ', mean_squared_error(tpredict, y_test))
    
    # save predictions for PPO agent
    predict_df = pd.DataFrame(test_predict, columns = ['close'])
    predict_df.reset_index(inplace=False)
    filename = 'ETH_predictions.csv'  
    predict_df.to_csv(filename, index=False)