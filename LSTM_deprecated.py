import numpy as np
import math
import pandas as pd
import ssl

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from BinanceDataRequest import getUnixTimestamp, OHLC_binance

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


def create_dataset(data, time_step):
    x_set = []
    y_set = []
    for i in range(len(data) - time_step - 1):
        x1 = data[i:(i + time_step), 0]    
        x_set.append(x1)
        y_set.append(data[i + time_step, 0])    
    return np.array(x_set), np.array(y_set)


class PredictionModel(nn.Module):
    def __init__(self, input_size=10, hidden_layers=50, num_layers=4, output_size=1, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size = input_size, hidden_size = hidden_layers, num_layers = num_layers, dropout = dropout)
        self.l1 = nn.Linear(hidden_layers, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.lstm(x)
        x = self.l1(x[0])
        x = self.relu(x)
        return x
    
        
if __name__=="__main__":
    
    ssl._create_default_https_context = ssl._create_unverified_context
    
    data_pd = OHLC_binance(pair='ETH/USDT', starttime='2017-08-17T00:00:00', 
                       endtime='2021-02-27T00:00:00', interval='1h')
    
    data_pd.reset_index(inplace=True)
    data = data_pd.drop(axis=1, labels=['unix', 'close_unix', 'volume_from', 'marketorder_volume', 
                                        'marketorder_volume_from', 'tradecount', 'date'])
            
    data_pd.isin([np.nan, np.inf, -np.inf]).sum()
    
    y = data['close']
    training_size = int(len(y) * 0.7)
    val_end = int(len(y) * 0.8)

    scaler = MinMaxScaler()
    y = scaler.fit_transform(np.array(y).reshape(-1, 1))

    train_data, val_data, test_data = y[0:training_size, :]
    val_data = y[training_size:val_end, :]
    test_data =  y[val_end:len(y), :]

    X_train, y_train = create_dataset(train_data, 10)
    X_val, y_val = create_dataset(val_data, 10)
    X_test, y_test = create_dataset(test_data, 10)
    
    model = PredictionModel(input_size = 10, hidden_layers = 50, num_layers = 4, output_size = 1, dropout = 0.2)
    model.to("cuda")

    optimizer = torch.optim.Adam(model.parameters(),  lr=0.001)
    criterion = nn.MSELoss()
    model = model.double()
    X_train = torch.tensor(X_train, requires_grad = True).reshape(X_train.shape[0],1,10).to("cuda")
    y_train = torch.tensor(y_train, requires_grad = True).to("cuda")

    for epoch in range(100):
        optimizer.zero_grad()
        target = model(X_train)
        loss = criterion(target.reshape(y_train.shape),y_train)
        loss.backward()
        optimizer.step()

        print(f'Epoch {epoch+1} \t\t Training Loss: {loss}')

    X_test = torch.tensor(X_test).reshape(X_test.shape[0],1,10)
    y_test = torch.tensor(y_test)
    model = model.cpu()
    
    train_predict = model(X_train.cpu()).reshape(-1,1).detach()
    tpredict = model(X_test).reshape(-1,1).detach()

    train_predict = scaler.inverse_transform(train_predict)
    test_predict = scaler.inverse_transform(tpredict)