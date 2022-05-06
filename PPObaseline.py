#!pip uninstall tensorflow
!pip install tensorflow-gpu==1.15.0 tensorflow==1.15.0 stable-baselines gym-anytrading gym
import pandas as pd
import ssl
import time
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from BinanceDataRequest import getUnixTimestamp, OHLC_binance
from stable_baselines import PPO2
from stable_baselines import A2C
from stable_baselines.common.policies import MlpLnLstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv

if __name__=="__main__":
    
    ssl._create_default_https_context = ssl._create_unverified_context
    
    data_pd = OHLC_binance(pair='ETH/USDT', starttime='2017-08-17T00:00:00', 
                       endtime='2021-02-27T00:00:00', interval='1h')
    
    data_pd.reset_index(inplace=True)
    data = data_pd.drop(axis=1, labels=['unix', 'close_unix', 'volume_from', 'marketorder_volume', 
                                        'marketorder_volume_from', 'tradecount', 'date'])

    training_data = (10,1000)
    test_data = (1000,1100)
    
    print("Random Initialization")
    curState = env.reset()
    i = 0
    
    while True : 
        curAction = env.action_space.sample()
        newState, newRew, isDone, information = env.step(curAction)
        if isDone:
            print(information)
            break
            
    plt.figure(figsize=(15,6))
    plt.cla()
    env.render_all()
    plt.show()
    
    env_creator = lambda: gym.make('stocks-v0', df=data, frame_bound = training_data, window_size=10)
    env = DummyVecEnv([env_creator])
    model = PPO2('MlpLstmPolicy', env, verbose=1, nminibatches = 1)
    model.learn(total_timesteps=10000)

    env = gym.make('stocks-v0', df=data, frame_bound=test_data, window_size=10)
    cur_obs = env.reset()
    
    while True:
        cur_obs = cur_obs[np.newaxis, ...]
        curAction, states = model.predict(cur_obs)
        cur_obs, rewards, done, info = env.step(curAction)
        if i%10==0:
            print(info)
        if done:
            print("info", info)
            break
    
    plt.figure(figsize=(15,6))
    plt.cla()
    env.render_all()
    plt.show()

    env_creator = lambda: gym.make('stocks-v0', df=data, frame_bound = training_data, window_size=10)
    env = DummyVecEnv([env_creator])
    model = A2C('MlpLstmPolicy', env, verbose=1)
    model.learn(total_timesteps=10000)

    env = gym.make('stocks-v0', df=data, frame_bound=test_data, window_size=10)
    cur_obs = env.reset()
    while True:
        cur_obs = cur_obs[np.newaxis, ...]
        curAction, states = model.predict(cur_obs)
        cur_obs, rewards, done, info = env.step(curAction)
        if i%10==0:
            print(info)
        if done:
            print("info", info)
            break
    
    plt.figure(figsize=(15,6))
    plt.cla()
    env.render_all()
    plt.show()