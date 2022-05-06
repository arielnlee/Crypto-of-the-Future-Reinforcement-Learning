import numpy as np
import math
import gym
from gym import spaces
import pandas as pd

from sklearn.preprocessing import MinMaxScaler

!pip install tensorflow-gpu==1.15.0 tensorflow==1.15.0 stable-baselines gym-anytrading gym 
from stable_baselines.common.policies import MlpLnLstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

# import warnings
# tensorflow 1.15 leads to lots of deprecation warnings
# warnings.filterwarnings('ignore', category=DeprecationWarning) 

!pip install empyrical
from empyrical.stats import omega_ratio

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure


class TradingETH(gym.Env):
    scaler = MinMaxScaler()
    
    def __init__(self, data, env_config, predictions_data):
        super(TradingETH, self).__init__()
        
        self.init_cash = env_config['initial_cash']
        self.curr_cash = self.init_cash
        self.commission = env_config['commission']
        self.forecast_len = env_config['forecast_len']
        self.reward_strategy = env_config['reward_strategy']
        self.data_preprocessed = env_config['data_preprocessed']
        self.obs_features = ['close', 'open', 'low', 'high', 'volume']
        self.min_limit = env_config['min_limit']
        self.data = data
        self.predictions_data = predictions_data['close']
        
        # https://medium.com/swlh/states-observation-and-action-spaces-in-reinforcement-learning-569a30a8d2a1
        # two discrete action spaces consisting of 3 and 9 actions, respectively
        # first action will never be larger than 2 and second action will never be larger than 8
        # for example, a sample of the action space could return array([2, 1]) or array([0, 6]), etc.
        # Discrete(3) --> 3 discrete points each mapped to interval range [0, 2], inclusive (buy, hold, sell)
        # Discrete(9) --> 9 discrete points each mapped to interval range [0, 8], inclusive - this will determine the 
        # fraction of coins to buy/sell
        self.action_space = spaces.MultiDiscrete([3, 9])
        
        # shape=(# obs_features + self.history.shape[1], forecast_len + 1)
        self.observation_space = spaces.Box(low=0, high=1, 
                                            shape=(7, self.forecast_len + 1), dtype=np.float16)
    
    
    def ret_price(self):
        # curr_price = random.uniform(self.df.loc[self.current, 'open'],
        #                            self.df.loc[self.current, 'close'])
        curr_price = self.df['close'].values[self.forecast_len + self.current]
        return curr_price
        
    
    def next_obs(self):
        # if data has not been preprocessed
        if self.data_preprocessed == False:
            df_obs = self.df.values[:1 + self.current + self.forecast_len]
            scaled = self.scaler.fit_transform(df_obs)
            df_obs = pd.DataFrame(scaled, columns=self.df.columns)
            df_obs_pred = [self.predictions_df[:1 + self.current + self.forecast_len]]
            scaled_pred = self.scaler.fit_transform(df_obs_pred)
            df_obs_pred = pd.DataFrame(scaled_pred[0])
        else:
            df_obs = self.df.copy()
            df_obs_pred = self.predictions_df.copy()

        observation = np.array([df_obs['volume'].values[self.current:1 + self.current + self.forecast_len],
                               df_obs_pred.values[self.current:1 + self.current + self.forecast_len].flatten()])

        # observation = np.array([df_obs[feat].values[self.current:1 + self.current + self.forecast_len] for feat in self.obs_features])
        
        # if data has not been preprocessed
        if self.data_preprocessed == False:
            scale_hist = self.scaler.fit_transform(self.history)
        else:
            scale_hist = self.history.copy()
            
        observation = np.append(observation, scale_hist[:, -(self.forecast_len + 1):], axis=0)
        
        return observation
        
    
    def action(self, action, curr_price):
        act = action[0]
        fraction_coins = action[1] / 9
        
        sell_ETH = 0
        buy_ETH = 0
        plus_cash = 0
        minus_cash = 0
        
        # action is buy
        if act == 0 and self.curr_cash >= self.min_limit:
            buy_ETH = self.curr_cash / curr_price * fraction_coins
            minus = buy_ETH * curr_price * (1 + self.commission)          
            self.ETH_held += buy_ETH
            self.curr_cash -= minus_cash

        # action is sell
        if act == 1 and self.ETH_held >= self.min_limit:
            sell_ETH = self.ETH_held * fraction_coins
            plus = sell_ETH * curr_price * (1 - self.commission)
            self.ETH_held -= sell_ETH
            self.curr_cash += plus_cash

        # else action is hold
            
        if buy_ETH > 0:
            self.trade_hist.append({'step': self.forecast_len + self.current, 
                                    'type': 'buy',
                                    'ETH amount': buy_ETH,
                                    'cash amount': minus_cash})
        elif sell_ETH > 0:
            self.trade_hist.append({'step': self.forecast_len + self.current, 
                                    'type': 'sell',
                                    'ETH amount': sell_ETH,
                                    'cash amount': plus_cash})
            
        self.net.append(self.curr_cash + self.ETH_held * curr_price)
        
        self.history = np.append(self.history, [[self.curr_cash], [buy_ETH], 
                                                [minus_cash], [sell_ETH], [plus_cash]], axis=1)   
        
    
    def reward(self):      
        if self.reward_strategy == 'basic':
            # reward is net profit between current and previous step
            reward = self.net[-1] - self.net[-2]
        elif self.reward_strategy == 'omega_ratio':
            if self.current < self.forecast_len:
                window = self.current
            else:
                window = self.forecast_len

            profit = np.diff(self.net[-window:])

            if np.count_nonzero(profit) < 1:
                reward = 0
            # from empyrical documentation, annualization is factor to convert returns into a daily value
            # using 365 * 24 makes it hourly 
            reward = omega_ratio(profit, annualization=365*24)

            # reward shouldn't be nan 
            if np.isnan(reward):
                reward = 0
            
        return reward
        
    
    def step(self, action):
        curr_price = self.ret_price()
        self.action(action, curr_price)
        self.current += 1
        obs = self.next_obs()
        reward = self.reward()
        # stop if net loss is more than 2000
        done = self.net[-1] <= self.init_cash - 2000  
        
        return obs, reward, done, {}
        
        
    def reset(self):
        self.curr_cash = self.init_cash
        self.df = self.data[:len(self.data) - 1]
        self.predictions_df = self.predictions_data[:len(self.data) - 1]
        self.price_history = []
        self.ETH_held = 0
        self.net = [self.init_cash]
        self.current = 0
        # current cash, bought, minus cash, sold, plus cash
        self.history = np.repeat([[self.curr_cash], [0], [0], [0], [0]], self.forecast_len + 1, axis=1) 
        self.trade_hist = []
        
        return self.next_obs()
  

    def render(self, **kwargs):
        self.price_history.append(self.ret_price())
        temp = self.current + self.forecast_len
        print('ETH price: ', self.ret_price())
        print('Bought: ', self.history[2][temp])
        print('Sold: ', self.history[4][temp])
        print('Remaining cash: ', self.history[0][temp])
        print('Net worth: ', self.net[-1])
        # print('ETH held: ', self.ETH_held)
        print('\n')


    def render_all(self):
        buy = []
        p_buy = []
        sell = []
        p_sell = []
        for i in range(len(self.history[1])):
            if self.history[1][i] != 0:
                buy.append(i)
                p_buy.append(self.df['close'].values[i])
            elif self.history[3][i] != 0:
                sell.append(i)
                p_sell.append(self.df['close'].values[i])
        plt.figure(figsize=(15, 15))
        plt.title('Profit: ' + str(self.net[-1] - self.init_cash))
        plt.plot(self.price_history)
        plt.plot(sell, p_sell, 'r.', markersize=6)
        plt.plot(buy, p_buy, 'g.', markersize=6)
        plt.show()
        
if __name__=="__main__":
            
    # start with $10,000, handling fee of 0.25% of each transaction amount, 0.001 minimum trading unit
    env_config = {'initial_cash': 10000,
                  'commission': 0.0025,
                  'forecast_len': 12,
                  'data_preprocessed': False, 
                  'reward_strategy': 'omega_ratio',
                  'min_limit': 0.001}
    data_pd = OHLC_binance(pair='ETH/USDT', starttime='2017-08-17T00:00:00',
                           endtime='2021-02-27T00:00:00', interval='1h')
    
    # convert raw data to CSV file if needed
    # filename = "ETH_Binance.csv"  
    # data_pd.to_csv(filename, index=False)
    
    data_pd.reset_index()

    # get rid of unnecessary features
    data = data_pd.drop(axis=1, labels=['unix', 'close_unix', 'volume_from', 'marketorder_volume', 
                                        'marketorder_volume_from', 'tradecount', 'date'])
    
    test_len = int(len(data) * 0.2)
    train_len = int(len(data)) - test_len
    train_df = data[:train_len]
    test_df = data[train_len + 9:]
    
    # for google colab only
    # from google.colab import drive
    # drive.mount('/content/gdrive')
    # directory = '/content/gdrive/Shared drives/Deep Learning Project/ETH_predictions.csv'
    # predictions_df = pd.read_csv (directory)
    predictions_df = pd.read_csv('ETH_predictions.csv')

    train_env = DummyVecEnv([lambda: TradingETH(train_df, env_config, train_df)])

    #train_env.seed(42)
    
    # change verbose=0 to stop print out
    model = PPO2(MlpLnLstmPolicy, train_env, verbose=1, nminibatches=1, learning_rate=0.0003)
    model.learn(total_timesteps=train_len)
    
    test_env = DummyVecEnv([lambda: TradingETH(test_df, env_config, predictions_df)])

    #test_env.seed(42)

    obs = test_env.reset()
    reward_sum = 0

    for i in range(test_len - 23):
        action, state = model.predict(obs)
        # PPO2 requires format obs, rewards, done, information
        obs, rewards, done, info = test_env.step(action)
        test_env.render()
        rewards_sum += rewards
    
    # plot agent actions
    test_env.env_method('render_all')
    test_env.close()
