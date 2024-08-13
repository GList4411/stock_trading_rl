import gym
from gym import spaces
import numpy as np
import alpaca_trade_api as tradeapi
import pandas as pd

class AlpacaTradingEnv(gym.Env):
    def __init__(self, api_key, api_secret, base_url, symbol, lookback_window=50):
        super(AlpacaTradingEnv, self).__init__()
        self.api = tradeapi.REST(api_key, api_secret, base_url, api_version='v2')
        self.symbol = symbol
        self.lookback_window = lookback_window

        self.current_step = 0
        self.cash = 10000  # Starting with $10,000
        self.shares = 0
        self.done = False

        # Fetch initial stock data
        self.stock_data = self._get_stock_data()

        # Define action and observation space
        self.action_space = spaces.Discrete(3)  # 0: Hold, 1: Buy, 2: Sell
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.lookback_window, len(self.stock_data.columns)), dtype=np.float32)

    def _get_stock_data(self):
        # Fetch historical data from Alpaca
        barset = self.api.get_barset(self.symbol, 'day', limit=self.lookback_window)
        bars = barset[self.symbol]
        data = {
            'Open': [bar.o for bar in bars],
            'High': [bar.h for bar in bars],
            'Low': [bar.l for bar in bars],
            'Close': [bar.c for bar in bars],
            'Volume': [bar.v for bar in bars],
        }
        return pd.DataFrame(data)

    def reset(self):
        self.current_step = 0
        self.cash = 10000
        self.shares = 0
        self.done = False
        self.stock_data = self._get_stock_data()  # Refresh data
        return self.stock_data.values

    def step(self, action):
        current_price = self.stock_data.iloc[self.current_step]['Close']
        reward = 0

        if action == 1:  # Buy
            if self.cash > current_price:
                self.shares += 1
                self.cash -= current_price
                reward = 0

        elif action == 2:  # Sell
            if self.shares > 0:
                self.shares -= 1
                self.cash += current_price
                reward = current_price

        self.current_step += 1

        if self.current_step >= len(self.stock_data) - 1:
            self.done = True
            reward += self.cash + self.shares * current_price

        obs = self.stock_data.values
        return obs, reward, self.done, {}

    def render(self, mode='human'):
        print(f'Step: {self.current_step}')
        print(f'Price: {self.stock_data.iloc[self.current_step]["Close"]}')
        print(f'Shares: {self.shares}')
        print(f'Cash: {self.cash}')

