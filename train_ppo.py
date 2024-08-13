import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stock_trading_env import AlpacaTradingEnv #import custon environment

# Replace these with your Alpaca credentials
API_KEY = 'your_api_key'
API_SECRET = 'your_api_secret'
BASE_URL = 'https://paper-api.alpaca.markets'

# Trading symbol
SYMBOL = 'AAPL'

# Create the custom environment
env = DummyVecEnv([lambda: AlpacaTradingEnv(API_KEY, API_SECRET, BASE_URL, SYMBOL)])

# Initialize PPO
model = PPO("MlpPolicy", env, verbose=1)

# Train the model
model.learn(total_timesteps=10000)

# Save the trained model
model.save("ppo_stock_trading")
