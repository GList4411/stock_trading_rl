from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stock_trading_env import AlpacaTradingEnv

# Replace these with your Alpaca credentials
API_KEY = 'your_api_key'
API_SECRET = 'your_api_secret'
BASE_URL = 'https://paper-api.alpaca.markets'
SYMBOL = 'AAPL'

# Create the custom environment
env = DummyVecEnv([lambda: AlpacaTradingEnv(API_KEY, API_SECRET, BASE_URL, SYMBOL)])

# Load the trained model
model = PPO.load("ppo_stock_trading")

# Initialize the environment and variables
obs = env.reset()
done = False
total_reward = 0
total_steps = 0

# Run the model for a single episode
while not done:
    # Predict the action based on the current observation
    action, _states = model.predict(obs, deterministic=True)
    
    # Take the action and get the results
    obs, reward, done, info = env.step(action)
    
    # Update total rewards and steps
    total_reward += reward
    total_steps += 1
    
    # Optionally render the environment (if applicable)
    env.render()

print(f"Total reward: {total_reward}")
print(f"Total steps: {total_steps}")
