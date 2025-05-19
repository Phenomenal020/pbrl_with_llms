import os
from stable_baselines3 import PPO
    
    
models_dir = "models/standard_rl/ppo"
os.makedirs(models_dir, exist_ok=True)
    
# Optionally, load and evaluate any agent
model = PPO.load(f"{models_dir}/standard_reach-sponge_ppo_{22000}")
obs = env.reset()
episodes = 100
for _ in range(episodes):
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    print(rewards)
    if done:
        obs = env.reset()