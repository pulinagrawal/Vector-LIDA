import os
from frozen_lake import FrozenLakeEnvironment, GymAdapter

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

core = FrozenLakeEnvironment(render_mode=None)
env = GymAdapter(core)

core = FrozenLakeEnvironment(render_mode=None)
eval_env = GymAdapter(core)
n_training_envs = 1
n_eval_envs = 5
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=1000)

vec_env = model.get_env()
obs = vec_env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    # VecEnv resets automatically
    # if done:
    #   obs = env.reset()

env.close()

mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=20, deterministic=True)
print(mean_reward, std_reward)