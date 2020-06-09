"""Training script for Neuro-symbolic Reinforcement Learning"""
import numpy as np
import gym

# Disable scientific printing
np.set_printoptions(threshold=10000, suppress=True, precision=5, linewidth=180)

env = gym.make('CartPole-v1')
print("Observation:")
print(env.observation_space)
print("Action:")
print(env.action_space)

for k in range(2):
  print("\n-------\n")
  observation: np.ndarray = env.reset()
  for t in range(100):
    # env.render()
    print(observation)
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    if done:
      print(f"Episode finished after {t+1} timesteps")
      print("\n-------\n")
      break
env.close()
