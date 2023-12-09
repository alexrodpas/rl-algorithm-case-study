import gym
import gym_examples
from mcts import run_ttt_episode

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

ttt_env = gym.make('tictactoe-v0')
obs = ttt_env.reset()
ttt_env.render()
reward = run_ttt_episode(ttt_env, obs, iter_budget=1000, print_depth=1)
print("Reward:", reward)