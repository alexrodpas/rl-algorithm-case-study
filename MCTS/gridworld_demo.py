import gym
import gym_examples
from mcts import run_Gridworld_episode
from utils import render_rgb

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

def gridworld_demo(ipynb=False):
    render_mode='rgb_array'
    env = gym.make('GridWorld-v0', render_mode=render_mode, size=4)
    obs, info = env.reset()
    if render_mode == 'rgb_array':
        render_rgb(env.render(), ipynb=ipynb)

    # Run one episode
    total_reward = run_Gridworld_episode(env, obs, print_depth=1, ipynb=ipynb)
    print(f'Total reward: {total_reward}')

gridworld_demo()