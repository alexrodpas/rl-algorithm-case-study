import gymnasium as gym
import gym_examples
from mcts import run_ttt_episode, run_Gridworld_episode
import numpy as np
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

ttt_env = gym.make('tictactoe-v0')
ttt_obs = ttt_env.reset()

gw_env = gym.make('GridWorld-v0', render_mode='rgb_array', size=4)
gw_obs, info = gw_env.reset()

# Hyperparameter values to test
nr_iterations = [10, 100, 500, 1000, 2500]
c_vals = [0, 1, 1.2, 1.4, 1.6, 2, 5]

def generate_plot(env, obs, param_vals, type):
    if type == "iterations":
        average_count = 15
    elif type == "c":
        average_count = 10
    rewards = np.zeros((len(param_vals), average_count))

    for param in param_vals:
        if type == "iterations":
            print("\nIteration budget: ", param)
            iter = param
            c = 1.4
        elif type == "c":
            print("\nExploration c: ", param)
            iter = 1000
            c = param
        for i in range(average_count):
            ttt_env.reset()
            if env.spec.id == "GridWorld-v0":
                reward = run_Gridworld_episode(gw_env, gw_obs, iter_budget=iter, c=c, render=False)
            elif env.spec.id == "tictactoe-v0":
                reward = run_ttt_episode(ttt_env, obs, iter_budget=iter, c=c, render=False)
            rewards[param_vals.index(param)][i] = reward
        print(param, rewards[param_vals.index(param)])
    
    # Calculate mean rewards
    mean_rewards = np.mean(rewards, axis=1)
    print("Mean rewards: ")
    for param in param_vals:
        print(param, ": ", mean_rewards[param_vals.index(param)])

    # Plot results
    plt.figure()
    plt.plot(param_vals, mean_rewards)
    plt.ylabel("Mean reward")
    if type == "iterations":
        plt.xlabel("Iteration budget")
        plt.title("Mean reward vs iteration budget")
    elif type == "c":
        plt.xlabel("Exploration c")
        plt.title("Mean reward vs exploration c")
    # save plot to file
    env_type = env.spec.id
    filename = f"mcts_{env_type}_{type}.png"
    plt.savefig(f"mcts/plots/{filename}")

generate_plot(ttt_env, ttt_obs, nr_iterations, "iterations")
generate_plot(ttt_env, ttt_obs, c_vals, "c")
generate_plot(gw_env, gw_obs, nr_iterations, "iterations")
generate_plot(gw_env, gw_obs, c_vals, "c")