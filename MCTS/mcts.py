import numpy as np
import gymnasium as gym
import itertools
from copy import deepcopy
from utils import action_arrow, opponent_random, render_rgb

class Node():
    """
    Node class for the MCTS tree.
    """
    def __init__(self, state, action, env, mcts, action_list, visits=0, rewards=0, parent=None):
        self.state = state
        self.action = action
        self.action_list = action_list
        self.children = []
        self.visits = visits
        self.sum_rewards = rewards
        self.parent = parent
        # self.env = env
        self.mcts = mcts

        # Env specific
        if self.mcts.env.spec.id == 'tictactoe-v0':
            self.unexplored_actions = self.mcts.env.get_valid_moves()
        elif self.mcts.env.spec.id == 'GridWorld-v0':
            self.unexplored_actions = action_list
        self.unexplored_actions = np.array(self.unexplored_actions)
    
    def add_child(self, child):
        child = child
        self.children.append(child)
    
    def get_value(self):
        return self.sum_rewards / self.visits if self.visits > 0 else 0
    
    def uct_value(self, parent_visits, c=1.4):
        # c is a constant that controls the degree of exploration

        # If the node or parent has not been visited, return infinity
        if self.visits == 0 or parent_visits == 0:
            return np.inf
        # Otherwise, return the standard UCT value
        else:
            return self.get_value() + c * np.sqrt(np.log(parent_visits) / self.visits)
    
    def selection(self, c=1.4):
        # check if the node has unexplored actions
        if len(self.unexplored_actions) > 0:
            return self
        # if not, select the child with the highest UCT value
        else:
            # If the node has no children, return itself
            if len(self.children) == 0:
                return self
            uct_values = [child.uct_value(self.visits, c) for child in self.children]
            max_child = self.children[np.argmax(uct_values)]
            return max_child.selection()
        
    def expansion(self):
        if len(self.unexplored_actions) == 0:
            return self
        # Randomly choose an unexplored action
        expand_index = np.random.choice(len(self.unexplored_actions))
        expand_action = self.unexplored_actions[expand_index]
        if self.mcts.env.spec.id == 'tictactoe-v0':
            expand_action = (0, expand_action)
        obs, reward, terminated, truncated, info = self.mcts.env.step(expand_action)
        if self.mcts.env.spec.id == 'TicTacToe-v0':
            if 'invalid' in info['info']:
                raise ValueError('Invalid action')
        new_node = Node(state=obs, action=expand_action, mcts=self.mcts, env=self.mcts.env, action_list=self.action_list, parent=self)
        self.children.append(new_node)
        if self.mcts.env.spec.id == 'tictactoe-v0':
            expand_action = expand_action[1]
        self.unexplored_actions = np.delete(self.unexplored_actions,np.where(self.unexplored_actions==expand_action))
        return new_node

    def rollout(self):
        if self.mcts.env.spec.id == 'GridWorld-v0':
            return self.rollout_gridworld()
        elif self.mcts.env.spec.id == 'tictactoe-v0':
            return self.rollout_tictactoe()
        else:
            raise NotImplementedError
        
    def rollout_gridworld(self):
        terminated = False
        total_reward = 0
        # Limit the rollout to k=4*size steps (as GridWorld is potentially infinite)
        k = 2 * self.mcts.env.size
        for _ in range(k):
            # Random rollout policy
            action = np.random.choice(self.mcts.env.action_space.n)
            obs, reward, terminated, truncated, info = self.mcts.env.step(action)
            total_reward += reward
            if terminated:
                break
        return total_reward

    def rollout_tictactoe(self):
        valid_actions = self.mcts.env.get_valid_moves()
        if len(valid_actions) == 0:
            return 0
        terminated = False
        total_reward = 0
        i=0
        # iterator to switch between players
        next_player = itertools.cycle([0, 1]).__next__
        while not terminated:
            player = next_player()
            # Random rollout policy of valid moves
            valid_actions = self.mcts.env.get_valid_moves()
            action = (player, np.random.choice(valid_actions))
            obs, reward, terminated, truncated, info = self.mcts.env.step(action)
            total_reward += reward
            i+=1
            if i>8:
                print('Rollout error')
                break
        return total_reward
    
    def backup(self, reward, discount):
        self.visits += 1
        self.sum_rewards += reward
        if self.parent:
            self.parent.backup(reward*discount, discount)

    def print_tree(self, depth=0, print_depth=0):
        if depth > print_depth:
            return
        # GridWorld specific
        if self.mcts.env.spec.id == 'GridWorld-v0':
            if depth == 0:
                print(f'Target: {self.state["target"]}')
            print(f'{"   " * depth} {action_arrow(self.action)} agent: {self.state["agent"]},',
                    f'visits: {self.visits}, value: {round(self.get_value(),2)}')
        # TicTacToe specific
        elif self.mcts.env.spec.id == 'tictactoe-v0':
            # if depth == 0:
            #     print(f'{"   " * depth} State: {self.state},',
            #         f'visits: {self.visits}, value: {round(self.get_value(),2)}')
            # else:
            print(f'{"   " * depth} Action: {self.action}, visits: {self.visits}, ',
                    f'value: {round(self.get_value(),2)}, State: {self.state},')
        for child in self.children:
            child.print_tree(depth + 1, print_depth)

    def _remove_action_nodes(self, actions):
        # get only the move part of the tuple
        moves = [action[1] for action in actions]
        for child in self.children:
            if child.action[1] in moves:
                self.children.remove(child)
            else:
                child._remove_action_nodes(actions)


class MCTS():
    """
    MCTS main class.
    """
    def __init__(self, env, state, render=True, iter_budget=100, discount=0.9, c=1.4):
        self.env = env
        self.action_list = self._get_action_list()
        self.root = Node(state=state, action=None, env=self.env, mcts=self, action_list=self.action_list)
        self.iter_budget = iter_budget
        self.discount = discount
        self.render = render
        self.c = c # exploration constant
    
    def best_action(self):
        """
        Select the action with the highest visit count. 
        For a tie, select the action with the highest value.
        """
        best_action = None
        max_visits = -1
        best_value = -float('inf')
        children = self.root.children

        for child in children:
            if child.visits > max_visits or (child.visits == max_visits and child.get_value() > best_value):
                best_action = child.action
                max_visits = child.visits
                best_value = child.get_value()
        
        return best_action


    def mcts_find_action(self, start_env=None, print_depth=0):
        for i in range(self.iter_budget):
            self.run_iteration(start_env)
            print(f'MCTS Iteration {i+1}/{self.iter_budget}', end='\r')
        print()
        # Choose the action with the highest visit count (+value tiebreaker) 
        action = self.best_action()
        
        if print_depth > 0:
            self.root.print_tree(0, print_depth)
        return action

    def run_iteration(self, start_env):
        '''Performs one iteration of the four steps of MCTS'''
        # Reset environment to the initial state
        self.env = deepcopy(start_env)
        # Selection
        selected_leaf = self.root.selection(self.c)
        # Expansion
        expand_node = selected_leaf.expansion()
        # Simulation
        reward = expand_node.rollout()
        # Backup
        expand_node.backup(reward, self.discount)

    def _get_action_list(self):
        # Assert that the action space is discrete
        # assert isinstance(self.env.action_space, gym.spaces.Discrete)
        if isinstance(self.env.action_space, gym.spaces.Discrete):
            return np.arange(self.env.action_space.n)
        elif isinstance(self.env.action_space, gym.spaces.MultiDiscrete):
            return [(0,n) for n in range(self.env.action_space.nvec[1])]
        
    def get_next_mcts(self, actions):
        # actions contains the agent's action (and if applicable the opponent's action)
        # Get the next MCTS object by selecting the child node with the given action
        self.root = [child for child in self.root.children if child.action == actions[0]][0]
        self.root.parent = None

        if self.env.spec.id == 'tictactoe-v0':
            # Remove all nodes with the selected action from the the tree as it is not valid anymore
            self.root._remove_action_nodes(actions)
        return self
    

def run_ttt_episode(env, obs, iter_budget=1000, print_depth=0, render=True, c=1.4):
    """
    Runs one episode of TicTacToe with MCTS as the agent.
    """
    if render:
        print('Agent (player 1): O \nRandom opponent (player 2): X\n')
    terminated = False
    total_reward = 0
    
    # Copy the env for MCTS simulations
    mcts_env = deepcopy(env)
    mcts = MCTS(mcts_env, obs, render=render, iter_budget=iter_budget, discount=1, c=c)

    # Other option: 
    while not terminated:
        # Agent step
        start_env = deepcopy(env)
        action = mcts.mcts_find_action(start_env=start_env, print_depth=print_depth)
        obs, reward, terminated, truncated, info = env.step(action)
        if render:
            print(f'Selected action: {action} Info: {info}')
        total_reward += reward
        if render:
            env.render()
        if terminated:
            print('Terminated', info)
            break
        
        # Opponent step
        action_opp = opponent_random(env, player=1)
        obs, _ , terminated, truncated, info = env.step(action_opp)
        if render:
            print(f'Opponent action: {action_opp} Info: {info}')
            env.render()

        if terminated:
            print('Terminated:', info)
            break

        # Set the root node to the child node corresponding to the action
        mcts = mcts.get_next_mcts([action, action_opp])
    
    return total_reward

def run_Gridworld_episode(env, obs, iter_budget=500, print_depth=0, ipynb=False, render=True, c=1.4):
    """
    Runs one episode of GridWorld with MCTS as the agent.
    """
    terminated = False
    total_reward = 0
    
    # Copy the env for MCTS simulations
    mcts_env = deepcopy(env)
    mcts = MCTS(mcts_env, obs, iter_budget=iter_budget, discount=0.9, c=c)

    # Limit number of steps to k=2*size as GridWorld is potentially infinite
    k = 2 * env.size
    for i in range(k):
        start_env = deepcopy(env)
        action = mcts.mcts_find_action(start_env=start_env, print_depth=print_depth)
        if render:
            print(f'Selected action: {action_arrow(action)}')
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if render:
            print(f'Obs: {obs}')
            render_rgb(env.render(), ipynb=ipynb)
        if terminated:
            print('Terminated')
            return total_reward
        # Set the root node to the child node corresponding to the action for the next MCTS
        mcts = mcts.get_next_mcts([action])

    print(f'Terminal state not reached within k=2*size={k} steps')
    return total_reward