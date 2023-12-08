import numpy as np

true_v = [[4.0187, 4.5548, 5.1575, 5.8336, 6.4553],
          [4.3716, 5.0324, 5.8013, 6.6473, 7.3907],
          [3.8672, 4.3900, 0.0000, 7.5769, 8.4637],
          [3.4182, 3.8319, 0.0000, 8.5738, 9.6946],
          [2.9977, 2.9309, 6.0733, 9.6946, 0.0000]]

# System parameters
alpha_theta = 1e-1
alpha_w = 1e-1
gamma = 0.9
threshold = 1e-3
max_steps = 10000
max_iterations = 10000

# Environment descriptors
R = np.zeros((5, 5))
R[4, 4] = 10
R[4, 2] = -10

terminal_states = [(4, 4)]

obstacles = [(2, 2), (3, 2)]

def main():        
    model = REINFORCE_with_baseline(max_steps)
    model.optimize(alpha_theta, alpha_w, gamma, threshold)
        
class REINFORCE_with_baseline():
    def __init__(self, step_horizon):
        self.theta = np.random.rand(5, 5, 4)
        self.pi = self._define_policy()
        
        self.v = np.zeros((5, 5))
        for obstacle in obstacles:
            self.v[obstacle[0], obstacle[1]] = 0.0
        
        for terminal in terminal_states:
            self.v[terminal[0], terminal[1]] = 0.0
            
        self.step_horizon = step_horizon
        
    def _define_policy(self):
        return np.exp(self.theta)/np.sum(np.exp(self.theta), axis=2, keepdims=True)
    
    def _visualize_policy(self):
        arrows = ['↑', '→', '↓', '←']
        
        visualization = np.array([[' '] * self.pi.shape[1]] * self.pi.shape[0])
        
        for i in range(self.pi.shape[0]):
            for j in range(self.pi.shape[1]):
                visualization[i, j] = arrows[np.argmax(self.pi[i, j])]
        
        for obstacle in obstacles:
            visualization[obstacle[0], obstacle[1]] = ' '
        
        for terminal in terminal_states:
            visualization[terminal[0], terminal[1]] = ' '
            
        return visualization
    
    def optimize(self, alpha_theta, alpha_w, gamma, threshold):
        # Algorithm tracking variables
        done = False
        restart = False
        iterations = 0
        
        # Loops through algorithm until done
        while ((not done or restart) and iterations < max_iterations):
            iterations += 1                     # Tracks number of iterations
            if (iterations % 100 == 0):
                print(f"Policy after {iterations} iterations:\n{self._visualize_policy()}")
                
            states, actions, rewards, restart = self.generate_episode(exploratory=True)
            done = self.update_parameters(states, actions, rewards, alpha_w, alpha_theta, gamma, threshold)   # Evaluates the current policy
            self.pi = self._define_policy()     # Improves the policy
            
        # Prints results
        print(f"It took the algorithm {iterations} iterations to converge")
        print(f"Final policy:\n{self._visualize_policy()}")
        print(f"Final v:\n{self.v}")
        print(f"Max-Norm v difference: {np.linalg.norm(self.v - true_v, ord=np.inf)}")
        
    def select_action(self, i, j):                
        return np.random.choice([a for a in range(len(self.pi[i, j]))], p=self.pi[i, j])
    
    def generate_episode(self, exploratory):
        i = 0
        j = 0
        
        if (exploratory):
            # Generates d_0
            while (True):
                i = np.random.randint(5)
                j = np.random.randint(5)
            
                if ((i, j) not in obstacles and (i, j) not in terminal_states):
                    break
        
        steps = 0
        states_visited = []
        actions_taken = []
        rewards = []
        restart = False
        # Loops through v and pi for policy iteration
        while ((i, j) not in terminal_states):
            # Adding a timeout to avoid infinite looping
            if (steps >= self.step_horizon):
                restart = True
                break
            
            # Determines value and location of moving up
            if (i < 1) or ((i - 1, j) in obstacles):
                up_i = i
                up_j = j
            else:
                up_i = i - 1
                up_j = j
            
            # Determines value and location of moving right
            if (j >= self.v.shape[1] - 1) or ((i, j + 1) in obstacles):
                right_i = i
                right_j = j
            else:
                right_i = i
                right_j = j + 1
                
            # Determines value and location of moving down
            if (i >= self.v.shape[0] - 1) or ((i + 1, j) in obstacles):
                down_i = i
                down_j = j
            else:
                down_i = i + 1
                down_j = j
                
            # Determines value and location of moving left
            if (j < 1) or ((i, j - 1) in obstacles):
                left_i = i
                left_j = j
            else:
                left_i = i
                left_j = j - 1
            
            random_num = np.random.rand(1)
            next_positions = [[up_i, up_j], [right_i, right_j], [down_i, down_j], [left_i, left_j]]
            action = self.select_action(i, j)
            
            if (random_num < 0.9):
                if (random_num < 0.8):
                    next_position = next_positions[action]
                elif (random_num < 0.85):
                    next_position = next_positions[(action + 1) % 4]
                else:
                    next_position = next_positions[(action - 1) % 4]
            else:
                next_position = [i, j]
            
            next_i = next_position[0]
            next_j = next_position[1]
            reward = R[next_i, next_j]
            states_visited.append((i, j))
            actions_taken.append(action)
            rewards.append(reward)
            i = next_i
            j = next_j
            steps += 1
            
        return states_visited, actions_taken, rewards, restart
    
    def update_parameters(self, states, actions, rewards, alpha_w, alpha_theta, gamma, threshold):
        done = True
        
        for t in range(len(states)):
            G = np.dot(rewards[t:], [gamma**k for k in range(len(states) - t)])
            delta = G - self.v[states[t][0], states[t][1]]
            
            if (abs(alpha_w * delta) >= threshold or abs(alpha_theta * gamma**t * delta) >= threshold):
                done = False
            
            self.v[states[t][0], states[t][1]] += alpha_w * delta
            self.theta[states[t][0], states[t][1]] -= alpha_theta * gamma**t * delta * self.pi[states[t][0], states[t][1]]
            self.theta[states[t][0], states[t][1], actions[t]] += alpha_theta * gamma**t * delta
            
        return done
                
if __name__ == "__main__":
    main()