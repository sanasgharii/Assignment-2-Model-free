import random
import numpy as np

class QLearningAgent(object):

    def __init__(self, n_actions, n_states, epsilon, alpha=0.1, gamma=0.99):
        self.n_actions = n_actions
        self.n_states = n_states
        self.epsilon = epsilon
        self.alpha = alpha 
        self.gamma = gamma
        self.Q = np.zeros((n_states, n_actions))  


    def select_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(range(self.n_actions))  
        else:
            return np.argmax(self.Q[state])  

    def update(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.Q[next_state])
        td_target = reward + self.gamma * self.Q[next_state][best_next_action]
        td_delta = td_target - self.Q[state][action]
        self.Q[state][action] += self.alpha * td_delta

class SARSAAgent(object):
    def __init__(self, n_actions, n_states, epsilon, alpha=0.1, gamma=0.99):
        self.n_actions = n_actions
        self.n_states = n_states
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.Q = np.zeros((n_states, n_actions))  

    def select_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(range(self.n_actions))  
        else:
            return np.argmax(self.Q[state])

    def update(self, state, action, reward, next_state, next_action):
        if next_action is not None:  
            next_action = int(next_action)
        else:
            td_target = reward 
        if next_action is not None:
            td_target = reward + self.gamma * self.Q[next_state, next_action]
        
        td_delta = td_target - self.Q[state, action]
        self.Q[state, action] += self.alpha * td_delta



class ExpectedSARSAAgent:
    def __init__(self, n_actions, n_states, epsilon, alpha=0.1, gamma=0.99):
        self.n_actions = n_actions
        self.n_states = n_states
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.Q = np.zeros((n_states, n_actions))

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.n_actions)
        else:
            return np.argmax(self.Q[state])

    def update(self, state, action, reward, next_state):
        current = self.Q[state, action]
        policy_s = np.ones(self.n_actions) * self.epsilon / self.n_actions  
        policy_s[np.argmax(self.Q[next_state])] += (1.0 - self.epsilon)  
        expected_q = np.dot(self.Q[next_state], policy_s)  
        td_target = reward + self.gamma * expected_q
        td_delta = td_target - current
        self.Q[state, action] += self.alpha * td_delta