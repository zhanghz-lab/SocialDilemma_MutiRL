import numpy as np
import random
from collections import defaultdict

class TabularQ:

    def __init__(self, N, n_actions):
        self.N = N
        self.n_actions = n_actions
        self.Q = [defaultdict(self._zeros) for _ in range(N)]

    def _zeros(self):
        return np.zeros(self.n_actions, dtype=float)

    def values(self, idx, s_key):
        return self.Q[idx][s_key]

    def best_action(self, idx, s_key):
        return int(np.argmax(self.Q[idx][s_key]))


class QLearning:
    def __init__(self, N, state_space, action_space, alpha, gamma, epsilon, tau, exploration_type):
        self.state_space = state_space
        self.action_space = action_space
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.table = TabularQ(N, action_space.n_actions())
        self.exploration_type = exploration_type
        self.tau = tau

    def softmax(self, x):
        z = x - np.max(x)
        numerator = np.exp(z)
        denominator = np.sum(numerator)
        softmax = numerator / denominator
        return softmax

    def choose_action(self, s, idx, exploration_type):
        if exploration_type == 'boltzmann':
            policyS = self.softmax(self.tau * self.table.values(idx, s))
            action = np.random.choice(len(policyS), p=policyS)
            return action
        elif exploration_type == 'epsilongreedy':
            if random.random() < self.epsilon or (not np.any(self.table.values(idx, s))):
                return self.action_space.sample_action()
            return self.table.best_action(idx, s)

    def update(self, s, env, idx, reward, encoder_type, exploration_type):
        action = env.actions[idx]
        Qsa = self.table.values(idx, s)
        env.step(idx, action)
        if encoder_type == "NOR":
            Qsa[action] += self.alpha * (reward - Qsa[action])
        else:
            s_next = self.state_space.encode(env, idx)
            max_next = np.max(self.table.values(idx, s_next))
            Qsa[action] += self.alpha * (reward + self.gamma * max_next - Qsa[action])



class SARSA:
    def __init__(self, N, state_space, action_space, alpha, gamma, epsilon, tau, exploration_type):
        self.state_space = state_space
        self.action_space = action_space
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.table = TabularQ(N, action_space.n_actions())
        self.exploration_type = exploration_type
        self.tau = tau

    def softmax(self, x):
        z = x - np.max(x)
        numerator = np.exp(z)
        denominator = np.sum(numerator)
        softmax = numerator / denominator
        return softmax

    def choose_action(self, s, idx, exploration_type):
        if exploration_type == 'boltzmann':
            policyS = self.softmax(self.tau * self.table.values(idx, s))
            action = np.random.choice(len(policyS), p=policyS)
            return action
        elif exploration_type == 'epsilongreedy':
            if random.random() < self.epsilon or (not np.any(self.table.values(idx, s))):
                return self.action_space.sample_action()
            return self.table.best_action(idx, s)

    def update(self, s, env, idx, reward, encoder_type, exploration_type):
        action = env.actions[idx]
        Qsa = self.table.values(idx, s)
        env.step(idx, action)
        s_next = self.state_space.encode(env, idx)
        a_next = self.choose_action(env, idx, exploration_type)
        Qsa_next = self.table.values(idx, s_next)
        Qsa[action] += self.alpha * (reward + self.gamma * Qsa_next[a_next] - Qsa[action])
