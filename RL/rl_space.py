import numpy as np


class BaseStateEncoder:
    def on_reset(self, env): pass

    def encode(self, env, idx): raise NotImplementedError


class NOREncoder(BaseStateEncoder):
    def encode(self, env, idx): return 0


class SREncoder(BaseStateEncoder):
    """
    The agent's action in the previous step
    """

    def encode(self, env, idx): return int(env.actions[idx])


class NREncoder(BaseStateEncoder):
    """
    Number of cooperators among neighbors of the agent in the previous step
    """

    def encode(self, env, idx):
        neigh = env.get_neighbors(idx)
        return int(sum(1 for j in neigh if env.actions[j] == 0))


class SNREncoder(BaseStateEncoder):
    """
    Number of cooperators among agents and neighbors in the previous step
        0: C
        1: D
    """

    def encode(self, env, idx):
        neigh = env.get_neighbors(idx)
        coop_count = sum(1 for j in neigh if env.actions[j] == 0)
        if env.actions[idx] == 0:
            coop_count += 1
        return int(coop_count)


class NPEncoder(BaseStateEncoder):
    """
    Proportion of cooperators among neighbors of agents in the previous step
        0: [0.5, 1]
        1: [0, 0.5)
    """

    def encode(self, env, idx, **kwargs):
        neighbors = env.get_neighbors(idx)
        coop_ratio = sum(1 for j in neighbors if env.actions[j] == 0) / len(neighbors)
        return 0 if coop_ratio >= 0.5 else 1


class SNPEncoder(BaseStateEncoder):
    """
    The combination of the agent's previous action and the proportion of cooperators among neighbors in the previous step
        0: (C, [0.5, 1])
        1: (C, [0, 0.5))
        2: (D, [0.5, 1])
        3: (D, [0, 0.5))
    """

    def encode(self, env, idx):
        self_action = int(env.actions[idx])
        neighbors = env.get_neighbors(idx)
        coop_ratio = sum(1 for j in neighbors if env.actions[j] == 0) / len(neighbors)
        local_coop_level = coop_ratio
        env_type = 1 if local_coop_level < 0.5 else 0
        state = self_action * 2 + env_type

        return state


class AMEncoder(BaseStateEncoder):
    """
    Compared to the number of defectors (n_D) and cooperators (n_C) among neighbors of agent in the previous step
    0 表示 nC > nD
    1 表示 nC = nD
    2 表示 nC < nD
    """

    def encode(self, env, idx):
        neigh = env.get_neighbors(idx)
        nC = sum(1 for j in neigh if env.actions[j] == 0)
        nD = sum(1 for j in neigh if env.actions[j] == 1)
        if nC > nD:
            return 0
        elif nC == nD:
            return 1
        else:
            return 2


class SAMEncoder(BaseStateEncoder):
    """
    Compared to the number of defectors (n_D) and cooperators (n_C$) among the agent itself and its neighbors in the previous step
    0 表示 nC > nD
    1 表示 nC <= nD
    """

    def encode(self, env, idx):
        neigh = env.get_neighbors(idx)
        nC = sum(1 for j in neigh if env.actions[j] == 0)
        nD = sum(1 for j in neigh if env.actions[j] == 1)
        if env.actions[idx] == 0:
            nC += 1
        else:
            nD += 1
        if nC > nD:
            return 0
        else:
            return 1


class ActionSpace:
    def __init__(self):
        self.actions = [0, 1]

    def sample_action(self):
        return np.random.choice(self.actions)

    def n_actions(self):
        return len(self.actions)
