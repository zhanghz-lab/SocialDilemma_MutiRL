import numpy as np


class BaseStateEncoder:
    def on_reset(self, env): pass

    def encode(self, env, idx): raise NotImplementedError


class S8Encoder(BaseStateEncoder):
    def encode(self, env, idx): return 0


class S1Encoder(BaseStateEncoder):
    """
    The agent's action in the previous step
    """

    def encode(self, env, idx): return int(env.actions[idx])


class S2Encoder(BaseStateEncoder):
    """
    Number of cooperators among neighbors of the agent in the previous step
    """

    def encode(self, env, idx):
        neigh = env.get_neighbors(idx)
        return int(sum(1 for j in neigh if env.actions[j] == 0))


class S3Encoder(BaseStateEncoder):
    """
    Combination of the agent’s own action and the number of cooperative neighbors
        0: (C,0)
        1: (C,1)
        2: (C,2)
        3: (C,3)
        4: (C,4)
        5: (D,0)
        6: (D,1)
        7: (D,2)
        8: (D,3)
        9: (D,4)
    """

    def encode(self, env, idx, **kwargs):
        self_action = int(env.actions[idx])
        neighbors = env.get_neighbors(idx)
        coop_count = sum(1 for j in neighbors if env.actions[j] == 0)
        state = self_action * 5 + coop_count
        return state


class S4Encoder(BaseStateEncoder):
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


class S5Encoder(BaseStateEncoder):
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


class S6Encoder(BaseStateEncoder):
    """
    Combination of the agent’s own action and the relative majority relation between cooperators and defectors among its neighbor
    0: (C, nc < nd)
    1: (C, nc = nd)
    2: (C, nc > nd)
    3: (D, nc < nd)
    4: (D, nc = nd)
    5: (D, nc > nd)
    """

    def encode(self, env, idx, **kwargs):
        self_action = int(env.actions[idx])
        neigh = env.get_neighbors(idx)
        nC = sum(1 for j in neigh if env.actions[j] == 0)
        nD = sum(1 for j in neigh if env.actions[j] == 1)
        if nC < nD:
            compare = 0
        elif nC == nD:
            compare = 1
        else:
            compare = 2
        state = self_action * 3 + compare
        return state


class S7Encoder(BaseStateEncoder):
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
