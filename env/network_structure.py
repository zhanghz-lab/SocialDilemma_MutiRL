import numpy as np
import networkx as nx

class NetworkEnv:
    def __init__(self, network_type, n_actions, N=10000, L=100, k=4, p=0.1, seed=42):
        rng = np.random.default_rng(seed)
        self.rng = rng
        self.N = N
        self.L = L
        self.n_actions = n_actions
        self.network_type = network_type

        if network_type == "lattice":
            self.G = self._build_lattice(L)
        elif network_type == "WS":
            self.G = nx.watts_strogatz_graph(N, k, p, seed=seed)
        elif network_type == "BA":
            self.G = nx.barabasi_albert_graph(N, m=2, seed=seed)
        else:
            raise ValueError(f"Unknown network type: {network_type}")

        self.actions = rng.integers(low=0, high=2, size=N, dtype=int)  # 0=C, 1=D


    def _build_lattice(self, L):
        G = nx.grid_2d_graph(L, L, periodic=True)
        mapping = {node: idx for idx, node in enumerate(G.nodes())}
        return nx.relabel_nodes(G, mapping)

    def get_neighbors(self, idx):
        return list(self.G.neighbors(idx))

    def random_agent(self):
        return np.random.randint(0, self.N)

    def step(self, idx, action):
        self.actions[idx] = int(action)
