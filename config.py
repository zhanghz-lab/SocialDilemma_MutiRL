L = 100  # network size
N = L * L
total_steps = 10000  # total MCS
avg_steps = 100  # average cooperation rate per sampling step
average = 20  # independent simulation

b = 0.06  # PD temptation parameter
T = 1 + b
S = -b
R = 1
P = 0
r = 4  # PGG synergy factor
c = 1  # PGG cost

# Reinforcement learning hyperparameters
alpha = 0.1  # learning rate
gamma = 0.9  # discount factor
epsilon = 0.02  # epsilon
tau = 1  # Bolzmann temperature Factor

random_seed = 42
network_type = "lattice"  # network type, can be replaced with BA, WS
encoder_type = "NP"  # state representation, can be replaced with others
rl_type = "qlearning"  # reinforcement learning algorithms, can be replaced with SARSA.
exploration_type = "boltzmann"  # explore mechanisms, can be replaced with boltzmann
game_type = "PDG"  # game model, can be replaced with PGG
