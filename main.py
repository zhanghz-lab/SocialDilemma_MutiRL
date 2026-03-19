from tqdm import trange
from RL.rl_algorithms import *
from RL.rl_space import *
from config import *
from env import game_model
from env.network_structure import NetworkEnv
from utils.plot_tools import *

if __name__ == '__main__':
    if encoder_type == "S8":
        state_space = NOREncoder()
    elif encoder_type == "SR":
        state_space = SREncoder()
    elif encoder_type == "NR":
        state_space = NREncoder()
    elif encoder_type == "SNR":
        state_space = SNREncoder()
    elif encoder_type == "AM":
        state_space = AMEncoder()
    elif encoder_type == "SAM":
        state_space = SAMEncoder()
    elif encoder_type == "NP":
        state_space = NPEncoder()
    elif encoder_type == "SNP":
        state_space =SNPEncoder()
    else:
        raise ValueError(f"unknown encoder_type: {encoder_type}")
    action_space = ActionSpace()#0=C, 1=D
    env = NetworkEnv(network_type, N=L * L, n_actions=action_space.n_actions(), L=L)
    state_space.on_reset(env)
    if rl_type == "qlearning":
        agent = QLearning(env.N, state_space, action_space, alpha, gamma, epsilon, tau, exploration_type)
    elif rl_type == "sarsa":
        agent = SARSA(env.N, state_space, action_space, alpha, gamma, epsilon, tau, exploration_type)
    dirName = 'result//result_%s_%s_%s_%s_%s_b=%s_avgrage=%s_N=%s' % (
        network_type, rl_type, game_type, encoder_type, exploration_type, b, average, N)
    print(dirName)
    if not os.path.exists(dirName):
        os.makedirs(dirName)

    for sim in trange(average):
        sim_dir = os.path.join(dirName, f"sim{sim:04d}")
        os.makedirs(sim_dir, exist_ok=True)

        cooperators_frac = np.zeros(total_steps)
        for step in trange(total_steps):
            cooperators_frac[step] = np.sum(env.actions == 0) / N
            for _ in range(N):
                idx = env.random_agent()
                s = state_space.encode(env, idx)
                if encoder_type == "NOR":
                    reward = game_model.reward(env, idx, game_type, b)
                    agent.update(s, env, idx, reward, encoder_type,exploration_type)
                    action = agent.choose_action(s, idx,exploration_type)
                    env.actions[idx] = action
                else:
                    idx = env.random_agent()
                    s = state_space.encode(env, idx)
                    action = agent.choose_action(s, idx,exploration_type)
                    env.actions[idx] = action
                    reward = game_model.reward(env, idx, game_type, b)
                    agent.update(s, env, idx, reward, encoder_type,exploration_type)

        np.savetxt(os.path.join(sim_dir, f"CountCooperators-sim{sim:04d}.csv"), cooperators_frac, fmt="%.6f",
                   delimiter=",")
        print(np.mean(cooperators_frac[-avg_steps:]))
        # visualisation
        plot_cooperators_ratio(cooperators_frac, sim_dir, sim, b)
