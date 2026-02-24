from config import *


def PDG_payoff(env, idx, b):
    str = env.actions[idx]
    payoff = 0.0

    for nb in env.get_neighbors(idx):
        opp_str = env.actions[nb]
        if str == 0 and opp_str == 0:
            payoff += R
        elif str == 1 and opp_str == 0:
            payoff += 1 + b
        elif str == 0 and opp_str == 1:
            payoff += -b
        else:
            payoff += P
    return payoff


def PGG_payoff(env, idx):
    group = env.get_neighbors(idx) + [idx]
    contrib = sum(1 for p in group if env.strategies[p] == 0) * c
    total = contrib * r
    payoff = total / len(group)
    if env.action[idx] == 0:
        payoff -= c
    return payoff


def reward(env, idx, game_type, b):
    if game_type == "PDG":
        return PDG_payoff(env, idx, b)
    elif game_type == "PGG":
        return PGG_payoff(env, idx)
