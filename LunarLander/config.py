import gymnasium

def make_env():
    env = gymnasium.make('LunarLander-v3')
    return env

def reward_shape(s, a, r, s_prime, term, trunc):
    return r