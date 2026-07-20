# LunarLander

This directory contains the implementation of RL models under [LunarLander-v3](https://gymnasium.farama.org/environments/box2d/lunar_lander/) discrete environment.

## MDP formulation:

- Observation space: 8-dimensional continuous
    ```
    Box([ -2.5 -2.5 -10. -10. -6.2831855 -10. -0. -0. ], [ 2.5 2.5 10. 10. 6.2831855 10. 1. 1. ], (8,), float32)
    ```
- Action space: 4 discrete actions
    - 0: do nothing
    - 1: fire left orientation engine
    - 2: fire main engine
    - 3: fire right orientation engine
- Reward: The reward is calculated as follows:
    - Reward of 100 for landing on the landing pad
    - Reward of 10 for each leg that makes contact with the ground
    - Reward of -0.3 for firing the main engine
    - Reward of -0.03 for firing the side engines
    - Reward of -100 for crashing
- Max episode length: 1000 timesteps

An episode is considered a solution if it scores at least 200 points.

## Models

- Rainbow DQN: [rainbow.py](models/rainbow.py) [statistics](https://wandb.ai/huangfusl/Reinforcement-Learning/runs/4iunar9l)
