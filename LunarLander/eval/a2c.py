import asyncio
import dataclasses
import torch
import os
import time

from utils.ctorch.rl.interface import run_episode
from utils.ctorch.device import get_best_device
from utils.parser import auto_cli

from ..models.a2c import Actor
from ..config import make_env, reward_shape, make_demo_env

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@auto_cli
@dataclasses.dataclass
class EvalConfig:
    checkpoint: str = 'checkpoints/lunarlander_a2c.pt'
    episodes: int = 10
    max_steps: int = 1000
    demo: bool = False
    video_dir: str = 'videos/eval'


async def main():
    config = EvalConfig.parse_args()

    # --- Model setup ---
    device = get_best_device()
    model = Actor(gamma=0.99)
    checkpoint = torch.load(config.checkpoint, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    model.to(device)

    print(f'[Eval] Loaded checkpoint: {config.checkpoint}')
    print(f'[Eval] Model device: {model.device}')

    # --- Run evaluation episodes ---
    env = make_env()
    rewards = []
    lengths = []

    for i in range(config.episodes):
        result = run_episode(env, model, max_episode_steps=config.max_steps)
        r = result.total_reward
        l = len(result)
        rewards.append(r)
        lengths.append(l)
        print(f'[Eval] Episode {i + 1:3d}:  reward = {r:+.2f}   length = {l}')

    env.close()

    # --- Summary ---
    if rewards:
        print(f'\n{"─" * 50}')
        print(f'Summary over {len(rewards)} episodes:')
        print(f'  Avg reward:  {sum(rewards) / len(rewards):+.2f}')
        print(f'  Max reward:  {max(rewards):+.2f}')
        print(f'  Min reward:  {min(rewards):+.2f}')
        print(f'  Avg length:  {sum(lengths) / len(lengths):.1f}')
        print(f'{"─" * 50}')

    # --- Demo video ---
    if config.demo:
        ts = time.strftime('%Y%m%d_%H%M%S')
        video_dir = os.path.join(config.video_dir, ts)
        print(f'\n[Demo] Recording video to {video_dir} ...')
        demo_env = make_demo_env(video_dir=video_dir)
        result = run_episode(demo_env, model, config.max_steps)
        print(f'[Demo] Episode: reward = {result.total_reward:+.2f}   length = {len(result)}')
        demo_env.close()
        print(f'[Demo] Video saved to {video_dir}')


if __name__ == '__main__':
    asyncio.run(main())
