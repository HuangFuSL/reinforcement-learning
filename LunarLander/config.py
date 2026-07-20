import gymnasium
import torch
import wandb
import os

def save_checkpoint(model: torch.nn.Module, name: str):
    ckpt_dir = os.path.join(os.path.dirname(__file__), 'checkpoints')
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    torch.save(
        model.state_dict(),
        os.path.join(ckpt_dir, f'{name}.pth')
    )

def init_wandb(model_name: str):
    wandb.init(
        project='Reinforcement-Learning',
        group='LunarLander-Discrete',
        name=model_name,
    )


def make_env():
    env = gymnasium.make('LunarLander-v3')
    return env

def make_demo_env(video_dir: str | None = None):
    if video_dir is None:
        video_dir = f'videos/{wandb.run.id}'
    if not os.path.exists(video_dir):
        os.makedirs(video_dir)
    env = gymnasium.make('LunarLander-v3', render_mode='rgb_array')
    env = gymnasium.wrappers.RecordVideo(
        env,
        video_folder=video_dir,
        episode_trigger=lambda x: True,
        name_prefix='demo',
    )
    return env

def reward_shape(s, a, r, s_prime, term, trunc):
    return r
