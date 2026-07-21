'''
PPO training for LunarLander-v3 (discrete).

Usage::

    # Default hyperparameters
    python -m LunarLander.train.ppo

    # Full parameter specification (shown with defaults)
    ./taskq.sh python -m LunarLander.train.ppo \
        --model-gamma 0.99 \
        --model-epsilon 0.2 \
        --model-lambd 0.95 \
        --model-device auto \
        --optim-name Adam \
        --optim-policy-lr 1e-3 \
        --optim-value-lr 1e-3 \
        --optim-policy-clip-grad-norm 10.0 \
        --optim-value-clip-grad-norm 10.0 \
        --train-epoch 300 \
        --train-num-workers 8 \
        --train-n-trajs 16 \
        --train-n-iters 8 \
        --train-entropy-coef 5e-3

    # Boolean flags use --flag / --no-flag
    # Set optim-policy-clip-grad-norm / optim-value-clip-grad-norm to None (disable grad clipping):
    python -m LunarLander.train.ppo --optim-policy-clip-grad-norm None --optim-value-clip-grad-norm None
    # Load from YAML/JSON config file (CLI args take priority)
    python -m LunarLander.train.ppo --config configs/ppo.yaml
'''
import dataclasses
import os
import random
import torch
import tqdm
import wandb
import wandb.integration.gym

from utils.ctorch.rl.async_env import AsyncEnvPool
from utils.ctorch.device import get_best_device
from utils.ctorch.rl.interface import run_episode
from utils.parser import auto_cli, parse_all_args

from ..models.ppo import PPO, __model_name__
from ..config import make_env, reward_shape, make_demo_env, init_wandb

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
torch.set_num_threads(1)

def _resolve_device(s: str) -> torch.device:
    return torch.device(get_best_device() if s == 'auto' else s)

@auto_cli
@dataclasses.dataclass
class ModelConfig:
    gamma: float = 0.99
    epsilon: float = 0.2
    lambd: float = 0.95

    device: str = 'auto'

    def get(self):
        device = _resolve_device(self.device)
        model = PPO(gamma=self.gamma, epsilon=self.epsilon, lambda_=self.lambd)
        model.to(device)
        return model


@auto_cli
@dataclasses.dataclass
class OptimizerConfig:
    name: str = 'Adam'
    policy_lr: float = 1e-3
    value_lr: float = 1e-3
    policy_clip_grad_norm: float | None = 10.0
    value_clip_grad_norm: float | None = 10.0

    def get(self, model: PPO):
        optimizer_cls = getattr(torch.optim, self.name)
        policy_optimizer = optimizer_cls(
            model.policy_parameters(), lr=self.policy_lr
        )
        value_optimizer = optimizer_cls(
            model.value_parameters(), lr=self.value_lr
        )
        return policy_optimizer, value_optimizer


@auto_cli
@dataclasses.dataclass
class TrainConfig:
    epoch: int = 300
    num_workers: int = 8
    n_trajs: int = 16
    n_iters: int = 8
    entropy_coef: float = 5e-3


def main(
    model_config: ModelConfig,
    optim_config: OptimizerConfig,
    train_config: TrainConfig
):
    init_wandb(__model_name__)
    wandb.integration.gym.monitor()
    model = model_config.get()
    p_optim, v_optim = optim_config.get(model)

    pool = AsyncEnvPool(
        model, make_env,
        num_workers=train_config.num_workers,
        inference_device=model.device
    )

    with pool:
        for epoch in tqdm.trange(train_config.epoch):

            trajs = pool.run_episode(
                model, num_trajectories=train_config.n_trajs,
                reward_shape=reward_shape, output_device=model.device
            )
            indexes = list(range(len(trajs))) * train_config.n_iters
            random.shuffle(indexes)

            log = {}
            log['epoch'] = epoch

            rewards, lengths = [], []
            for traj in trajs:
                rewards.append(traj.total_reward)
                lengths.append(len(traj))
            log['train/reward'] = sum(rewards) / len(rewards)
            log['train/length'] = sum(lengths) / len(lengths)

            for idx in indexes:
                if log:
                    wandb.log(log)
                    log.clear()

                model.train()
                p_optim.zero_grad()
                v_optim.zero_grad()

                # Get batch and loss weight
                traj = trajs[idx]
                p_loss, v_loss, entropy, kl = model.loss(traj)
                total_loss = p_loss + v_loss - train_config.entropy_coef * entropy

                log = {
                    'train/policy_loss': p_loss.item(),
                    'train/value_loss': v_loss.item(),
                    'train/entropy': entropy.item(),
                    'train/kl': kl.item(),
                    'train/loss': total_loss.item(),
                }
                total_loss.backward()
                p_grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.policy_parameters(), optim_config.policy_clip_grad_norm
                ) if optim_config.policy_clip_grad_norm is not None else None
                v_grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.value_parameters(), optim_config.value_clip_grad_norm
                ) if optim_config.value_clip_grad_norm is not None else None
                if p_grad_norm is not None:
                    log['train/policy_grad_norm'] = p_grad_norm.item()
                if v_grad_norm is not None:
                    log['train/value_grad_norm'] = v_grad_norm.item()
                p_optim.step()
                v_optim.step()

            # --- Checkpoint ---
            if epoch % 10 == 0:
                if not os.path.exists('checkpoints'):
                    os.makedirs('checkpoints')
                torch.save(model.state_dict(), f'checkpoints/lunarlander_ppo_{epoch}.pt')

            # --- Demo ---
                demo_env = make_demo_env(video_dir=f'videos/{wandb.run.id}/epoch_{epoch:03d}')
                result = run_episode(demo_env, model, 1000)
                log['demo/reward'] = result.total_reward
                log['demo/length'] = len(result)
                del demo_env # AttributeError: 'RecordVideo' object has no attribute 'enabled'

            model.train()
            wandb.log(log)

    # --- Final demo & save ---
    demo_env = make_demo_env()
    run_episode(demo_env, model, 1000)
    model.to('cpu')
    torch.save(model.state_dict(), 'checkpoints/lunarlander_ppo.pt')


if __name__ == '__main__':
    configs = parse_all_args(
        model=ModelConfig,
        optim=OptimizerConfig,
        train=TrainConfig
    )
    main(
        model_config=configs['model'],
        optim_config=configs['optim'],
        train_config=configs['train'],
    )
