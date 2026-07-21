'''
A2C training for LunarLander-v3 (discrete).

Usage::

    # Default hyperparameters
    python -m LunarLander.train.a2c

    # Full parameter specification (shown with defaults)
    ./taskq.sh python -m LunarLander.train.a2c \
        --model-gamma 0.99 \
        --model-lambd 0.95 \
        --model-device auto \
        --optim-name Adam \
        --optim-policy-lr 1e-4 \
        --optim-value-lr 3e-4 \
        --optim-policy-clip-grad-norm 1.0 \
        --optim-value-clip-grad-norm 5.0 \
        --optim-policy-weight-decay 1e-4 \
        --optim-value-weight-decay 1e-4 \
        --train-epoch 600 \
        --train-num-workers 8 \
        --train-n-trajs 8 \
        --train-n-iters 4 \
        --train-entropy-coef 1e-2

    # Boolean flags use --flag / --no-flag
    # Set optim-policy-clip-grad-norm / optim-value-clip-grad-norm to None (disable grad clipping):
    python -m LunarLander.train.a2c --optim-policy-clip-grad-norm None --optim-value-clip-grad-norm None
    # Load from YAML/JSON config file (CLI args take priority)
    python -m LunarLander.train.a2c --config configs/a2c.yaml
'''
import dataclasses
import os
import torch
import tqdm
import wandb
import wandb.integration.gym

from utils.ctorch.rl.async_env import AsyncEnvPool
from utils.ctorch.device import get_best_device
from utils.ctorch.rl.interface import run_episode
from utils.parser import auto_cli, parse_all_args

from ..models.a2c import Actor, __model_name__
from ..config import make_env, make_demo_env, init_wandb

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
torch.set_num_threads(1)

def _resolve_device(s: str) -> torch.device:
    return torch.device(get_best_device() if s == 'auto' else s)

def reward_shape(s, a, r, s_prime, term, trunc):
    return r / 100

@auto_cli
@dataclasses.dataclass
class ModelConfig:
    gamma: float = 0.99
    lambd: float = 0.95

    device: str = 'auto'

    def get(self):
        device = _resolve_device(self.device)
        model = Actor(gamma=self.gamma, lambda_=self.lambd)
        model.to(device)
        return model


@auto_cli
@dataclasses.dataclass
class OptimizerConfig:
    name: str = 'Adam'
    policy_lr: float = 1e-4
    value_lr: float = 3e-4
    policy_clip_grad_norm: float | None = 1.0
    value_clip_grad_norm: float | None = 5.0
    policy_weight_decay: float = 1e-4
    value_weight_decay: float = 1e-4

    def get(self, model: Actor):
        optimizer_cls = getattr(torch.optim, self.name)
        policy_optimizer = optimizer_cls(
            model.policy_parameters(), lr=self.policy_lr,
            weight_decay=self.policy_weight_decay
        )
        value_optimizer = optimizer_cls(
            model.value_parameters(), lr=self.value_lr,
            weight_decay=self.value_weight_decay
        )
        return policy_optimizer, value_optimizer


@auto_cli
@dataclasses.dataclass
class TrainConfig:
    epoch: int = 600
    num_workers: int = 8
    n_trajs: int = 8
    n_iters: int = 4
    entropy_coef: float = 1e-2


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
            indexes = [
                i // train_config.n_iters
                for i in range(train_config.n_trajs * train_config.n_iters)
            ]

            log = {}
            log['epoch'] = epoch

            rewards, lengths = [], []
            for traj in trajs:
                rewards.append(traj.total_reward)
                lengths.append(len(traj))
            log['train/reward'] = sum(rewards) / len(rewards)
            log['train/length'] = sum(lengths) / len(lengths)

            for i, idx in enumerate(indexes):
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
                if i % train_config.n_iters == 0:
                    # Only update policy once per trajectory
                    p_optim.step()
                v_optim.step()

            # --- Checkpoint ---
            if epoch % 10 == 0:
                if not os.path.exists('checkpoints'):
                    os.makedirs('checkpoints')
                torch.save(model.state_dict(), f'checkpoints/lunarlander_a2c_{epoch}.pt')

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
    torch.save(model.state_dict(), 'checkpoints/lunarlander_a2c.pt')


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
