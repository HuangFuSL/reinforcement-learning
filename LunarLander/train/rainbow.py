'''
Rainbow DQN training for LunarLander-v3 (discrete).

Usage::

    # Default hyperparameters
    python -m LunarLander.train.rainbow

    # Full parameter specification (shown with defaults)
    ./taskq.sh python -m LunarLander.train.rainbow \
        --model-gamma 0.99 \
        --model-tau 5 \
        --model-atom-min -50 \
        --model-atom-max 100 \
        --model-atom-step 51 \
        --model-device auto \
        --optim-name Adam \
        --optim-lr 5e-4 \
        --optim-clip-grad-norm 10.0 \
        --per-size 200000 \
        --per-pre-fill-trajs 64 \
        --per-beta 0.4 \
        --per-alpha 0.6 \
        --per-beta-decay 1.002 \
        --per-alpha-decay 1.0 \
        --per-beta-max 1.0 \
        --per-alpha-max 1.0 \
        --per-device auto \
        --train-epoch 30 \
        --train-n-iters 4000 \
        --train-num-workers 2 \
        --train-batch-size 128 \
        --train-eps 0.05 \
        --train-n-trajs 16

    # Boolean flags use --flag / --no-flag (e.g. --model-double / --no-model-double)
    # Load from YAML/JSON config file (CLI args take priority)
    python -m LunarLander.train.rainbow --config configs/rainbow.yaml
'''

import asyncio
import dataclasses
import os
import torch
import tqdm
import wandb
import wandb.integration.gym

from utils.ctorch.rl.replaybuffer import PrioritizedReplayBuffer
from utils.ctorch.rl.parallel import SyncedEnvPool
from utils.ctorch.device import get_best_device
from utils.ctorch.rl.interface import run_episode
from utils.parser import auto_cli, parse_all_args

from ..models.rainbow import QNet, __model_name__
from ..config import make_env, reward_shape, make_demo_env, init_wandb

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
torch.set_num_threads(1)

def _resolve_device(s: str) -> torch.device:
    return torch.device(get_best_device() if s == 'auto' else s)

@auto_cli
@dataclasses.dataclass
class ModelConfig:
    gamma: float = 0.99
    double: bool = True
    tau: int = 5

    atom_min: float = -50
    atom_max: float = 100
    atom_step: int = 51

    device: str = 'auto'

    def get(self):
        device = _resolve_device(self.device)
        atoms = torch.linspace(
            self.atom_min, self.atom_max, self.atom_step, device=device
        )
        model = QNet(
            state_dim=8, num_actions=4,
            atoms=atoms, gamma=self.gamma, double=self.double, tau=self.tau
        )
        model.to(device)
        return model


@auto_cli
@dataclasses.dataclass
class OptimizerConfig:
    name: str = 'Adam'
    lr: float = 5e-4
    clip_grad_norm: float | None = 10.0

    def get(self, model):
        return getattr(torch.optim, self.name)(
            model.parameters(), self.lr
        )


@auto_cli
@dataclasses.dataclass
class PERConfig:
    size: int = 500_000
    pre_fill_trajs: int = 64
    beta: float = 0.4
    alpha: float = 0.6
    beta_decay: float = 1.002
    alpha_decay: float = 1.0
    beta_max: float = 1.0
    alpha_max: float = 1.0

    device: str = 'auto'

    def get(self):
        device = _resolve_device(self.device)
        per = PrioritizedReplayBuffer(self.size)
        per.to(device)
        return per


@auto_cli
@dataclasses.dataclass
class TrainConfig:
    epoch: int = 30
    n_iters: int = 4000
    batch_size: int = 128
    num_workers: int = 8
    eps: float = 0.05
    n_trajs: int = 16


async def main(
    model_config: ModelConfig,
    optim_config: OptimizerConfig,
    per_config: PERConfig,
    train_config: TrainConfig
):
    init_wandb(__model_name__)
    wandb.integration.gym.monitor()
    env = make_env()
    buffer = per_config.get()
    model = model_config.get()
    optimizer = optim_config.get(model)

    pool = SyncedEnvPool(model, make_env, num_workers=train_config.num_workers)

    with pool:
        # --- Pre-fill replay buffer ---
        pre_fill = pool.run_episode(
            model, num_trajectories=per_config.pre_fill_trajs,
            reward_shape=reward_shape, eps=0.0
        )
        for traj in pre_fill:
            buffer.store(traj)

        # --- Training loop ---
        beta = per_config.beta
        alpha = per_config.alpha

        for epoch in tqdm.trange(train_config.epoch):
            beta = min(beta * per_config.beta_decay, per_config.beta_max)
            alpha = min(alpha * per_config.alpha_decay, per_config.alpha_max)

            task = asyncio.create_task(pool.async_run_episode(
                model, num_trajectories=train_config.n_trajs,
                reward_shape=reward_shape, eps=train_config.eps
            ))
            await asyncio.sleep(0)

            log = {}

            for _ in range(train_config.n_iters):
                if log:
                    wandb.log(log)
                    log.clear()

                model.train()
                optimizer.zero_grad()

                # Get batch and loss weight
                sample_idx = buffer.sample_index(train_config.batch_size)
                ipw = buffer.get_ipw(sample_idx, beta=beta, eps=5e-4).to(model.device)
                batch = buffer.get_batch(sample_idx).to(model.device)

                # Calculate TD loss
                loss, error = model.loss(batch)
                buffer.set_weight(sample_idx, error.to(buffer.device), alpha=alpha)

                loss = (loss * ipw).mean()
                log = {
                    'train/alpha': alpha,
                    'train/beta': beta,
                    'train/loss': loss.item(),
                    'train/td_error': error.mean().item(),
                }
                if loss.item() > 1e3:
                    print('loss exploded:', loss.item())
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), optim_config.clip_grad_norm
                ) if optim_config.clip_grad_norm is not None else None
                if grad_norm is not None:
                    log['train/grad_norm'] = grad_norm.item()
                optimizer.step()
                model.update_target(weight=0.995)

            # --- Collect trajectories ---
            trajectories = await task
            rewards, lengths = [], []
            for traj in trajectories:
                buffer.store(traj)
                rewards.append(traj.total_reward)
                lengths.append(len(traj))

            log |= {
                f'sigma/{n}': p.abs().mean().item()
                for n, p in model.named_parameters()
                if '_sigma' in n
            }
            log['train/reward'] = sum(rewards) / len(rewards)
            log['train/length'] = sum(lengths) / len(lengths)
            log['epoch'] = epoch

            # --- Checkpoint ---
            if not os.path.exists('checkpoints'):
                os.makedirs('checkpoints')
            torch.save(model.state_dict(),f'checkpoints/lunarlander_rainbow_{epoch}.pt')

            # --- Demo ---
            demo_env = make_demo_env(video_dir=f'videos/{wandb.run.id}/epoch_{epoch:03d}')
            result = run_episode(demo_env, model, 1000, eps=0.0)
            log['demo/reward'] = result.total_reward
            log['demo/length'] = len(result)
            del demo_env # AttributeError: 'RecordVideo' object has no attribute 'enabled'

            model.train()
            wandb.log(log)

    # --- Final demo & save ---
    demo_env = make_demo_env()
    run_episode(demo_env, model, 1000, eps=0.0)
    model.to('cpu')
    torch.save(model.state_dict(), 'checkpoints/lunarlander_rainbow.pt')


if __name__ == '__main__':
    configs = parse_all_args(
        model=ModelConfig,
        optim=OptimizerConfig,
        per=PERConfig,
        train=TrainConfig
    )
    asyncio.run(main(
        model_config=configs['model'],
        optim_config=configs['optim'],
        per_config=configs['per'],
        train_config=configs['train'],
    ))
