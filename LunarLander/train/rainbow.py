import asyncio
import torch
import tqdm
import wandb

from utils.ctorch.rl.replaybuffer import PrioritizedReplayBuffer
from utils.ctorch.rl.parallel import SyncedEnvPool
from utils.ctorch.device import get_best_device
from utils.ctorch.rl.interface import run_episode

from ..models.rainbow import QNet
from ..config import make_env, reward_shape

async def main():
    wandb.init(
        project='Reinforcement-Learning',
        group='LunarLander-Discrete'
    )
    env = make_env()
    device = get_best_device()
    buffer = PrioritizedReplayBuffer(40000)
    atoms = torch.linspace(-50.0, 100.0, steps=51)
    model = QNet(
        state_dim=8,
        num_actions=4,
        atoms=atoms,
        gamma=0.99, double=True, tau=5
    )
    model.to(device)
    buffer.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    num_episodes = 1000
    beta = 0.4
    alpha = 0.6
    beta_decay = 1.002
    n_iters = 50

    pool = SyncedEnvPool(model, make_env, num_workers=8)

    with pool:
        pre_fill = pool.run_episode(
            model, num_trajectories=128, reward_shape=reward_shape, eps=0.0
        )
        for traj in pre_fill:
            buffer.store(traj)

        for i in tqdm.trange(num_episodes):
            beta = min(beta * beta_decay, 1.0)
            task = asyncio.create_task(pool.async_run_episode(
                model, num_trajectories=16, reward_shape=reward_shape, eps=0.05
            ))

            log = {}

            for j in range(n_iters):
                if log:
                    wandb.log(log)
                    log = {}

                model.train()
                optimizer.zero_grad()

                # Get batch and loss weight
                sample_idx = buffer.sample_index(128)
                ipw = buffer.get_ipw(sample_idx, beta=beta, eps=5e-4)
                batch = buffer.get_batch(sample_idx)

                # Calculate TD loss
                q_value = model.Q_all(batch.state).detach()
                loss, error = model.loss(batch)
                buffer.set_weight(sample_idx, error, alpha=alpha)

                loss = (loss * ipw).mean()
                log = {
                    'train/alpha': alpha,
                    'train/beta': beta,
                    'train/loss': loss.item(),
                    'train/td_error': error.mean().item(),
                    'train/q_mean': q_value.mean().item(),
                    'train/q_std': q_value.std().item()
                }
                if loss.item() > 1e3:
                    print('loss exploded:', loss.item())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                optimizer.step()
                model.update_target(weight=0.995)

            trajectories = await task
            for traj in trajectories:
                buffer.store(traj)

            log |= {
                f'sigma/{n}': p.abs().mean().item()
                for n, p in model.named_parameters()
                if '_sigma' in n
            }
            log['train/reward'] = result.total_reward
            log['train/length'] = len(result)
            model.eval()
            result = run_episode(env, model, eps=0.0)
            log['eval/reward'] = result.total_reward
            log['eval/length'] = len(result)
            model.train()
            wandb.log(log)

    model.to('cpu')
    torch.save(model.state_dict(), 'lunarlander_rainbow.pt')

if __name__ == '__main__':
    asyncio.run(main())