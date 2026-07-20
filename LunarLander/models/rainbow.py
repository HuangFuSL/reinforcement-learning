'''
DQN

Tricks:
1. Dueling DQN - Q = V + A - mean(A)
2. Double DQN - Use a target network to select the action (and possibly soft update)
3. NoisyNet - Factorized Noisy Linear Layer
4. Prioritized Experience Replay - Proportional Prioritization + Importance Sampling + (beta annealing)
5. Multi-step TD Learning
6. Distributional DQN (C51, QR-DQN, IQN)

Minor tricks:
1. Gradient Clipping
2. Reward Shaping (reward normalization is not used here)
3. Initialization

Future work:
1. Parallel collectors
'''
import torch

from utils.ctorch import nn
from utils.ctorch.rl import BaseDistributionalQNetwork, Trajectory

global __model_name__

__model_name__ = 'Rainbow'


class QNet(BaseDistributionalQNetwork):
    def __init__(
        self,
        state_dim, num_actions, atoms,
        gamma=0.99, double=False, tau=1
    ):
        super(QNet, self).__init__(
            state_dim, num_actions, atoms, gamma=gamma, tau=tau
        )

        self.state_encoding = nn.FactorizedNoisyLinear(self.state_shape[0], 128)
        self.network = nn.DNN(
            128, 128, 128, activation='leaky_relu',
            layer_type=nn.FactorizedNoisyLinear
        )
        self.output = nn.DNN(
            128, (num_actions + 1) * self.num_atoms,
            activation=None, layer_type=nn.FactorizedNoisyLinear
        )
        if double:
            self.setup_target()

        for k, v in self.named_parameters():
            if any(name in k for name in ['_mu', '_sigma', '_device_tracker']):
                continue
            if 'weight' in k:
                if 'norm' in k:
                    torch.nn.init.normal_(v)
                else:
                    torch.nn.init.kaiming_normal_(v)
            elif 'bias' in k:
                torch.nn.init.zeros_(v)
        self.update_target()

    def forward(self, state, action: torch.Tensor | None = None):
        state = state.to(self.device).float()
        state_embedding = self.state_encoding(state)
        x = state_embedding
        x = self.network(x)
        *B, _ = x.shape
        out = self.output(x).view(*B, self.num_actions + 1, self.num_atoms)
        a, v = torch.split(out, self.num_actions, dim=-2)

        # ret: (B, num_actions, num_atoms)
        ret = v + a - a.mean(dim=-2, keepdim=True)

        if action is not None:
            action = action.to(device=self.device, dtype=torch.long).view(-1, 1, 1).expand(-1, 1, self.num_atoms)
            ret = ret.gather(1, action).squeeze(1)
        return ret

    def loss(self, trajectory: Trajectory):
        current, target = self.td_step(trajectory)
        kl = -(target.exp() * current).sum(dim=-1)  # KL divergence
        return kl, kl.abs()
