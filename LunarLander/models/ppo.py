
import torch

from utils.ctorch import nn
from utils.ctorch.rl import BasePolicyNetwork, BaseValueNetwork, Trajectory
from torch.distributions import Categorical

global __model_name__

__model_name__ = 'PPO'

class Critic(BaseValueNetwork):
    def __init__(self, gamma=0.99):
        super().__init__(8, gamma=gamma)
        input_numel = int(torch.prod(torch.tensor(self.state_shape)).item())
        self.network = nn.DNN(
            input_numel, 128, 128, 128,
            activation='leaky_relu'
        )
        self.value_fn = nn.DNN(128, 1, activation=None)

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
    def forward(self, state: torch.Tensor, action: None = None) -> torch.Tensor:
        if action is not None:
            raise ValueError("Critic does not take action as input.")
        state = state.to(self.device).float()
        x = self.network(state)
        return self.value_fn(x).squeeze(-1)

    def loss(self, batch: Trajectory) -> torch.Tensor:
        current, target = self.value_td_step(batch)
        loss = torch.nn.functional.smooth_l1_loss(current, target)
        return loss

class PPO(BasePolicyNetwork):
    def __init__(self, gamma=0.99, epsilon=0.2, lambda_=0.95):
        super().__init__(8, gamma=gamma)
        self.epsilon = epsilon
        self.lambda_ = lambda_
        input_numel = int(torch.prod(torch.tensor(self.state_shape)).item())
        self.network = nn.DNN(
            input_numel, 128, 128, 128, 4,
            activation='leaky_relu',
            bare_last_layer=True
        )
        self.value_fn = Critic(gamma=gamma)
        self.value_loss = torch.nn.SmoothL1Loss(reduction='none')

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


    @property
    def value_model(self):
        return self.value_fn

    def forward(self, state: torch.Tensor):
        state = state.to(self.device).float()
        x = self.network(state)
        return Categorical(logits=x)

    def loss(self, trajectory):
        s, a, r, s_prime, done, log_pi = trajectory
        pi = self(s)
        logits = pi.log_prob(a)
        ratio = (logits - log_pi).exp()
        clipped_ratio = ratio.clamp(1 - self.epsilon, 1 + self.epsilon)
        entropy = pi.entropy()
        approx_kl = (ratio - 1) - (logits - log_pi)  # 2nd-order Taylor approx

        current, target = self.value_model.value_td_step(trajectory)
        adv = (target - current).detach()
        adv_raw = self.cumulative_reward(adv, self.lambda_)
        adv_norm = self.normalize_trajectory(adv_raw)

        td_target = (current + adv_raw).detach()

        pg = -torch.min(adv_norm * ratio, adv_norm * clipped_ratio)

        sv = self.value_loss(current, td_target)
        return pg.mean(), sv.mean(), entropy.mean(), approx_kl.mean()
