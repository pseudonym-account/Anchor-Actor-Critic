import numpy as np
import scipy.signal

from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.optim import Adam

import torch.optim as optim
import itertools

import torch.distributions as distributions

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

LOG_STD_MAX = 2
LOG_STD_MIN = -20


# def mlp(sizes, activation, output_activation=nn.Identity):
#     layers = []
#     for j in range(len(sizes) - 1):
#         act = activation if j < len(sizes) - 2 else output_activation
#         layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
#     return nn.Sequential(*layers)

def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        layers.append(nn.Linear(sizes[j], sizes[j+1]))
        if j < len(sizes)-2:  # Do not add LayerNorm in the output layer
            layers.append(nn.LayerNorm(sizes[j+1]))
            layers.append(activation())
        else:
            layers.append(output_activation())
    return nn.Sequential(*layers)

def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


class SquashedGaussianMLPActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit):
        super().__init__()
        self.net = mlp([obs_dim] + list(hidden_sizes), activation, activation)
        self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.act_limit = act_limit

    def forward(self, obs, deterministic=False):
        net_out = self.net(obs)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        if not deterministic:
            pi_distribution = Normal(mu, std)
            pi = pi_distribution.rsample()
        else:
            pi = mu

        return pi, mu, std


class LegacySquashedGaussianMLPActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit):
        super().__init__()
        self.net = mlp([obs_dim] + list(hidden_sizes), activation, activation)
        self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.act_limit = act_limit

    def forward(self, obs, deterministic=False, with_logprob=True, with_std=False):
        # print(obs)
        net_out = self.net(obs)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        # Pre-squash distribution and sample
        pi_distribution = Normal(mu, std)
        if deterministic:
            # Only used for evaluating policy at test time.
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()

        if with_logprob:
            # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
            # NOTE: The correction formula is a little bit magic. To get an understanding
            # of where it comes from, check out the original SAC paper (arXiv 1801.01290)
            # and look in appendix C. This is a more numerically-stable equivalent to Eq 21.
            # Try deriving it yourself as a (very difficult) exercise. :)
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_pi -= (2 * (np.log(2) - pi_action - F.softplus(-2 * pi_action))).sum(
                axis=1
            )
        else:
            logp_pi = None

        pi_action = torch.tanh(pi_action)
        pi_action = self.act_limit * pi_action

        if with_std:
            return pi_action, logp_pi, std
        else:
            return pi_action, logp_pi


def create_actor_critic(actor_critic, observation_space, action_space, lr, **ac_kwargs):
    # Create actor-critic module and target networks
    ac = actor_critic(observation_space, action_space, lr=lr, **ac_kwargs).to(device)
    ac_targ = deepcopy(ac).to(device)

    # Freeze target networks with respect to optimizers (only update via polyak averaging)
    for p in ac_targ.parameters():
        p.requires_grad = False

    return ac, ac_targ


class MLPQFunction(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs, act):
        q = self.q(torch.cat([obs, act], dim=-1))
        return torch.squeeze(q, -1)  # Critical to ensure q has right shape.


class ActorCritiCModel(nn.Module):
    def __init__(
        self,
        observation_space,
        action_space,
        hidden_sizes=(256, 256),
        activation=nn.ReLU,
        lr=0.0003,
        rm_dims=[],
    ):
        super().__init__()
        self.rm_dims = rm_dims
        self.obs_dim = observation_space.shape[0]
        self.act_dim = action_space.shape[0]
        self.act_limit = action_space.high[0]

        # build policy and value functions
        self.pi = SquashedGaussianMLPActor(
            self.obs_dim, self.act_dim, hidden_sizes, nn.ReLU, self.act_limit
        ).to(device)
        self.q1 = MLPQFunction(self.obs_dim, self.act_dim, hidden_sizes, activation)
        self.q2 = MLPQFunction(self.obs_dim, self.act_dim, hidden_sizes, activation)

        self.pi_optimizer = Adam(self.pi.parameters(), lr=lr)
        self.q_params = itertools.chain(self.q1.parameters(), self.q2.parameters())
        self.q_optimizer = Adam(self.q_params, lr=lr)

    def scale_action(self, a):
        a = torch.tanh(a)
        a = self.act_limit * a
        return a

    def act(self, obs, deterministic=False):
        with torch.no_grad():
            a, _, _ = self.pi(obs, deterministic)
            a = self.scale_action(a)
            return a.cpu().numpy()


class FixedPi(nn.Module):
    def __init__(self, act_dim, std) -> None:
        super().__init__()
        self._ = nn.Linear(1, 1)

        self.mu = torch.tensor([0 for _ in range(act_dim)], dtype=torch.float32)
        self.std = torch.tensor([std for _ in range(act_dim)], dtype=torch.float32)

    def forward(self, obs, deterministic=False, with_logprob=True, with_std=False):
        device = obs.device  # Get the device of the observations
        batch_size = obs.shape[0]

        pi_action = self.mu.to(device).expand(batch_size, -1)
        pi_distribution = Normal(
            self.mu.to(device).expand(batch_size, -1),
            self.std.to(device).expand(batch_size, -1),
        )

        if with_logprob:
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_pi -= (2 * (np.log(2) - pi_action - F.softplus(-2 * pi_action))).sum(
                axis=1
            )
        else:
            logp_pi = None

        if with_std:
            return pi_action, logp_pi, self.std.to(device).expand(batch_size, -1)
        else:
            return pi_action, logp_pi


class FixedActorCritic(nn.Module):
    def __init__(self, act_dim, std=0.8):
        super().__init__()
        self.pi = FixedPi(act_dim, std=std)

    def act(self):
        return 0
