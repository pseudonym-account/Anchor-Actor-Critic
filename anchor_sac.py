import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

import time
import os.path as osp

from copy import deepcopy
import numpy as np
import torch
from torch.optim import Adam
import gym
import time

from logx import EpochLogger, setup_logger_kwargs

from models import *
from memory import *
from utils import *
import threading

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
LOG_ALPHA_MIN = -20
LOG_ALPHA_MAX = 2

 
class ASActorCritiCModel(nn.Module):
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


class AnchorSACAnchor:
    def __init__(
        self,
        config,
        env_fn,
    ) -> None:
        self.env_fn = env_fn
        for key, value in config.to_dict().items():
            setattr(self, key, value)
        del key
        del value
        self.alpha = config.anchor.alpha
        self.exp_name = self.exp_name + "_anchor_model"
        logger_kwargs = setup_logger_kwargs(self.exp_name, self.seed)
        self.logger = EpochLogger(**logger_kwargs)
        self_locals = locals()
        del self_locals["self"]
        self.logger.save_config(config=self_locals)

        self.train_step = 0
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self.update_interval = self.update_interval

        self.env, self.test_env = self.env_fn(), self.env_fn()
        self.a_limit = self.env.action_space.high[0]

        self.ac_kwargs = dict(hidden_sizes=[self.hid] * self.l)
        self.ac, self.ac_targ = create_actor_critic(
            ASActorCritiCModel,
            observation_space=self.env.observation_space,
            action_space=self.env.action_space,
            lr=self.lr,
            **self.ac_kwargs,
        )

        # Experience buffer
        obs_dim = self.env.observation_space.shape[0]
        act_dim = self.env.action_space.shape[0]
        self.replay_buffer = ReplayBuffer(
            obs_dim=obs_dim, act_dim=act_dim, size=self.replay_size
        )

        self.logger.setup_pytorch_saver([self.ac])

    def update_q(self, data):
        o, a, r, o2, d = (
            data["obs"],
            data["act"],
            data["rew"],
            data["obs2"],
            data["done"],
        )

        with torch.no_grad():
            pi2, pi2_mu, pi2_std = self.ac.pi(o2)
            logp_pi2 = adjusted_log(pi2_mu, pi2_std, pi2)
            a2 = self.ac.scale_action(pi2)

            q1_pi_targ = self.ac_targ.q1(o2, a2)
            q2_pi_targ = self.ac_targ.q2(o2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)

            backup = r + self.gamma * (1 - d) * (q_pi_targ - self.alpha * logp_pi2)

        q1 = self.ac.q1(o, a)
        q2 = self.ac.q2(o, a)

        loss_q1 = ((q1 - backup) ** 2).mean()
        loss_q2 = ((q2 - backup) ** 2).mean()
        loss_q = loss_q1 + loss_q2

        self.ac.q_optimizer.zero_grad()
        loss_q.backward()
        self.ac.q_optimizer.step()

        self.logger.store(
            RefLossQ=loss_q.item(),
            RefQ1Vals=q1.cpu().detach().numpy(),
            RefQ2Vals=q2.cpu().detach().numpy(),
        )

    # Set up function for computing Agent pi loss
    def update_pi(self, data):
        o = data["obs"]

        pi, pi_mu, pi_std = self.ac.pi(o)
        logp_pi = adjusted_log(pi_mu, pi_std, pi)
        a = self.ac.scale_action(pi)

        q1_pi = self.ac.q1(o, a)
        q2_pi = self.ac.q2(o, a)
        q_pi = torch.min(q1_pi, q2_pi)
        
        loss_pi = (self.alpha * logp_pi - q_pi).mean()
        self.ac.pi_optimizer.zero_grad()
        loss_pi.backward()
        self.ac.pi_optimizer.step()

        self.logger.store(
            RefStd=pi_std.mean().item(),
            RefLossPi=loss_pi.item(),
            RefLogPi=logp_pi.cpu().detach().numpy(),
        )

    def update_target(self, step):     
        if step % self.update_interval == 0:
            self.ac_targ.load_state_dict(self.ac.state_dict())

    def update(self, data):
        self.update_q(data)

        for p in self.ac.q_params:
            p.requires_grad = False

        self.update_pi(data)

        for p in self.ac.q_params:
            p.requires_grad = True

    def save(self):
        self.logger.save_state(None)

    def get_action(self, o, deterministic=False):
        return self.ac.act(
            torch.as_tensor(o, dtype=torch.float32).to(device), deterministic
        )

    def test_agent(self):
        with torch.no_grad():
            for j in range(self.num_test_episodes):
                o, d, ep_ret, ep_len = self.test_env.reset()[0], False, 0, 0
                while not (d or (ep_len == self.max_ep_len)):
                    o, r, d, _, _ = self.test_env.step(
                        self.get_action(o, deterministic=True)
                    )
                    ep_ret += r
                    ep_len += 1
                self.logger.store(RefTestEpRet=ep_ret, RefTestEpLen=ep_len)

    def log_epoch_info(self, epoch, start_time, t, static_anchor=True):
        if not static_anchor:
            self.logger.log_tabular("RefLogPi", average_only=True)
            self.logger.log_tabular("RefLossPi", average_only=True)
            self.logger.log_tabular("RefStd", average_only=True)
            self.logger.log_tabular("RefQ1Vals", average_only=True)
            self.logger.log_tabular("RefQ2Vals", average_only=True)
            self.logger.log_tabular("RefLossQ", average_only=True)
            self.logger.dump_tabular()
        else:
            # Log info about epoch
            self.logger.log_tabular(
                "Experiment Name", self.exp_name + "_" + str(self.seed)
            )
            self.logger.log_tabular("Epoch", epoch)
            self.logger.log_tabular("RefEpRet", average_only=True)
            self.logger.log_tabular("RefTestEpRet", average_only=True)
            self.logger.log_tabular("RefEpLen", average_only=True)
            self.logger.log_tabular("RefTestEpLen", average_only=True)
            self.logger.log_tabular("RefTotalEnvInteracts", t)
            self.logger.log_tabular("RefQ1Vals", average_only=True)
            self.logger.log_tabular("RefQ2Vals", average_only=True)
            self.logger.log_tabular("RefLogPi", average_only=True)
            self.logger.log_tabular("RefLossPi", average_only=True)
            self.logger.log_tabular("RefLossQ", average_only=True)
            self.logger.log_tabular("RefStd", average_only=True)
            self.logger.log_tabular("train_step", average_only=True)

            self.logger.log_tabular("Time", time.time() - start_time)
            self.logger.dump_tabular()

    def get_episode(self, actor, rand=False, train=False):
        with torch.no_grad():
            o, d, ep_ret, ep_len = self.env.reset()[0], False, 0, 0
            while not (d or (ep_len == self.max_ep_len)):
                if rand:
                    a = self.env.action_space.sample()
                else:
                    obs = (
                        torch.as_tensor(o, dtype=torch.float32).unsqueeze(0).to(device)
                    )
                    pi, pi_mu, pi_std = actor(obs, deterministic=False)
                    a = self.ac.scale_action(pi)
                    a = a.detach().cpu().numpy()[0]

                o2, r, d, _, info = self.env.step(a)
                d = False if ep_len == self.max_ep_len else d
                self.replay_buffer.store(o, a, r, o2, d)
                o = o2
                ep_len += 1
                self.t += 1
                ep_ret += r

            self.logger.store(RefEpRet=ep_ret, RefEpLen=ep_len)
            return ep_len

    def train(self):
        # Prepare for interaction with environment
        total_steps = self.steps_per_epoch * self.epochs
        start_time = time.time()

        self.t = 0
        train = False
        print_step = 0
        while self.t < total_steps:
            rand = False
            if self.t < self.start_steps:
                rand = True
            ep_len = self.get_episode(self.ac.pi, rand)

            # Update handling
            if self.t >= self.update_after:
                train = True
                for ts in range(self.t - ep_len, self.t):
                    self.train_step += 1
                    self.logger.store(train_step=self.train_step)
                    batch = self.replay_buffer.sample_batch(self.batch_size)
                    self.update(data=batch)
                    self.update_target(self.train_step)

            # End of epoch handling
            if int(self.t / self.steps_per_epoch) > print_step:
                epoch = (self.t + 1) // self.steps_per_epoch
                print_step += 1
                if (epoch % self.save_freq == 0) or (epoch == self.epochs):
                    self.logger.save_state(None)

                # Test the performance of the deterministic version of the agent.
                self.test_agent()

                if self.t < self.update_after:
                    continue
                if train:
                    self.log_epoch_info(epoch, start_time, self.t)


class AnchorSACAgent:
    def __init__(
        self,
        env_fn,
        config,
        anchor=None,
    ) -> None:
        self.env_fn = env_fn
        for key, value in config.to_dict().items():
            setattr(self, key, value)
        
        del key
        del value
        
        assert self.static_anchor is not None
        self.anchor = anchor

        logger_kwargs = setup_logger_kwargs(self.exp_name, self.seed)
        self.logger = EpochLogger(**logger_kwargs)
        config = locals()
        del config["anchor"]
        del config["self"]
        self.logger.save_config(config=config)

        self.train_step = 0
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        self.threads_results = []
        self.env, self.test_env = self.env_fn(), self.env_fn()

        # Action limit for clamping: critically, assumes all dimensions share the same bound!
        self.a_limit = self.env.action_space.high[0]
        self.clip_norm_value = 1

        self.ac_kwargs = dict(hidden_sizes=[self.hid] * self.l)
        self.ac, self.ac_targ = create_actor_critic(
            ASActorCritiCModel,
            observation_space=self.env.observation_space,
            action_space=self.env.action_space,
            lr=self.lr,
            **self.ac_kwargs,
        )

        if self.auto_a:
            self.alpha = 1
            self.log_alpha = torch.tensor([0.0], requires_grad=True, device=device)
            self.a_optimizer = optim.Adam(params=[self.log_alpha], lr=self.lr)
            # self.log_alpha = torch.clamp(self.log_alpha, LOG_ALPHA_MIN, LOG_ALPHA_MAX)

        # Experience buffer
        obs_dim = self.env.observation_space.shape[0]
        act_dim = self.env.action_space.shape[0]
        self.replay_buffer = ReplayBuffer(
            obs_dim=obs_dim, act_dim=act_dim, size=self.replay_size
        )

        self.data = []

        # Set up model saving
        self.logger.setup_pytorch_saver([self.ac])

    # Set up function for computing Agent Q-losses
    def update_q(self, data):
        o, a, r, o2, d = (
            data["obs"],
            data["act"],
            data["rew"],
            data["obs2"],
            data["done"],
        )

        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            pi2, pi2_mu, pi2_std = self.ac.pi(o2)
            logp_pi2 = adjusted_log(pi2_mu, pi2_std, pi2)
            a2 = self.ac.scale_action(pi2)

            # Target Q-values
            q1_pi_targ = self.ac_targ.q1(o2, a2)
            q2_pi_targ = self.ac_targ.q2(o2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)

            ref_pi2, ref_pi2_mu, ref_pi2_std = self.anchor.ac.pi(o2)
            ref_logp_pi2 = adjusted_log(ref_pi2_mu, ref_pi2_std, pi2)
            # ref_a2 = self.anchor.ac.scale_action(ref_pi2)

            f = logp_pi2 - ref_logp_pi2
            self.logger.store(f = f.mean().item())
            backup = r + self.gamma * (1 - d) * (q_pi_targ - self.alpha * f)

        q1 = self.ac.q1(o, a)
        q2 = self.ac.q2(o, a)

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup) ** 2).mean()
        loss_q2 = ((q2 - backup) ** 2).mean()
        loss_q = loss_q1 + loss_q2

        # First run one gradient descent step for Q1 and Q2
        self.ac.q_optimizer.zero_grad()
        loss_q.backward()
        torch.nn.utils.clip_grad_norm_(self.ac.q_params, self.clip_norm_value)
        self.ac.q_optimizer.step()

        # Record things
        q_info = dict(
            Q1Vals=q1.cpu().detach().numpy(), Q2Vals=q2.cpu().detach().numpy()
        )
        self.logger.store(LossQ=loss_q.item(), **q_info)

    # Set up function for computing Agent pi loss
    def update_pi(self, data):
        o = data["obs"]

        pi, pi_mu, pi_std = self.ac.pi(o)
        logp_pi = adjusted_log(pi_mu, pi_std, pi)
        a = self.ac.scale_action(pi)

        q1_pi = self.ac.q1(o, a)
        q2_pi = self.ac.q2(o, a)
        q_pi = torch.min(q1_pi, q2_pi)

        ref_pi, ref_pi_mu, ref_pi_std = self.anchor.ac.pi(o)
        ref_logp_pi = adjusted_log(ref_pi_mu, ref_pi_std, pi)
        # ref_a = self.anchor.ac.scale_action(ref_pi)

        f = logp_pi - ref_logp_pi
        loss_pi = -(q_pi - self.alpha * f).mean()

        self.ac.pi_optimizer.zero_grad()
        loss_pi.backward()
        torch.nn.utils.clip_grad_norm_(self.ac.pi.parameters(), self.clip_norm_value)
        self.ac.pi_optimizer.step()

        # Record things
        pi_info = dict(LogPi=logp_pi.cpu().detach().numpy())
        self.logger.store(std=pi_std.mean().item())
        self.logger.store(LossPi=loss_pi.item(), **pi_info)

    def update_target(self, step):
        if step % self.update_interval == 0:
            if not self.static_anchor:
                self.anchor.update_target(step)
            self.ac_targ.load_state_dict(self.ac.state_dict())

    def update_alpha(self, data):
        o = data["obs"]

        pi, pi_mu, pi_std = self.ac.pi(o)
        logp_pi = adjusted_log(pi_mu, pi_std, pi)

        ref_pi, ref_pi_mu, ref_pi_std = self.anchor.ac.pi(o)
        ref_logp_pi = adjusted_log(ref_pi_mu, ref_pi_std, pi)
        a_loss = (self.log_alpha * (-logp_pi + ref_logp_pi + self.C)).mean()

        self.a_optimizer.zero_grad()
        a_loss.backward()
        self.a_optimizer.step()
        self.alpha = torch.exp(self.log_alpha).cpu().detach().item()
        self.logger.store(alpha=self.alpha, a_loss=a_loss.item())

    def update(self, data):
        if not self.static_anchor:
            self.anchor.update(data)

        self.update_q(data)

        for p in self.ac.q_params:
            p.requires_grad = False

        self.update_pi(data)

        for p in self.ac.q_params:
            p.requires_grad = True

        if self.auto_a:
            self.update_alpha(data)

    def get_action(self, o, deterministic=False):
        return self.ac.act(
            torch.as_tensor(o, dtype=torch.float32).to(device), deterministic
        )

    def test_agent(self):
        with torch.no_grad():
            for j in range(self.num_test_episodes):
                o, d, ep_ret, ep_len = self.test_env.reset()[0], False, 0, 0
                while not (d or (ep_len == self.max_ep_len)):
                    # Take deterministic actions at test time
                    o, r, d, _, _ = self.test_env.step(
                        self.get_action(o, deterministic=True)
                    )
                    ep_ret += r
                    ep_len += 1
                self.logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)

    def log_epoch_info(self, epoch, start_time, t):
        # Log info about epoch
        self.logger.log_tabular("Experiment Name", self.exp_name + "_" + str(self.seed))
        self.logger.log_tabular("Epoch", epoch)
        self.logger.log_tabular("EpRet", average_only=True)
        self.logger.log_tabular("TestEpRet", average_only=True)
        self.logger.log_tabular("EpLen", average_only=True)
        self.logger.log_tabular("TestEpLen", average_only=True)
        self.logger.log_tabular("TotalEnvInteracts", t)
        self.logger.log_tabular("Q1Vals", average_only=True)
        self.logger.log_tabular("Q2Vals", average_only=True)
        self.logger.log_tabular("LogPi", average_only=True)
        self.logger.log_tabular("LossPi", average_only=True)
        self.logger.log_tabular('f', average_only=True)
        self.logger.log_tabular("LossQ", average_only=True)
        self.logger.log_tabular("std", average_only=True)
        try:
            self.logger.log_tabular("train_step", average_only=True)
        except:
            pass

        if self.auto_a:
            self.logger.log_tabular("alpha", average_only=True)
            self.logger.log_tabular("a_loss", average_only=True)

        self.logger.log_tabular("Time", time.time() - start_time)
        self.logger.dump_tabular()

        if self.anchor and not self.static_anchor:
            self.anchor.log_epoch_info(epoch, start_time, t, self.static_anchor)

    def get_episode(self, actor, rand=False):
        with torch.no_grad():
            o, d, ep_ret, ep_len = self.env.reset()[0], False, 0, 0
            while not (d or (ep_len == self.max_ep_len)):
                if rand:
                    a = self.env.action_space.sample() 
                else:
                    obs = torch.as_tensor(o, dtype=torch.float32).unsqueeze(0).to(device)
                    pi, pi_mu, pi_std = actor(obs, deterministic=False)
                    a = self.ac.scale_action(pi)
                    a = a.detach().cpu().numpy()[0]
                o2, r, d, _, info = self.env.step(a)
                d = False if ep_len == self.max_ep_len else d
                self.replay_buffer.store(o, a, r, o2, d)
                o = o2
                ep_len += 1
                self.t += 1
                ep_ret += r
            self.logger.store(EpRet=ep_ret, EpLen=ep_len)
            return ep_len

    def train(self):
        # Prepare for interaction with environment
        total_steps = self.steps_per_epoch * self.epochs
        start_time = time.time()

        self.t = 0
        train = False
        print_step = 0
        while self.t < total_steps:
            rand = False
            if self.t < self.start_steps:
                rand = True
            ep_len = self.get_episode(self.ac.pi, rand)

            # Update handling
            if self.t >= self.update_after:
                train = True
                for _ in range(ep_len):
                    self.train_step += 1
                    self.logger.store(train_step=self.train_step)
                    batch = self.replay_buffer.sample_batch(self.batch_size)
                    self.update(data=batch)
                    self.update_target(self.train_step)

            # End of epoch handling
            if int(self.t / self.steps_per_epoch) > print_step:
                epoch = (self.t + 1) // self.steps_per_epoch
                print_step += 1
                if (epoch % self.save_freq == 0) or (epoch == self.epochs):
                    self.logger.save_state(None)
                    if self.anchor:
                        self.anchor.save()

                # Test the performance of the deterministic version of the agent.
                self.test_agent()

                if self.t < self.update_after:
                    continue
                if train:
                    self.log_epoch_info(epoch, start_time, self.t)
