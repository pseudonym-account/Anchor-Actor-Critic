import numpy as np
import scipy.signal
import random
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

import time
import os.path as osp

from copy import deepcopy
import itertools
import numpy as np
import torch
from torch.optim import Adam
import gym
import time
import os
from logx import EpochLogger, setup_logger_kwargs

import os
import imageio
import numpy as np
from PIL import Image
import PIL.ImageDraw as ImageDraw
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from warnings import filterwarnings

filterwarnings(
    action="ignore",
    category=DeprecationWarning,
    message="`np.bool` is a deprecated alias",
)


class BipedalWalkerHardcoreWrapper(object):
    def __init__(self, action_repeat=3):
        self._env = gym.make("BipedalWalker-v3")
        self.action_repeat = action_repeat
        self.act_noise = 0.3
        self.reward_scale = 5.0
        self.observation_space = self._env.observation_space

    def __getattr__(self, name):
        return getattr(self._env, name)

    def reset(self):
        obs = self._env.reset()
        return obs

    def step(self, action):
        action += self.act_noise * (-2 * np.random.random(4) + 1)
        r = 0.0
        for _ in range(self.action_repeat):
            obs_, reward_, done_, info_, x = self._env.step(action)
            if done_ and reward_ == -100:
                reward_ = 0
            r = r + reward_
            if done_ and self.action_repeat != 1:
                return obs_, 0.0, done_, info_, x
            if self.action_repeat == 1:
                return obs_, r, done_, info_, x
        return obs_, self.reward_scale * r, done_, info_, x


def adjusted_log(mu, std, pi):
    # https://github.com/openai/spinningup/issues/279
    logp_pi = Normal(mu, std).log_prob(pi).sum(axis=-1)
    logp_pi -= (2 * (np.log(2) - pi - F.softplus(-2 * pi))).sum(axis=1)
    return logp_pi


def create_actor_critic(actor_critic, observation_space, action_space, lr, **ac_kwargs):
    # Create actor-critic module and target networks
    ac = actor_critic(observation_space, action_space, lr=lr, **ac_kwargs).to(device)
    ac_targ = deepcopy(ac).to(device)

    # Freeze target networks with respect to optimizers (only update via polyak averaging)
    for p in ac_targ.parameters():
        p.requires_grad = False

    return ac, ac_targ


def draw(xys, path):
    plt.figure()

    def symmetrize_axes(axes):
        y_max = max(np.abs(axes.get_ylim()).max() + 1, 10)
        axes.set_ylim(ymin=-y_max, ymax=y_max)

        x_max = max(np.abs(axes.get_xlim()).max() + 1, 10)
        axes.set_xlim(xmin=-x_max, xmax=x_max)

    for i, position in enumerate(xys[:-1]):
        x, y = position
        x_, y_ = xys[i + 1]
        plt.plot((x, x_), (y, y_), "b")

    ax = plt.gca()  # .set_aspect('equal', 'datalim')
    symmetrize_axes(ax)
    plt.savefig(path)
    plt.close()


def get_last_file_number(directory):
    files = os.listdir(directory)
    max_number = -1
    for file in files:
        if file.startswith("info_") and file.endswith(".csv"):
            try:
                number = int(file.split("_")[1].split(".")[0])
                if number > max_number:
                    max_number = number
            except ValueError:
                continue
    return max_number


def simple_test(env, path: str, mode="human"):
    env = gym.make(env, render_mode=mode)
    ac = torch.load(path).to("cpu")

    o, d, ep_ret, ep_len = env.reset()[0], False, 0, 0
    while not (d) and ep_len < 1000:
        # Take deterministic actions at test time
        a = ac.act(torch.as_tensor(o, dtype=torch.float32), deterministic=True)
        o, r, d, _, info = env.step(a)
        print(o[0])
        ep_ret += r
        ep_len += 1
    print(ep_ret, ep_len)


def test_env(env):
    env = gym.make(env)
    o, d, ep_ret, ep_len = env.reset()[0], False, 0, 0
    while not (d) and ep_len < 1000:
        a = env.action_space.sample()
        o, r, d, _, info = env.step(a)
        ep_ret += r
        ep_len += 1
    print(ep_ret, ep_len)


def test_model_human(env, path: str, mode="human"):
    env = gym.make(env, render_mode=mode)
    ac = torch.load(path).to("cpu")

    from pathlib import Path

    path = Path(path)
    path_to_info_csv = os.path.join(path.parent.parent.absolute(), "infos")
    path_to_state_csv = os.path.join(path.parent.parent.absolute(), "states")
    path_to_action_csv = os.path.join(path.parent.parent.absolute(), "actions")
    path_to_gif = os.path.join(path.parent.parent.absolute(), "videos")
    path_to_map = os.path.join(path.parent.parent.absolute(), "maps")

    for dir in [
        path_to_info_csv,
        path_to_state_csv,
        path_to_action_csv,
        path_to_gif,
        path_to_map,
    ]:
        if not os.path.exists(dir):
            os.mkdir(dir)

    m = get_last_file_number(path_to_info_csv)
    path_to_info_csv = os.path.join(path_to_info_csv, f"info_{m+1}.csv")
    path_to_state_csv = os.path.join(path_to_state_csv, f"states_{m+1}.csv")
    path_to_action_csv = os.path.join(path_to_action_csv, f"actions_{m+1}.csv")
    path_to_gif = os.path.join(path_to_gif, f"video_{m+1}.gif")
    path_to_map = os.path.join(path_to_map, f"map_{m+1}.png")

    o, d, ep_ret, ep_len = env.reset()[0], False, 0, 0
    df_index = 0
    xys = []
    while not (d) and ep_len < 1000:
        # Take deterministic actions at test time
        a = ac.act(torch.as_tensor(o, dtype=torch.float32), deterministic=True)
        o, r, d, _, info = env.step(a)

        xys.append((info["x_position"], info["y_position"]))
        if df_index == 0:
            df_info = pd.DataFrame(columns=list(info.keys()))
            df_state = pd.DataFrame(columns=list(range(o.shape[0])))
            df_action = pd.DataFrame(columns=list(range(a.shape[0])))

        df_info.loc[df_index] = info
        df_state.loc[df_index] = {e: v for e, v in zip(range(o.shape[0]), o)}
        df_action.loc[df_index] = {e: v for e, v in zip(range(a.shape[0]), a)}
        frame = env.render()

        ep_ret += r
        ep_len += 1
        df_index += 1

    print(ep_ret)
    df_info.to_csv(path_to_info_csv, index=False)
    df_state.to_csv(path_to_state_csv, index=False)
    df_action.to_csv(path_to_action_csv, index=False)

    def symmetrize_axes(axes):
        y_max = np.abs(axes.get_ylim()).max() + 1
        axes.set_ylim(ymin=-y_max, ymax=y_max)

        x_max = np.abs(axes.get_xlim()).max() + 1
        axes.set_xlim(xmin=-x_max, xmax=x_max)

    for i, position in enumerate(xys[:-1]):
        x, y = position
        x_, y_ = xys[i + 1]
        plt.plot((x, x_), (y, y_), "b")

    ax = plt.gca()
    symmetrize_axes(ax)
    plt.savefig(path_to_map)
