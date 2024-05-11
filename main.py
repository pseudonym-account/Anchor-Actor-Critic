import torch
import gym

from logx import EpochLogger, setup_logger_kwargs

from models import *
from memory import *
from config import Config
from utils import *

from sac import SACAgent
from eac import EACAgent
from dpi import DefaultAgent
from anchor_sac import *
from mop import AnchorMOPAnchor, AnchorMOPAgent
from laplacian import LaplacianAnchor, LaplacianAgent

import multiprocessing as mp
import random

from envs import *

from warnings import filterwarnings

from gym.envs.registration import register

register(
    id='no_reward_ant',
    entry_point='envs:NoRewardAntEnv',
)

register(
    id='MazeAnt',
    entry_point='envs:MazeAntEnv',
)


filterwarnings(
    action="ignore",
    category=DeprecationWarning,
    message="`np.bool8` is a deprecated alias for `np.bool_`",
)


def simple_test(env, model):
    o, d, ep_ret, ep_len = env.reset()[0], False, 0, 0
    while not (d) and ep_len < 10000:
        a = model.act(torch.as_tensor(o, dtype=torch.float32), deterministic=False)
        o, r, d, _, info = env.step(a)
        ep_ret += r
        ep_len += 1
        env.render()
    print(ep_ret, ep_len)


def worker(args, seed):
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)
    torch.set_num_threads(torch.get_num_threads())
    env = lambda: gym.make(args.env)
    config = Config("config.json", **vars(args))
    if args.model == "sac":
        agent = SACAgent(config=config, env_fn=lambda: env())

    elif args.model == "mop":
        anchor = AnchorMOPAnchor(config=config, env_fn=lambda: env())
        if config.static_anchor:
            anchor.train()
        agent = AnchorMOPAgent(config=config, env_fn=lambda: env(), anchor=anchor)

    elif args.model == "anchor_sac":
        assert config.static_anchor==False
        anchor = AnchorSACAnchor(config=config, env_fn=lambda: env())
        if config.static_anchor:
            anchor.train()
        agent = AnchorSACAgent(config=config, env_fn=lambda: env(), anchor=anchor)

    elif args.model == "lap":
        anchor = LaplacianAnchor(config=config, env_fn=lambda: env())
        agent = LaplacianAgent(config=config, env_fn=lambda: env(), anchor=anchor)
    
    agent.train()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str)
    parser.add_argument("--path", type=str)
    parser.add_argument("--exp_name", type=str)
    parser.add_argument("--model", type=str)
    parser.add_argument("--hid", type=int)
    parser.add_argument("--l", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--gamma", type=float)
    parser.add_argument("--polyak", type=float)
    parser.add_argument("--seed", "-s", type=int)
    parser.add_argument("--update_interval", "-i", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--alpha", type=float)
    parser.add_argument("--as_alpha", type=float)
    parser.add_argument("-C", type=float)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--static_anchor", action="store_true")
    parser.add_argument("--auto_a", action="store_true")
    args = parser.parse_args()

    torch.set_num_threads(torch.get_num_threads())

    if args.test:
        env = gym.make(args.env, render_mode="human") 
        model = torch.load(args.path).to("cpu")
        simple_test(env, model)
    else:
        worker(args, args.seed)