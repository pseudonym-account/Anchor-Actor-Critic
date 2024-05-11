import numpy as np
import math
import gymnasium as gym
from gym import utils
from gym.envs.mujoco import MujocoEnv
from gym.spaces import Box
import numpy as np
import gym
import time
from gym.envs.mujoco import MujocoEnv
from gym.spaces import Box
from gym.envs.mujoco.ant_v4 import AntEnv

import numpy as np

from gym import utils
from gym.envs.mujoco import MujocoEnv
from gym.envs.mujoco.humanoid_v4 import HumanoidEnv, mass_center
from gym.envs.mujoco.ant_v4 import AntEnv
from gym.spaces import Box
import mujoco
import random
import os
os.path.dirname(os.path.abspath(__file__))

DEFAULT_CAMERA_CONFIG = {
    "distance": 4.0,
}

class TwoTaskHumEnv(HumanoidEnv):
    def __init__(self, task=1, random_tasks=True, **kwargs):
        
        # self.env = AntEnv(xml_file=xml_file_path, **kwargs)
        super(TwoTaskHumEnv, self).__init__(**kwargs)
        
        self.random_task = random_tasks
        self.task = task
        obs_shape = 377
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_shape,), dtype=np.float64)
    
    def step(self, action):
        xy_position_before = mass_center(self.model, self.data)
        self.do_simulation(action, self.frame_skip)
        xy_position_after = mass_center(self.model, self.data)

        xy_velocity = (xy_position_after - xy_position_before) / self.dt
        x_velocity, y_velocity = xy_velocity

        ctrl_cost = self.control_cost(action)

        forward_reward = self._forward_reward_weight * x_velocity
        forward_reward *= self.task
        
        healthy_reward = self.healthy_reward

        rewards = forward_reward + healthy_reward

        observation = self._get_obs()
        reward = rewards - ctrl_cost
        terminated = self.terminated
        info = {
            "reward_linvel": forward_reward,
            "reward_quadctrl": -ctrl_cost,
            "reward_alive": healthy_reward,
            "x_position": xy_position_after[0],
            "y_position": xy_position_after[1],
            "distance_from_origin": np.linalg.norm(xy_position_after, ord=2),
            "x_velocity": x_velocity,
            "y_velocity": y_velocity,
            "forward_reward": forward_reward,
        }

        if self.render_mode == "human":
            self.render()
        observation = np.concatenate([np.array([self.task], dtype=float), observation])
        return observation, reward, terminated, False, info
    
    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = self.init_qvel + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nv
        )
        self.set_state(qpos, qvel)

        observation = self._get_obs()
        if self.random_task:
            self.task = random.choice([1, -1])
        observation = np.concatenate([np.array([self.task], dtype=float), observation])
        return observation
    
class FourTaskHumEnv(HumanoidEnv):
    def __init__(self, task=1, random_tasks=True, **kwargs):
        
        # self.env = AntEnv(xml_file=xml_file_path, **kwargs)
        super(FourTaskHumEnv, self).__init__(**kwargs)
        
        self.random_task = random_tasks
        self.task = task
        obs_shape = 377
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_shape,), dtype=np.float64)
    
    def cal_reward(self, xv, yv):
        if self.task == 0:
            return xv
        elif self.task == 1:
            return -xv
        elif self.task == 2:
            return yv
        elif self.task == 3:
            return -yv
        raise "Not a non case"
    
    def choose_random_task(self):
        return random.choice([0, 1, 2, 3])
     
    def step(self, action):
        xy_position_before = mass_center(self.model, self.data)
        self.do_simulation(action, self.frame_skip)
        xy_position_after = mass_center(self.model, self.data)

        xy_velocity = (xy_position_after - xy_position_before) / self.dt
        x_velocity, y_velocity = xy_velocity
        
        velocity_reward = self.cal_reward(xv=x_velocity, yv=y_velocity)

        ctrl_cost = self.control_cost(action)

        forward_reward = self._forward_reward_weight * velocity_reward
        
        healthy_reward = self.healthy_reward

        rewards = forward_reward + healthy_reward

        observation = self._get_obs()
        reward = rewards - ctrl_cost
        terminated = self.terminated
        info = {
            "reward_linvel": forward_reward,
            "reward_quadctrl": -ctrl_cost,
            "reward_alive": healthy_reward,
            "x_position": xy_position_after[0],
            "y_position": xy_position_after[1],
            "distance_from_origin": np.linalg.norm(xy_position_after, ord=2),
            "x_velocity": x_velocity,
            "y_velocity": y_velocity,
            "forward_reward": forward_reward,
        }

        if self.render_mode == "human":
            self.render()
        observation = np.concatenate([np.array([self.task], dtype=float), observation])
        return observation, reward, terminated, False, info
    
    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = self.init_qvel + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nv
        )
        self.set_state(qpos, qvel)

        observation = self._get_obs()
        
        if self.random_task:
            self.task = self.choose_random_task()
        
        observation = np.concatenate([np.array([self.task], dtype=float), observation])
        return observation

class NoRewardAntEnv(AntEnv):
    def __init__(self, **kwargs):
        super(NoRewardAntEnv, self).__init__(**kwargs)
        self.healthy_z_range = (0.3, 1)
    
    def step(self, action):
        xy_position_before = self.get_body_com("torso")[:2].copy()
        self.do_simulation(action, self.frame_skip)
        xy_position_after = self.get_body_com("torso")[:2].copy()

        xy_velocity = (xy_position_after - xy_position_before) / self.dt
        x_velocity, y_velocity = xy_velocity

        # forward_reward = x_velocity
        forward_reward = 0
        
        # Modify forward reward to consider task.
        healthy_reward = self.healthy_reward
        rewards = forward_reward + healthy_reward
        costs = ctrl_cost = self.control_cost(action)
        
        terminated = self.terminated
        observation = self._get_obs()
        info = {
            "reward_forward": forward_reward,
            "reward_ctrl": -ctrl_cost,
            "reward_survive": healthy_reward,
            "x_position": xy_position_after[0],
            "y_position": xy_position_after[1],
            "distance_from_origin": np.linalg.norm(xy_position_after, ord=2),
            "x_velocity": x_velocity,
            "y_velocity": y_velocity,
            "forward_reward": forward_reward,
        }
        
        if self._use_contact_forces:
            contact_cost = self.contact_cost
            costs += contact_cost
            info["reward_ctrl"] = -contact_cost

        reward = rewards - costs

        if self.render_mode == "human":
            self.render()
        reward = 1
        return observation, reward, terminated, False, info
    
    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = (
            self.init_qvel
            + self._reset_noise_scale * self.np_random.standard_normal(self.model.nv)
        )
        
        self.set_state(qpos, qvel)
        observation = self._get_obs()
        
        return observation    

class TwoTaskAntEnv(AntEnv):
    def __init__(self, task=1, random_tasks=True, **kwargs):
        
        # self.env = AntEnv(xml_file=xml_file_path, **kwargs)
        super(TwoTaskAntEnv, self).__init__(**kwargs)
        
        self.random_task = random_tasks
        self.task = task
        obs_shape = 28
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_shape,), dtype=np.float64)
    
    
    def step(self, action):
        # print("=" * 50)
        xy_position_before = self.get_body_com("torso")[:2].copy()
        self.do_simulation(action, self.frame_skip)
        xy_position_after = self.get_body_com("torso")[:2].copy()

        xy_velocity = (xy_position_after - xy_position_before) / self.dt
        x_velocity, y_velocity = xy_velocity

        forward_reward = x_velocity
        
        # Modify forward reward to consider task.
        forward_reward *= self.task
        
        healthy_reward = self.healthy_reward

        rewards = forward_reward + healthy_reward

        costs = ctrl_cost = self.control_cost(action)

        terminated = self.terminated
        observation = self._get_obs()
        info = {
            "reward_forward": forward_reward,
            "reward_ctrl": -ctrl_cost,
            "reward_survive": healthy_reward,
            "x_position": xy_position_after[0],
            "y_position": xy_position_after[1],
            "distance_from_origin": np.linalg.norm(xy_position_after, ord=2),
            "x_velocity": x_velocity,
            "y_velocity": y_velocity,
            "forward_reward": forward_reward,
        }
        if self._use_contact_forces:
            contact_cost = self.contact_cost
            costs += contact_cost
            info["reward_ctrl"] = -contact_cost

        reward = rewards - costs

        if self.render_mode == "human":
            self.render()
        # print(self.task)
        # Add task to observation
        observation = np.concatenate([np.array([self.task], dtype=float), observation])
        
        return observation, reward, terminated, False, info
    
    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = (
            self.init_qvel
            + self._reset_noise_scale * self.np_random.standard_normal(self.model.nv)
        )
        
        self.set_state(qpos, qvel)

        observation = self._get_obs()
        
        # print("/"*50)
        if self.random_task:
            self.task = random.choice([1, -1])
        # print(self.task)
        # Add task to observation
        observation = np.concatenate([np.array([self.task], dtype=float), observation])
        return observation    

class FourTaskAntEnv(AntEnv):
    def __init__(self, task=0, random_tasks=True, **kwargs):
        
        # self.env = AntEnv(xml_file=xml_file_path, **kwargs)
        super(FourTaskAntEnv, self).__init__(**kwargs)
        
        self.random_task = random_tasks
        self.task = task
        obs_shape = 28
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_shape,), dtype=np.float64)
    
    def choose_random_task(self):
        return random.choice([0, 1, 2, 3])
    
    def cal_reward(self, xv, yv):
        if self.task == 0:
            return xv
        elif self.task == 1:
            return -xv
        elif self.task == 2:
            return yv
        elif self.task == 3:
            return -yv
        raise "Not a non case"
     
    def step(self, action):
        xy_position_before = self.get_body_com("torso")[:2].copy()
        self.do_simulation(action, self.frame_skip)
        xy_position_after = self.get_body_com("torso")[:2].copy()

        xy_velocity = (xy_position_after - xy_position_before) / self.dt
        x_velocity, y_velocity = xy_velocity

        # forward_reward = x_velocity
        
        # Modify forward reward to consider task.
        forward_reward = self.cal_reward(xv=x_velocity, yv=y_velocity)
        
        healthy_reward = self.healthy_reward
        rewards = forward_reward + healthy_reward
        costs = ctrl_cost = self.control_cost(action)

        terminated = self.terminated
        observation = self._get_obs()
        info = {
            "reward_forward": forward_reward,
            "reward_ctrl": -ctrl_cost,
            "reward_survive": healthy_reward,
            "x_position": xy_position_after[0],
            "y_position": xy_position_after[1],
            "distance_from_origin": np.linalg.norm(xy_position_after, ord=2),
            "x_velocity": x_velocity,
            "y_velocity": y_velocity,
            "forward_reward": forward_reward,
        }
        if self._use_contact_forces:
            contact_cost = self.contact_cost
            costs += contact_cost
            info["reward_ctrl"] = -contact_cost

        reward = rewards - costs

        if self.render_mode == "human":
            self.render()

        observation = np.concatenate([np.array([self.task], dtype=float), observation])
        
        return observation, reward, terminated, False, info
    
    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = (
            self.init_qvel
            + self._reset_noise_scale * self.np_random.standard_normal(self.model.nv)
        )
        
        self.set_state(qpos, qvel)

        observation = self._get_obs()
        
        if self.random_task:
            self.task = self.choose_random_task()
        observation = np.concatenate([np.array([self.task], dtype=float), observation])
        return observation    

class MazeAntEnv(AntEnv):
    def __init__(self, xml_file_path="maze.xml", exclude_current_positions_from_observation=False, **kwargs):
        super(MazeAntEnv, self).__init__(
            xml_file=f"{os.path.dirname(os.path.abspath(__file__))}/{xml_file_path}", 
            exclude_current_positions_from_observation=exclude_current_positions_from_observation, 
            **kwargs)
    
    def step(self, action):
        xy_position_before = self.get_body_com("torso")[:2].copy()
        self.do_simulation(action, self.frame_skip)
        xy_position_after = self.get_body_com("torso")[:2].copy()

        xy_velocity = (xy_position_after - xy_position_before) / self.dt
        x_velocity, y_velocity = xy_velocity

        # print(xy_position_after[0])
        # forward_reward = x_velocity
        forward_reward = math.sqrt(xy_position_after[0]**2 + xy_position_after[1]**2)
        # print(forward_reward)
        # Modify forward reward to consider task.
        healthy_reward = self.healthy_reward
        rewards = forward_reward + healthy_reward
        costs = ctrl_cost = self.control_cost(action)
        
        terminated = self.terminated
        observation = self._get_obs()
        info = {
            "reward_forward": forward_reward,
            "reward_ctrl": -ctrl_cost,
            "reward_survive": healthy_reward,
            "x_position": xy_position_after[0],
            "y_position": xy_position_after[1],
            "distance_from_origin": np.linalg.norm(xy_position_after, ord=2),
            "x_velocity": x_velocity,
            "y_velocity": y_velocity,
            "forward_reward": forward_reward,
        }
        
        if self._use_contact_forces:
            contact_cost = self.contact_cost
            costs += contact_cost
            info["reward_ctrl"] = -contact_cost

        reward = rewards - costs

        if self.render_mode == "human":
            self.render()
        return observation, reward, terminated, False, info 
         
if __name__ == "__main__":
    env = TwoTaskAntEnv()
    env.reset()
    env.step(env.action_space.sample())
    
    
    