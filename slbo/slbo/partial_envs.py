# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import numpy as np
from slbo.envs.mujoco.half_cheetah_env import HalfCheetahEnv
from slbo.envs.mujoco.walker2d_env import Walker2DEnv
from slbo.envs.mujoco.humanoid_env import HumanoidEnv
from slbo.envs.mujoco.ant_env import AntEnv
from slbo.envs.mujoco.hopper_env import HopperEnv
from slbo.envs.mujoco.swimmer_env import SwimmerEnv
from slbo.envs.mujoco.inverted_double_pendulum_env import InvertedDoublePendulumEnv


def make_env(id: str,goal=None):
    envs = {
        'HalfCheetah-v2': HalfCheetahEnv,
        'Walker2D-v2': Walker2DEnv,
        'Humanoid-v2': HumanoidEnv,
        'Ant-v2': AntEnv,
        'Hopper-v2': HopperEnv,
        'Swimmer-v2': SwimmerEnv,
        'InvertedDoublePendulum-v2': InvertedDoublePendulumEnv
    }
    env = envs[id]()
    if not hasattr(env, 'reward_range'):
        env.reward_range = (-np.inf, np.inf)
    if not hasattr(env, 'metadata'):
        env.metadata = {}
    env.seed(np.random.randint(2**60))
    if goal is not None:
        if id == 'Swimmer-v2':
            env.mult_coef_factor(goal)
        elif id== 'HalfCheetah-v2':
            env.set_goal(goal)
        elif id == 'InvertedDoublePendulum-v2':
            env.change_bonus(goal)
    return env
