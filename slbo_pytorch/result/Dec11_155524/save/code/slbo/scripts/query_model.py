import os
import shutil
import time
from collections import deque
from operator import itemgetter

import numpy as np
import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter

from slbo.algos import SLBO, PPO, TRPO
from slbo.configs.config import Config
from slbo.envs.wrapped_envs import make_vec_envs, make_vec_virtual_envs
from slbo.misc.ou_noise import OUNoise
from slbo.misc.utils import log_and_write, evaluate
from slbo.models import Actor, ActorCritic, Dynamics, VCritic, Normalizers
from slbo.storages.off_policy_buffer import OffPolicyBuffer
from slbo.storages.on_policy_buffer import OnPolicyBuffer

config, hparam_dict = Config('slbo_config.yaml')
import datetime

ph_envs = make_vec_envs(config.env.env_name, config.seed, config.env.num_real_envs, config.env.gamma, './', 'cpu',
                                allow_early_resets=True, norm_reward=False, norm_obs=False, test=False,
                                max_episode_steps=config.env.max_episode_steps,goal=0)

state_dim = ph_envs.observation_space.shape[0] 
action_space = ph_envs.action_space
action_dim = action_space.shape[0]

normalizers = Normalizers(action_dim, state_dim)

dynamics_model = Dynamics(state_dim, action_dim, config.slbo.dynamics_hidden_dims, normalizer=normalizers)

dict = torch.load('/Users/gavenma/Desktop/actor_critic_dynamics_epoch9.pt')

dynamics_model.load_state_dict(dict['dynamics'])

dynamics_model.eval()