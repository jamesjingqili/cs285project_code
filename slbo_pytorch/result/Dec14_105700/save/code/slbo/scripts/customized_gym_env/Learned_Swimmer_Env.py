import numpy as np
import gym
from gym import utils
from gym.envs.mujoco import mujoco_env
from gym.utils import seeding
from gym import spaces
import torch
import sys
from slbo.scripts.query_model import query_learned_model


class LearnedSwimmerEnv(gym.Env):
    def __init__(self, state_bound, action_bound,state_size,action_size, dynamics_model_path,coeff_scale=1):
        #default should be state_size:16 action_size:2 high:[50. 50.] low:[-50. -50.]
        self.state_bound = state_bound
        self.action_bound = action_bound
        self.state_size = state_size
        self.action_size = action_size
        self.observation_space = spaces.Box(low=-state_bound, high=state_bound, shape=(state_size,), dtype=np.float32)
        self.action_space = spaces.Box(low=-action_bound, high=action_bound, shape=(action_size,), dtype=np.float32)
        self.seed()
        self.model_path = dynamics_model_path
        # model_module = module_from_file("query_model", "/Users/gavenma/Documents/GitHub/cs285_final_project/code/slbo_pytorch/slbo/scripts/query_model.py")
        # self.model = model_module.query_learned_model
        self.model = query_learned_model
        self.coeff_scale = coeff_scale
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def step(self, action):
        next_states = self.model(self.model_path,torch.tensor(self.state,dtype=torch.float), torch.tensor(action,dtype=torch.float))
        next_states = np.clip(next_states.detach().cpu().numpy(), 
            self.observation_space.low, self.observation_space.high)

        # TODO: write down the reward for Swimmer
        scaling = 0.5 * (self.action_space.high - self.action_space.low)
        ctrl_cost = self.coeff_scale*0.005 * np.sum(np.square(action / scaling), axis=-1)
        fwd_reward = next_states[..., -3]
        reward = fwd_reward - ctrl_cost
        ob = self._get_obs()
        return ob, reward, False, {}

    def change_coeff_scale(self,scale):
        self.coeff_scale = scale
        return

    def _get_obs(self):
        return self.state

    def reset(self):
        # TODO: reset the initial condition of the Swimmer
        self.state = np.random.uniform(-1/2*self.state_bound, 
            1/2*self.state_bound, size = self.state_size)

        #self.state = np.random.normal()??
        return self._get_obs()
    
    def render(self):
        pass
