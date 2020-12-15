import numpy as np
from rllab.envs.mujoco import inverted_double_pendulum_env
from slbo.envs import BaseModelBasedEnv


class InvertedDoublePendulumEnv(inverted_double_pendulum_env.InvertedDoublePendulumEnv, BaseModelBasedEnv):

    def change_bonus(self, bonus):
        self.alive_bonus=bonus

    def mb_step(self, states: np.ndarray, actions: np.ndarray, next_states: np.ndarray):
        x, _, y = self.model.data.site_xpos[0]
        # alive_bonus = 10
        r = -next_states[:,-3]+self.alive_bonus
        if y <= 1:
            done = np.ones_like(r, dtype=np.bool)
        else:
            done = np.zeros_like(r, dtype=np.bool)
        return r, done
