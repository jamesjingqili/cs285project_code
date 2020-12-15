import os
import shutil
import time
from collections import deque
from operator import itemgetter

import numpy as np
import tqdm
import torch

from torch.utils.tensorboard import SummaryWriter

from slbo.models import Dynamics,Normalizers




def query_learned_model(model_path,state,action):
    # state, action must be torch tensors
    # model_path refers to path that saves the dictionary that contains model, which is produced by run_slbo.py

    state_dim = 16
    action_dim = 2

    normalizers = Normalizers(action_dim, state_dim)

    random_state = torch.rand(state_dim)
    random_action = torch.rand(action_dim)

    dynamics_model = Dynamics(state_dim, action_dim, [500, 500], normalizer=normalizers)

    dict = torch.load(model_path)

    dynamics_model.load_state_dict(dict['dynamics'])

    dynamics_model.eval()
    return dynamics_model(state,action)