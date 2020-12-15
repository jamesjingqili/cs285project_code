import torch
import numpy as np
import torch.nn.functional as F
from typing import Any, Dict, Union, Optional
from torch.distributions import Independent, Normal

from tianshou.data import Batch, to_torch
#from tianshou.policy import BasePolicy
from distral_BasePolicy import DistralBasePolicy

class DistilledPolicy(DistralBasePolicy):
    """Implementation of vanilla imitation learning.

    :param torch.nn.Module model: a model following the rules in
        :class:`~tianshou.policy.BasePolicy`. (s -> a)
    :param torch.optim.Optimizer optim: for optimizing the model.
    :param str mode: indicate the imitation type ("continuous" or "discrete"
        action space), defaults to "continuous".

    .. seealso::

        Please refer to :class:`~tianshou.policy.BasePolicy` for more detailed
        explanation.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optim: torch.optim.Optimizer,
        action_range,
        mode: str = "continuous",
        deterministic_eval: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.model = model
        self.optim = optim
        assert (
            mode in ["continuous", "discrete"]
        ), f"Mode {mode} is not in ['continuous', 'discrete']."
        self.mode = mode
        self.__eps = np.finfo(np.float32).eps.item()
        self._action_bias = (action_range[0] + action_range[1]) / 2.0
        self._action_scale = (action_range[1] - action_range[0]) / 2.0
        self._deterministic_eval = deterministic_eval
        self.action_range = action_range

    def forward(
        self,
        batch: Batch,
        state: Optional[Union[dict, Batch, np.ndarray]] = None,
        **kwargs: Any,
    ) -> Batch:
        logits, h = self.model(batch.obs, state=state, info=batch.info)
        assert isinstance(logits, tuple)
        dist = Independent(Normal(*logits), 1)
        
        if self._deterministic_eval and not self.training:
            x = logits[0]
        else:
            x = dist.rsample()
        
        #x = logits[0]
        y = torch.tanh(x)
        act = y * self._action_scale + self._action_bias
        y = self._action_scale * (1 - y.pow(2)) + self.__eps
        log_prob = dist.log_prob(x).unsqueeze(-1)
        log_prob = log_prob - torch.log(y).sum(-1, keepdim=True)


        '''
        if self.mode == "discrete":
            a = logits.max(dim=1)[1]
        else:
            a = logits
        '''
        
        act = act.clamp(self.action_range[0], self.action_range[1])
        #import pdb; pdb.set_trace()
        return Batch(logits=logits, act=act, state=h, log_prob = log_prob)


    def learn(self, batch1: Batch, batch2: Batch, **kwargs: Any) -> Dict[str, float]:
        self.optim.zero_grad()
        if self.mode == "continuous":  # regression
            a1 = self(batch1).act
            a_1 = to_torch(batch1.act, dtype=torch.float32, device=a1.device)
            a2 = self(batch2).act
            a_2 = to_torch(batch1.act, dtype=torch.float32, device=a2.device)
            loss = F.mse_loss(a1, a_1) + F.mse_loss(a2,a_2)  # type: ignore
        elif self.mode == "discrete":  # classification
            a = self(batch1).logits
            a_ = to_torch(batch1.act, dtype=torch.long, device=a.device)
            loss = F.nll_loss(a, a_)  # type: ignore
        #import pdb; pdb.set_trace()
        loss.backward()
        self.optim.step()
        return {"loss": loss.item()}

