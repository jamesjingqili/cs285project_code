import time
import tqdm
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, List, Union, Callable, Optional

from tianshou.data import Collector
from tianshou.policy import BasePolicy
from tianshou.utils import tqdm_config, MovAvg
from tianshou.trainer import test_episode, gather_info
#from learn_pi_0 import pi_0_update
# 只加了一个train_collector2
def Distral_offpolicy_trainer(
    policy: BasePolicy,
    train_collector1: Collector,
    train_collector2: Collector,
    test_collector: Collector,
    max_epoch: int,
    step_per_epoch: int,
    collect_per_step: int,
    episode_per_test: Union[int, List[int]],
    batch_size: int,
    update_per_step: int = 1,
    train_fn: Optional[Callable[[int, int], None]] = None,
    test_fn: Optional[Callable[[int, Optional[int]], None]] = None,
    stop_fn: Optional[Callable[[float], bool]] = None,
    save_fn: Optional[Callable[[BasePolicy], None]] = None,
    writer: Optional[SummaryWriter] = None,
    log_interval: int = 1,
    verbose: bool = True,
    test_in_train: bool = True,
    exist_distilled_policy = False,
    distilled_policy = None,
) -> Dict[str, Union[float, str]]:
    """A wrapper for off-policy trainer procedure.

    The "step" in trainer means a policy network update.

    :param policy: an instance of the :class:`~tianshou.policy.BasePolicy`
        class.
    :param train_collector: the collector used for training.
    :type train_collector: :class:`~tianshou.data.Collector`
    :param test_collector: the collector used for testing.
    :type test_collector: :class:`~tianshou.data.Collector`
    :param int max_epoch: the maximum of epochs for training. The training
        process might be finished before reaching the ``max_epoch``.
    :param int step_per_epoch: the number of step for updating policy network
        in one epoch.
    :param int collect_per_step: the number of frames the collector would
        collect before the network update. In other words, collect some frames
        and do some policy network update.
    :param episode_per_test: the number of episodes for one policy evaluation.
    :param int batch_size: the batch size of sample data, which is going to
        feed in the policy network.
    :param int update_per_step: the number of times the policy network would
        be updated after frames are collected, for example, set it to 256 means
        it updates policy 256 times once after ``collect_per_step`` frames are
        collected.
    :param function train_fn: a function receives the current number of epoch
        and step index, and performs some operations at the beginning of
        training in this epoch.
    :param function test_fn: a function receives the current number of epoch
        and step index, and performs some operations at the beginning of
        testing in this epoch.
    :param function save_fn: a function for saving policy when the undiscounted
        average mean reward in evaluation phase gets better.
    :param function stop_fn: a function receives the average undiscounted
        returns of the testing result, return a boolean which indicates whether
        reaching the goal.
    :param torch.utils.tensorboard.SummaryWriter writer: a TensorBoard
        SummaryWriter.
    :param int log_interval: the log interval of the writer.
    :param bool verbose: whether to print the information.
    :param bool test_in_train: whether to test in the training phase.

    :return: See :func:`~tianshou.trainer.gather_info`.
    """
    env_step, gradient_step = 0, 0
    best_epoch, best_reward, best_reward_std = -1, -1.0, 0.0
    stat: Dict[str, MovAvg] = {}
    start_time = time.time()
    train_collector1.reset_stat()
    train_collector2.reset_stat()
    test_collector.reset_stat()
    test_in_train = test_in_train and train_collector1.policy == policy

    for epoch in range(1, 1 + max_epoch):
        # train
        policy.train()
        # ------------------------------------------------------------- the below
        #if exist_distilled_policy:
        #    policy.exist_distilled_policy = True
        #    #policy.distilled_policy = pi_0_update(distilled_policy)
        # ------------------------------------------------------------- the above
        with tqdm.tqdm(
            total=step_per_epoch, desc=f"Epoch #{epoch}", **tqdm_config
        ) as t:
            while t.n < t.total:
                if train_fn:
                    train_fn(epoch, env_step)
                result1 = train_collector1.collect(n_step=collect_per_step)
                env_step += int(result1["n/st"])
                data1 = {
                    "env_step": str(env_step),
                    "rew": f"{result1['rew']:.2f}",
                    "len": str(int(result1["len"])),
                    "n/ep": str(int(result1["n/ep"])),
                    "n/st": str(int(result1["n/st"])),
                    "v/ep": f"{result1['v/ep']:.2f}",
                    "v/st": f"{result1['v/st']:.2f}",
                }
                result2 = train_collector2.collect(n_step=collect_per_step)
                env_step += int(result2["n/st"])
                data2 = {
                    "env_step": str(env_step),
                    "rew": f"{result2['rew']:.2f}",
                    "len": str(int(result2["len"])),
                    "n/ep": str(int(result2["n/ep"])),
                    "n/st": str(int(result2["n/st"])),
                    "v/ep": f"{result2['v/ep']:.2f}",
                    "v/st": f"{result2['v/st']:.2f}",
                }


                '''
                if writer and env_step % log_interval == 0:
                    for k in result1.keys():
                        writer.add_scalar(
                            "train/" + k, result1[k], global_step=env_step)
                
                
                if test_in_train and stop_fn and stop_fn(result["rew"]):
                    test_result1 = test_episode(
                        policy, test_collector, test_fn,
                        epoch, episode_per_test, writer, env_step)
                    if stop_fn(test_result1["rew"]):
                        if save_fn:
                            save_fn(policy)
                        for k in result1.keys():
                            data[k] = f"{result1[k]:.2f}"
                        t.set_postfix(**data)
                        return gather_info(
                            start_time, train_collector1, test_collector,
                            test_result1["rew"], test_result1["rew_std"])
                    else:
                        policy.train()
                '''
                for i in range(update_per_step * min(
                        result1["n/st"] // collect_per_step, t.total - t.n)):
                    gradient_step += 1
                    losses = policy.update(batch_size, train_collector1.buffer, train_collector2.buffer)
                    '''
                    for k in losses.keys():
                        if stat.get(k) is None:
                            stat[k] = MovAvg()
                        stat[k].add(losses[k])
                        data1[k] = f"{stat[k].get():.6f}"
                        if writer and gradient_step % log_interval == 0:
                            writer.add_scalar(
                                k, stat[k].get(), global_step=gradient_step)
                    '''
                    t.update(1)
                    t.set_postfix(**data1)
            if t.n <= t.total:
                t.update()





        # test
        result = test_episode(policy, test_collector, test_fn, epoch,
                              episode_per_test, writer, env_step)
        if best_epoch == -1 or best_reward < result["rew"]:
            best_reward, best_reward_std = result["rew"], result["rew_std"]
            best_epoch = epoch
            if save_fn:
                save_fn(policy)
        if verbose:
            print(f"Epoch #{epoch}: test_reward: {result['rew']:.6f} ± "
                  f"{result['rew_std']:.6f}, best_reward: {best_reward:.6f} ± "
                  f"{best_reward_std:.6f} in #{best_epoch}")
        if stop_fn and stop_fn(best_reward):
            break
    return gather_info(start_time, train_collector1, test_collector,
                       best_reward, best_reward_std)
