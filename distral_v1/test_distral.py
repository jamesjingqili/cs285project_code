# test pi_0
import os
import sys
import gym
import torch
import pprint
import argparse
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from tianshou.env import DummyVectorEnv
from tianshou.utils.net.common import Net
from tianshou.trainer import offpolicy_trainer    ##
from tianshou.data import Collector, ReplayBuffer ##
from tianshou.utils.net.continuous import Actor, ActorProb, Critic


#-------------------------------------------------------------
# The belows are special for policy distral
from distral_offpolicy_trainer import Distral_offpolicy_trainer
####from distral_collector import Distral_Collector    # 惊了，这个也不需要
from distral_task_policy import TaskPolicy
from distral_distilled_policy import DistilledPolicy
#-------------------------------------------------------------



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='Pendulum-v0')  # ‘Pendulum-v0’
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--buffer-size', type=int, default=20000)
    parser.add_argument('--actor-lr', type=float, default=3e-4)
    parser.add_argument('--critic-lr', type=float, default=1e-3)
    parser.add_argument('--il-lr', type=float, default=1e-3)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--tau', type=float, default=0.005)
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--epoch', type=int, default=2)
    parser.add_argument('--step-per-epoch', type=int, default=800)
    parser.add_argument('--collect-per-step', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--layer-num', type=int, default=1)
    parser.add_argument('--training-num', type=int, default=8)
    parser.add_argument('--test-num', type=int, default=100)
    parser.add_argument('--logdir', type=str, default='log')
    parser.add_argument('--render', type=float, default=0.)
    parser.add_argument('--rew-norm', type=int, default=1)
    parser.add_argument('--ignore-done', type=int, default=1)
    parser.add_argument('--n-step', type=int, default=4)
    parser.add_argument('--distilled-policy-training-num', type=int, default = 10)
    parser.add_argument(
        '--device', type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_known_args()[0]
    return args


gym.envs.register(
     id='LearnedSwimmer-v0',
     entry_point='gym.envs.mujoco.Learned_Swimmer_Env:LearnedSwimmerEnv',
     max_episode_steps=1000,
     reward_threshold=360.0,
     kwargs={'state_bound' : 50., 'action_bound':50., 'state_size': 16, 'action_size': 2, 'dynamics_model_path': '/Users/gavenma/Documents/GitHub/cs285_final_project/code/slbo_pytorch/result/Dec12_053104/save/actor_critic_dynamics_epoch9.pt'},
)

args = get_args()
np.random.seed(args.seed)
torch.manual_seed(args.seed)
os.environ['KMP_DUPLICATE_LIB_OK']='True'




torch.set_num_threads(1)  # we just need only one thread for NN
env = gym.make(args.task)
if args.task == 'Pendulum-v0':
    env.spec.reward_threshold = -250
args.state_shape = env.observation_space.shape or env.observation_space.n
args.action_shape = env.action_space.shape or env.action_space.n
args.max_action = env.action_space.high[0]
# you can also use tianshou.env.SubprocVectorEnv
# train_envs = gym.make(args.task)

# model 1

train_envs_1 = DummyVectorEnv(
    [lambda: gym.make(args.task) for _ in range(args.training_num)])
# test_envs = gym.make(args.task)
test_envs_1 = DummyVectorEnv(
    [lambda: gym.make(args.task) for _ in range(args.test_num)])

# seed
train_envs_1.seed(args.seed)
test_envs_1.seed(args.seed)

net_1 = Net(args.layer_num, args.state_shape, device=args.device)
actor_1 = ActorProb(
    net_1, args.action_shape, args.max_action, args.device, unbounded=True
).to(args.device)
actor_optim_1 = torch.optim.Adam(actor_1.parameters(), lr=args.actor_lr)
net_c1_1 = Net(args.layer_num, args.state_shape,
               args.action_shape, concat=True, device=args.device)
critic1_1 = Critic(net_c1_1, args.device).to(args.device)
critic1_optim_1 = torch.optim.Adam(critic1_1.parameters(), lr=args.critic_lr)
net_c2_1 = Net(args.layer_num, args.state_shape,
               args.action_shape, concat=True, device=args.device)
critic2_1 = Critic(net_c2_1, args.device).to(args.device)
critic2_optim_1 = torch.optim.Adam(critic2_1.parameters(), lr=args.critic_lr)
policy_1 = TaskPolicy(
    actor_1, actor_optim_1, critic1_1, critic1_optim_1, critic2_1, critic2_optim_1,
    action_range=[env.action_space.low[0], env.action_space.high[0]],
    tau=args.tau, gamma=args.gamma, alpha=args.alpha,
    reward_normalization=args.rew_norm,
    ignore_done=args.ignore_done,
    estimation_step=args.n_step)

# model 2

train_envs_2 = DummyVectorEnv(
    [lambda: gym.make(args.task) for _ in range(args.training_num)])
# test_envs = gym.make(args.task)
test_envs_2 = DummyVectorEnv(
    [lambda: gym.make(args.task) for _ in range(args.test_num)])

# seed
train_envs_2.seed(args.seed)
test_envs_2.seed(args.seed)

net_2 = Net(args.layer_num, args.state_shape, device=args.device)
actor_2 = ActorProb(
    net_2, args.action_shape, args.max_action, args.device, unbounded=True
).to(args.device)
actor_optim_2 = torch.optim.Adam(actor_2.parameters(), lr=args.actor_lr)
net_c1_2 = Net(args.layer_num, args.state_shape,
               args.action_shape, concat=True, device=args.device)
critic1_2 = Critic(net_c1_2, args.device).to(args.device)
critic1_optim_2 = torch.optim.Adam(critic1_2.parameters(), lr=args.critic_lr)
net_c2_2 = Net(args.layer_num, args.state_shape,
               args.action_shape, concat=True, device=args.device)
critic2_2 = Critic(net_c2_2, args.device).to(args.device)
critic2_optim_2 = torch.optim.Adam(critic2_2.parameters(), lr=args.critic_lr)
policy_2 = TaskPolicy(
    actor_2, actor_optim_2, critic1_2, critic1_optim_2, critic2_2, critic2_optim_2,
    action_range=[env.action_space.low[0], env.action_space.high[0]],
    tau=args.tau, gamma=args.gamma, alpha=args.alpha,
    reward_normalization=args.rew_norm,
    ignore_done=args.ignore_done,
    estimation_step=args.n_step)


# collector
train_collector_1 = Collector(
    policy_1, train_envs_1, ReplayBuffer(args.buffer_size))
test_collector_1 = Collector(policy_1, test_envs_1)
train_collector_2 = Collector(
    policy_2, train_envs_2, ReplayBuffer(args.buffer_size))
test_collector_2 = Collector(policy_2, test_envs_2)
# train_collector.collect(n_step=args.buffer_size)
# log
log_path = os.path.join(args.logdir, args.task, 'sac_distral')
writer = SummaryWriter(log_path)


def save_fn(policy):
    torch.save(policy.state_dict(), os.path.join(log_path, 'policy.pth'))


def stop_fn(mean_rewards):
    return mean_rewards >= env.spec.reward_threshold

# training
distilled_oplicy_training_epoch = 10

for itr in range(10):
    if itr == 0:
        # train policy 1
        result = offpolicy_trainer(
                policy_1, train_collector_1, test_collector_1, args.epoch,
                args.step_per_epoch, args.collect_per_step, args.test_num,
                args.batch_size, stop_fn=stop_fn, save_fn=save_fn, writer=writer)
        if __name__ == '__main__':
            pprint.pprint(result)
            # Let's watch its performance!
            policy_1.eval()
            collector = Collector(policy_1, env)
            result1 = collector.collect(n_episode=1, render=args.render)
            print(f'Final reward: {result1["rew"]}, length: {result1["len"]}')
        
        # train policy 2
        result = offpolicy_trainer(
                policy_2, train_collector_2, test_collector_2, args.epoch,
                args.step_per_epoch, args.collect_per_step, args.test_num,
                args.batch_size, stop_fn=stop_fn, save_fn=save_fn, writer=writer)
        if __name__ == '__main__':
            pprint.pprint(result)
            # Let's watch its performance!
            policy_2.eval()
            collector = Collector(policy_2, env)
            result2 = collector.collect(n_episode=1, render=args.render)
            print(f'Final reward: {result2["rew"]}, length: {result2["len"]}')
        
        # train distilled policy
        if args.task == 'Pendulum-v0':
            env.spec.reward_threshold = -300  # lower the goal
        net = ActorProb(
                Net(1, args.state_shape, device=args.device),
                args.action_shape, args.max_action, args.device
                ).to(args.device)
        
        optim = torch.optim.Adam(net.parameters(), lr=args.il_lr)

        distilled_policy = DistilledPolicy(net, optim, 
                    action_range=[env.action_space.low[0], env.action_space.high[0]],
                            mode='continuous')

        distilled_policy_test_collector = Collector(
                    distilled_policy,
                    DummyVectorEnv(
                        [lambda: gym.make(args.task) for _ in range(args.test_num)])
                )

        train_collector_1.reset()
        train_collector_2.reset()

        result = Distral_offpolicy_trainer(
                    distilled_policy, train_collector_1, train_collector_2, 
                    distilled_policy_test_collector, distilled_oplicy_training_epoch,
                    args.step_per_epoch // 5, args.collect_per_step, args.test_num,
                    args.batch_size, stop_fn=stop_fn, save_fn=save_fn, writer=writer)
        if __name__ == '__main__':
            pprint.pprint(result)
            # Let's watch its performance!
            distilled_policy.eval()
            collector = Collector(distilled_policy, env)
            result = collector.collect(n_episode=1, render=args.render)
            print(f'Final reward: {result["rew"]}, length: {result["len"]}')
    else:
        if stop_fn(result1["rew"]) and stop_fn(result2["rew"]):
            print('Training finished! All tasks are solved.')
            break
        # train policy 1
        result = offpolicy_trainer(
                policy_1, train_collector_1, test_collector_1, args.epoch,
                args.step_per_epoch, args.collect_per_step, args.test_num,
                args.batch_size, stop_fn=stop_fn, save_fn=save_fn, writer=writer)
        if __name__ == '__main__':
            pprint.pprint(result)
            # Let's watch its performance!
            policy_1.eval()
            collector = Collector(policy_1, env)
            result1 = collector.collect(n_episode=1, render=args.render)
            print(f'Final reward: {result1["rew"]}, length: {result1["len"]}')
        # train policy 2
        result = offpolicy_trainer(
                policy_2, train_collector_2, test_collector_2, args.epoch,
                args.step_per_epoch, args.collect_per_step, args.test_num,
                args.batch_size, stop_fn=stop_fn, save_fn=save_fn, writer=writer)
        if __name__ == '__main__':

            pprint.pprint(result)
            # Let's watch its performance!
            policy_2.eval()
            collector = Collector(policy_2, env)
            result2 = collector.collect(n_episode=1, render=args.render)
            print(f'Final reward: {result2["rew"]}, length: {result2["len"]}')
        # train distilled policy
        train_collector_1.reset()
        train_collector_2.reset()
        
        result = Distral_offpolicy_trainer(
                    distilled_policy, train_collector_1, train_collector_2, 
                    distilled_policy_test_collector, distilled_oplicy_training_epoch,
                    args.step_per_epoch // 5, args.collect_per_step, args.test_num,
                    args.batch_size, stop_fn=stop_fn, save_fn=save_fn, writer=writer)
        if __name__ == '__main__':

            pprint.pprint(result)
            # Let's watch its performance!
            distilled_policy.eval()
            collector = Collector(distilled_policy, env)
            result = collector.collect(n_episode=1, render=args.render)
            print(f'Final reward: {result["rew"]}, length: {result["len"]}')