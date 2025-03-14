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

try:
    from slbo.misc import logger
except ImportError:
    from stable_baselines import logger


def virtual_training(config,model,model_buffer,virt_envs,policy_buffer,actor,critic,episode_rewards_virtual,episode_lengths_virtual,agent,start,writer,epoch,device,dynamics,normalizers,save_dir):
    for i in range(config.slbo.num_iters):
        logger.log('Updating the model  - iter {:02d}'.format(i + 1))
        losses = model.update(model_buffer)

        logger.log('Updating the policy - iter {:02d}'.format(i + 1))
        for _ in tqdm.tqdm(range(config.slbo.num_policy_updates)):
            # collect data in the virtual env
            if config.slbo.start_strategy == 'buffer':
                virt_envs.reset()
                initial_states = next(model_buffer.get_batch_generator(config.env.num_virtual_envs))['states']
                virt_envs.set_state(initial_states.cpu().numpy())
            elif config.slbo.start_strategy == 'reset':
                initial_states = virt_envs.reset()
            policy_buffer.states[0].copy_(initial_states)
            for step in range(config.trpo.num_env_steps):
                with torch.no_grad():
                    unscaled_actions, action_log_probs, dist_entropy, *_ = actor.act(policy_buffer.states[step])
                    values = critic(policy_buffer.states[step])

                # virtual_envs deal with action rescaling internally
                states, rewards, dones, infos = virt_envs.step(unscaled_actions)

                mask = torch.tensor([[0.0] if done else [1.0] for done in dones], dtype=torch.float32)
                bad_mask = torch.tensor([[0.0] if 'TimeLimit.truncated' in info.keys() else [1.0] for info in infos],
                                        dtype=torch.float32)
                policy_buffer.insert(states=states, actions=unscaled_actions, action_log_probs=action_log_probs,
                                        values=values, rewards=rewards, masks=mask, bad_masks=bad_mask)

                episode_rewards_virtual.extend([info['episode']['r'] for info in infos if 'episode' in info.keys()])
                episode_lengths_virtual.extend([info['episode']['l'] for info in infos if 'episode' in info.keys()])

            with torch.no_grad():
                next_value = critic(policy_buffer.states[-1])
                policy_buffer.compute_returns(next_value)
            losses.update(agent.update(policy_buffer))

        if (i + 1) % config.trpo.log_interval == 0 and len(episode_rewards_virtual) > 0:
            log_info = [('perf/ep_rew_virtual', np.mean(episode_rewards_virtual)),
                        ('perf/ep_len_virtual', np.mean(episode_lengths_virtual)),
                        ]
            for loss_name, loss_value in losses.items():
                log_info.append(('loss/' + loss_name, loss_value))
            log_info.append(('time_elapsed', time.time() - start))
            log_and_write(logger, writer, log_info, global_step=epoch * config.slbo.num_iters + i)

    if (epoch + 1) % config.save_freq == 0:
        torch.save({'actor': actor.state_dict(),
                    'critic': critic.state_dict(),
                    'dynamics': dynamics.state_dict(),
                    'normalizers': normalizers.state_dict()},
                    os.path.join(save_dir, 'actor_critic_dynamics_epoch{}.pt'.format(epoch)))
        logger.info('Saved model to {}'.format(os.path.join(save_dir, 'models_epoch{}.pt'.format(epoch))))

    if (epoch + 1) % config.eval_freq == 0:
        episode_rewards_real_eval, episode_lengths_real_eval = \
            evaluate(actor, config.env.env_name, config.seed, 10, None, device, config.env.max_episode_steps,
                        norm_reward=False, norm_obs=False, obs_rms=None, test=False)
        log_info = [('perf/ep_rew_real_eval', np.mean(episode_rewards_real_eval)),
                    ('perf/ep_len_real_eval', np.mean(episode_lengths_real_eval))]
        log_and_write(logger, writer, log_info, global_step=(epoch + 1) * config.slbo.num_iters)


# noinspection DuplicatedCode
def main():
    config, hparam_dict = Config('slbo_config.yaml')
    assert config.mf_algo == 'trpo'

    import datetime
    current_time = datetime.datetime.now().strftime('%b%d_%H%M%S')
    log_dir = os.path.join(config.proj_dir, config.result_dir, current_time, 'log')
    eval_log_dir = os.path.join(config.proj_dir, config.result_dir, current_time, 'log_eval')
    save_dir = os.path.join(config.proj_dir, config.result_dir, current_time, 'save')
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(eval_log_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    writer.add_hparams(hparam_dict, metric_dict={})

    logger.info('Hyperparms:')
    for key, value in hparam_dict.items():
        logger.log('{:30s}: {}'.format(key, value))
    # save current version of code
    shutil.copytree(config.proj_dir, save_dir + '/code', ignore=shutil.ignore_patterns('result', 'data', 'ref'))

    device = torch.device('cuda' if config.use_cuda else 'cpu')

    # make placeholder envs to query state dim and action dim etc
    ph_envs = make_vec_envs(config.env.env_name, config.seed, config.env.num_real_envs, config.env.gamma, log_dir, device,
                                allow_early_resets=True, norm_reward=False, norm_obs=False, test=False,
                                max_episode_steps=config.env.max_episode_steps,goal=0)

    lo, hi = torch.tensor(ph_envs.action_space.low, device=device), \
                torch.tensor(ph_envs.action_space.high, device=device)

    state_dim = ph_envs.observation_space.shape[0]
    action_space = ph_envs.action_space
    action_dim = action_space.shape[0]

    normalizers = Normalizers(action_dim, state_dim)
    normalizers.to(device)

    dynamics = Dynamics(state_dim, action_dim, config.slbo.dynamics_hidden_dims, normalizer=normalizers)
    dynamics.to(device)

    num_task = 2
    # for cheetah
    #task_goals = np.linspace(0.0,2.0,num_task)
    # For swimmer
    task_goals = np.linspace(0.5,1.5,num_task)
    # For inverted pendulum
    #task_goals = np.linspace(8,12,num_task)
    all_returns = []

    for task_goal in task_goals:
        task_returns = []

        noise = OUNoise(ph_envs.action_space, config.ou_noise.theta, config.ou_noise.sigma)

        # norm_obs is done by the Normalizer and norm_reward does not affect model learning
        real_envs = make_vec_envs(config.env.env_name, config.seed, config.env.num_real_envs, config.env.gamma, log_dir, device,
                                allow_early_resets=True, norm_reward=False, norm_obs=False, test=False,
                                max_episode_steps=config.env.max_episode_steps,
                                goal = task_goal)

        if config.mf_algo == 'ppo':
            actor_critic = ActorCritic(state_dim, action_space,
                                    actor_hidden_dims=config.ppo.actor_hidden_dims,
                                    critic_hidden_dims=config.ppo.critic_hidden_dims)
            actor_critic.to(device)

            agent = PPO(actor_critic, config.ppo_clip_param, config.ppo_num_grad_epochs, config.ppo_batch_size,
                        config.ppo_value_loss_coef,
                        config.ppo_entropy_loss_coef, lr=config.ppo_lr, max_grad_norm=config.ppo_max_grad_norm)
            noise_wrapped_actor = noise.wrap(actor_critic)
            mf_algo_config = config.ppo
            raise NotImplemented

        elif config.mf_algo == 'trpo':
            actor = Actor(state_dim, action_space, hidden_dims=config.trpo.actor_hidden_dims,
                        state_normalizer=normalizers.state_normalizer,
                        use_limited_entropy=config.trpo.use_limited_ent_actor)
            critic = VCritic(state_dim, hidden_dims=config.trpo.critic_hidden_dims,
                            state_normalizer=normalizers.state_normalizer)
            actor.to(device)
            critic.to(device)
            agent = TRPO(actor, critic, entropy_coef=config.trpo.entropy_coef, l2_reg_coef=config.trpo.l2_reg_coef,
                        verbose=config.verbose)
            noise_wrapped_actor = noise.wrap(actor)
            mf_algo_config = config.trpo

        # noinspection PyUnboundLocalVariable
        virt_envs = make_vec_virtual_envs(config.env.env_name, dynamics, config.seed, config.env.num_virtual_envs,
                                        config.env.gamma, device, allow_early_resets=True,
                                        max_episode_steps=config.env.max_episode_steps,
                                        norm_reward=mf_algo_config.norm_reward,
                                        goal = task_goal)
        # noinspection PyUnboundLocalVariable
        policy_buffer = \
            OnPolicyBuffer(mf_algo_config.num_env_steps, config.env.num_virtual_envs, real_envs.observation_space.shape,
                        action_space, use_gae=mf_algo_config.use_gae, gae_lambda=mf_algo_config.gae_lambda,
                        gamma=config.env.gamma, use_proper_time_limits=mf_algo_config.use_proper_time_limits)
        policy_buffer.to(device)

        model = SLBO(dynamics, normalizers, config.slbo.batch_size, num_rollout_steps=config.slbo.num_rollout_steps,
                    num_updates=config.slbo.num_model_updates, lr=config.slbo.lr, l2_reg_coef=config.slbo.l2_reg_coef)

        model_buffer = OffPolicyBuffer(config.slbo.buffer_size, config.env.num_real_envs, state_dim, action_dim)
        model_buffer.to(device)

        if config.model_load_path is not None:
            raise NotImplemented
            actor_state_dict, critic_state_dict,  dynamics_state_dict, normalizers_state_dict \
                = itemgetter(*['actor', 'critic', 'dynamics', 'normalizers'])(torch.load(config.model_load_path))
            # noinspection PyUnboundLocalVariable
            actor.load_state_dict(actor_state_dict)
            dynamics.load_state_dict(dynamics_state_dict)
            # noinspection PyUnboundLocalVariable
            critic.load_state_dict(critic_state_dict)
            normalizers.load_state_dict(normalizers_state_dict)

        if config.buffer_load_path is not None:
            raise NotImplemented
            model_buffer.load(config.buffer_load_path)

        episode_rewards_real = deque(maxlen=10)
        episode_lengths_real = deque(maxlen=10)
        episode_rewards_virtual = deque(maxlen=10)
        episode_lengths_virtual = deque(maxlen=10)

        start = time.time()

        for epoch in range(config.slbo.num_epochs):

            logger.info('Epoch {}:'.format(epoch + 1))

            if not config.slbo.use_prev_data:
                model_buffer.clear()

            # reset current_buffer and env
            cur_model_buffer = OffPolicyBuffer(config.slbo.num_env_steps, config.env.num_real_envs, state_dim, action_dim)
            cur_model_buffer.to(device)
            states = real_envs.reset()

            for step in range(config.slbo.num_env_steps):
                # noinspection PyUnboundLocalVariable
                unscaled_actions = noise_wrapped_actor.act(states)[0]
                # for mujoco envs, actions is equal to unscaled_actions
                actions = lo + (unscaled_actions + 1.) * 0.5 * (hi - lo)

                next_states, rewards, dones, infos = real_envs.step(actions)
                masks = torch.tensor([[0.0] if done else [1.0] for done in dones])
                bad_masks = torch.tensor([[0.0] if 'TimeLimit.truncated' in info.keys() else [1.0] for info in infos])
                cur_model_buffer.insert(states, unscaled_actions, rewards, next_states, masks, bad_masks)

                states = next_states
                episode_rewards_real.extend([info['episode']['r'] for info in infos if 'episode' in info])
                episode_lengths_real.extend([info['episode']['l'] for info in infos if 'episode' in info])

                # mask out last collected tuples
                cur_model_buffer.bad_masks[-1, :, :] = 0.

            task_returns.append(np.mean(episode_rewards_real))
            serial_env_steps = (epoch + 1) * config.slbo.num_env_steps
            total_env_steps = serial_env_steps * config.env.num_real_envs

            model_buffer.add_buffer(cur_model_buffer)

            if (epoch + 1) % config.slbo.log_interval == 0 and len(episode_rewards_real) > 0:
                log_info = [('serial_timesteps', serial_env_steps), ('total_timesteps', total_env_steps),
                            ('perf/ep_rew_real', np.mean(episode_rewards_real)),
                            ('perf/ep_len_real', np.mean(episode_lengths_real)),
                            ('time_elapsed', time.time() - start)]
                log_and_write(logger, writer, log_info, global_step=epoch * config.slbo.num_iters)

            if epoch == 0:
                samples = next(model_buffer.get_sequential_batch_generator(100, config.slbo.num_rollout_steps))
                states, next_states, masks, bad_masks = itemgetter(*['states', 'next_states', 'masks', 'bad_masks'])(samples)
                for i in range(config.slbo.num_rollout_steps - 1):
                    assert torch.allclose(states[:, i + 1] * masks[:, i] * bad_masks[:, i],
                                        next_states[:, i] * masks[:, i] * bad_masks[:, i])
            else:
                virtual_training(config,model,model_buffer,virt_envs,policy_buffer,actor,critic,episode_rewards_virtual,episode_lengths_virtual,agent,start,writer,epoch,device,dynamics,normalizers,save_dir)

            if epoch == 50:
                # noinspection PyUnboundLocalVariable
                agent.entropy_coef = 0.

            normalizers.state_normalizer.update(cur_model_buffer.states.reshape([-1, state_dim]))
            # normalizers.action_normalizer.update(cur_model_buffer.actions)
            # FIXME: rule out incorrect tuples
            normalizers.diff_normalizer.update((cur_model_buffer.next_states - cur_model_buffer.states).
                                            reshape([-1, state_dim]))

            del cur_model_buffer

            virtual_training(config,model,model_buffer,virt_envs,policy_buffer,actor,critic,episode_rewards_virtual,episode_lengths_virtual,agent,start,writer,epoch,device,dynamics,normalizers,save_dir)
        
        all_returns.append(task_returns)
    
    np.save('all_returns.npy',all_returns)




if __name__ == '__main__':
    main()
