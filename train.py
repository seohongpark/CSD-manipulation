import os
import pathlib
import sys
import time
from collections import defaultdict

import click
import gym
import numpy as np
import json
from mpi4py import MPI

from baselines import logger
from baselines.common import set_global_seeds
from baselines.common.mpi_moments import mpi_moments
import baselines.her.experiment.config as config
from baselines.her.rollout import RolloutWorker
from baselines.her.util import mpi_fork, snn

import os.path as osp
import tempfile
import datetime
from baselines.her.util import (dumpJson, loadJson, save_video, save_weight, load_weight)
import pickle
import tensorflow as tf
import wandb

from utils import FigManager, plot_trajectories, setup_evaluation, record_video, draw_2d_gaussians, RunningMeanStd, \
    get_option_colors

g_start_time = int(datetime.datetime.now().timestamp())


def mpi_average(value):
    if value == []:
        value = [0.]
    if not isinstance(value, list):
        value = [value]
    return mpi_moments(np.array(value))[0]

def sample_skill(num_skills, rollout_batch_size, use_skill_n=None, skill_type='discrete'):
    # sample skill z

    if skill_type == 'discrete':
        z_s = np.random.randint(0, num_skills, rollout_batch_size)
        if use_skill_n:
            use_skill_n = use_skill_n - 1
            z_s.fill(use_skill_n)

        z_s_onehot = np.zeros([rollout_batch_size, num_skills])
        z_s = np.array(z_s).reshape(rollout_batch_size, 1)
        for i in range(rollout_batch_size):
            z_s_onehot[i, z_s[i]] = 1
        return z_s, z_s_onehot
    else:
        z_s = np.zeros((rollout_batch_size, 1))
        z_s_onehot = np.random.randn(rollout_batch_size, num_skills)
        return z_s, z_s_onehot


def iod_eval(eval_dir, env_name, evaluator, video_evaluator, num_skills, skill_type, plot_repeats, epoch, goal_generation, n_random_trajectories):
    if goal_generation == 'Zero':
        generated_goal = np.zeros(evaluator.g.shape)
    else:
        generated_goal = False

    if env_name != 'Maze':
        # Video eval
        if skill_type == 'discrete':
            video_eval_options = np.eye(num_skills)
            if num_skills == 1:
                video_eval_options = np.ones((9, 1))
        else:
            if num_skills == 2:
                video_eval_options = []
                for dist in [4.5]:
                    for angle in [3, 2, 1, 4]:
                        video_eval_options.append([dist * np.cos(angle * np.pi / 4), dist * np.sin(angle * np.pi / 4)])
                video_eval_options.append([0, 0])
                for dist in [4.5]:
                    for angle in [0, 5, 6, 7]:
                        video_eval_options.append([dist * np.cos(angle * np.pi / 4), dist * np.sin(angle * np.pi / 4)])
                video_eval_options = np.array(video_eval_options)
            elif num_skills <= 5:
                video_eval_options = []
                for dist in [-4.5, -2.25, 2.25, 4.5]:
                    for dim in range(num_skills):
                        cur_option = [0] * num_skills
                        cur_option[dim] = dist
                        video_eval_options.append(cur_option)
                video_eval_options.append([0.] * num_skills)
                video_eval_options = np.array(video_eval_options)
            else:
                video_eval_options = np.random.randn(9, num_skills) * 4.5 / 1.25

        # Record each option twice
        video_eval_options = np.repeat(video_eval_options, 2, axis=0)

        if skill_type == 'continuous':
            video_eval_options = video_eval_options / 4.5 * 1.25
        video_evaluator.clear_history()
        video_evaluator.render = 'rgb_array'
        i = 0
        imgss = []
        while i < len(video_eval_options):
            z = video_eval_options[i:i + video_evaluator.rollout_batch_size]
            if len(z) != video_evaluator.rollout_batch_size:
                remainder = video_evaluator.rollout_batch_size - z.shape[0]
                z = np.concatenate([z, np.zeros((remainder, z.shape[1]))], axis=0)
            else:
                remainder = 0
            imgs, _ = video_evaluator.generate_rollouts(generated_goal=generated_goal, z_s_onehot=z)
            for j in range(len(imgs) - remainder):
                imgss.append(imgs[j])
                # filename = eval_dir + f'/videos/video_epoch_{epoch}_skill_{z[j]}.avi'
                # save_video(imgs[j], filename)
            i += video_evaluator.rollout_batch_size
        video_evaluator.render = False
        filename = eval_dir + f'/videos/video_epoch_{epoch}.mp4'
        record_video(filename, imgss)
        label = 'video'
        logger.record_tabular(label, (filename, label))

    # Plot eval
    if skill_type == 'discrete':
        eval_options = np.eye(num_skills)
        colors = np.arange(0, num_skills)
        eval_options = eval_options.repeat(plot_repeats, axis=0)
        colors = colors.repeat(plot_repeats, axis=0)
        num_evals = len(eval_options)
        eval_option_colors = []
        from matplotlib import cm
        cmap = 'tab10' if num_skills <= 10 else 'tab20'
        for i in range(num_evals):
            eval_option_colors.extend([cm.get_cmap(cmap)(colors[i])[:3]])
        eval_option_colors = np.array(eval_option_colors)
        random_eval_options = eval_options
        random_eval_option_colors = eval_option_colors
    else:
        random_eval_options = np.random.randn(n_random_trajectories, num_skills)
        random_eval_option_colors = get_option_colors(random_eval_options * 2)

    for cur_type in ['Random']:
        grips = []
        achs = []
        xzs = []
        yzs = []
        xyzs = []
        obs = []
        options = []
        infos = defaultdict(list)
        evaluator.clear_history()
        i = 0
        num_trajs = len(random_eval_options)
        cur_colors = random_eval_option_colors

        while i < num_trajs:
            z = random_eval_options[i:i + evaluator.rollout_batch_size]

            if len(z) != evaluator.rollout_batch_size:
                remainder = evaluator.rollout_batch_size - z.shape[0]
                z = np.concatenate([z, np.zeros((remainder, z.shape[1]))], axis=0)
            else:
                remainder = 0
            rollouts = evaluator.generate_rollouts(generated_goal=generated_goal, z_s_onehot=z)

            grip_coords = rollouts['o'][:, :, 0:2]
            if 'Kitchen' in env_name:
                target_coords = [23, 24, 25]
            else:
                target_coords = [3, 4, 5]
            ach_coords = rollouts['o'][:, :, [target_coords[0], target_coords[1]]]
            xz_coords = rollouts['o'][:, :, [target_coords[0], target_coords[2]]]
            yz_coords = rollouts['o'][:, :, [target_coords[1], target_coords[2]]]
            xyz_coords = rollouts['o'][:, :, target_coords]
            ob = rollouts['o'][:, :, :]
            grips.extend(grip_coords[:evaluator.rollout_batch_size - remainder])
            achs.extend(ach_coords[:evaluator.rollout_batch_size - remainder])
            xzs.extend(xz_coords[:evaluator.rollout_batch_size - remainder])
            yzs.extend(yz_coords[:evaluator.rollout_batch_size - remainder])
            xyzs.extend(xyz_coords[:evaluator.rollout_batch_size - remainder])
            obs.extend(ob[:evaluator.rollout_batch_size - remainder])
            options.extend(z[:evaluator.rollout_batch_size - remainder])
            if 'Kitchen' in env_name:
                for key, val in rollouts.items():
                    if not key.startswith('info_Task'):
                        continue
                    infos[key].extend(val[:, :, 0].max(axis=1))

            i += evaluator.rollout_batch_size

        for label, trajs in [(f'EvalOp__TrajPlotWithCFrom{cur_type}', achs), (f'EvalOp__GripPlotWithCFrom{cur_type}', grips),
                             (f'EvalOp__XzPlotWithCFrom{cur_type}', xzs), (f'EvalOp__YzPlotWithCFrom{cur_type}', yzs)]:
            with FigManager(label, epoch, eval_dir) as fm:
                if 'Fetch' in env_name:
                    plot_axis = [0, 2, 0, 2]
                elif env_name == 'Maze':
                    plot_axis = [-2, 2, -2, 2]
                elif 'Kitchen' in env_name:
                    plot_axis = [-3, 3, -3, 3]
                else:
                    plot_axis = None
                plot_trajectories(
                    trajs, cur_colors, plot_axis=plot_axis, ax=fm.ax
                )

        if cur_type == 'Random':
            coords = np.concatenate(xyzs, axis=0)
            coords = coords * 10
            uniq_coords = np.unique(np.floor(coords), axis=0)
            uniq_xy_coords = np.unique(np.floor(coords[:, :2]), axis=0)
            logger.record_tabular('Fetch/NumTrajs', len(xyzs))
            logger.record_tabular('Fetch/AvgTrajLen', len(coords) / len(xyzs) - 1)
            logger.record_tabular('Fetch/NumCoords', len(coords))
            logger.record_tabular('Fetch/NumUniqueXYZCoords', len(uniq_coords))
            logger.record_tabular('Fetch/NumUniqueXYCoords', len(uniq_xy_coords))
            if 'Kitchen' in env_name:
                for key, val in infos.items():
                    logger.record_tabular(f'Kitchen/{key[9:]}', np.minimum(1., np.max(val)))


def train(
        logdir, policy, rollout_worker, env_name,
        evaluator, video_evaluator, n_epochs, train_start_epoch, n_test_rollouts, n_cycles, n_batches, policy_save_interval,
        save_policies, num_cpu, collect_data, collect_video, goal_generation, num_skills, use_skill_n, batch_size,
        sk_r_scale,
        skill_type, plot_freq, plot_repeats, n_random_trajectories, sk_clip, et_clip, done_ground,
        **kwargs
):

    rank = MPI.COMM_WORLD.Get_rank()

    latest_policy_path = os.path.join(logger.get_dir(), 'policy_latest.pkl')
    best_policy_path = os.path.join(logger.get_dir(), 'policy_best.pkl')
    periodic_policy_path = os.path.join(logger.get_dir(), 'policy_{}.pkl')
    restore_info_path = os.path.join(logger.get_dir(), 'restore_info.pkl')

    with open(restore_info_path, 'wb') as f:
        pickle.dump(dict(
            dimo=policy.dimo,
            dimz=policy.dimz,
            dimg=policy.dimg,
            dimu=policy.dimu,
            hidden=policy.hidden,
            layers=policy.layers,
        ), f)

    logger.info("Training...")
    best_success_rate = -1
    t = 1
    start_time = time.time()
    cur_time = time.time()
    for epoch in range(n_epochs):
        # train
        episodes = []
        rollout_worker.clear_history()
        for cycle in range(n_cycles):
            z_s, z_s_onehot = sample_skill(num_skills, rollout_worker.rollout_batch_size, use_skill_n, skill_type=skill_type)

            if goal_generation == 'Zero':
                generated_goal = np.zeros(rollout_worker.g.shape)
            else:
                generated_goal = False

            if train_start_epoch <= epoch:
                episode = rollout_worker.generate_rollouts(generated_goal=generated_goal, z_s_onehot=z_s_onehot)
            else:
                episode = rollout_worker.generate_rollouts(generated_goal=generated_goal, z_s_onehot=z_s_onehot, random_action=True)
            episodes.append(episode)
            policy.store_episode(episode)

            for batch in range(n_batches):
                t = epoch
                if train_start_epoch <= epoch:
                    policy.train(t)

                # train skill discriminator
                if sk_r_scale > 0:
                    o_s = policy.buffer.buffers['o'][0: policy.buffer.current_size]
                    o2_s = policy.buffer.buffers['o'][0: policy.buffer.current_size][:, 1:, :]
                    z_s = policy.buffer.buffers['z'][0: policy.buffer.current_size]
                    u_s = policy.buffer.buffers['u'][0: policy.buffer.current_size]
                    T = z_s.shape[-2]
                    episode_idxs = np.random.randint(0, policy.buffer.current_size, batch_size)
                    t_samples = np.random.randint(T, size=batch_size)
                    o_s_batch = o_s[episode_idxs, t_samples]
                    o2_s_batch = o2_s[episode_idxs, t_samples]
                    z_s_batch = z_s[episode_idxs, t_samples]
                    u_s_batch = u_s[episode_idxs, t_samples]
                    if train_start_epoch <= epoch:
                        policy.train_sk(o_s_batch, z_s_batch, o2_s_batch, u_s_batch)
                    if policy.dual_dist != 'l2':
                        add_dict = dict()
                        policy.train_sk_dist(o_s_batch, z_s_batch, o2_s_batch, add_dict)
                # #

            if train_start_epoch <= epoch:
                policy.update_target_net()

        if collect_data and (rank == 0):
            dumpJson(logdir, episodes, epoch, rank)

        if plot_freq != 0 and epoch % plot_freq == 0:
            iod_eval(logdir, env_name, evaluator, video_evaluator, num_skills, skill_type, plot_repeats, epoch, goal_generation, n_random_trajectories)

        # test
        evaluator.clear_history()
        for _ in range(n_test_rollouts):
            z_s, z_s_onehot = sample_skill(num_skills, evaluator.rollout_batch_size, use_skill_n, skill_type=skill_type)
            evaluator.generate_rollouts(generated_goal=False, z_s_onehot=z_s_onehot)

        # record logs
        logger.record_tabular('time/total_time', time.time() - start_time)
        logger.record_tabular('time/epoch_time', time.time() - cur_time)
        cur_time = time.time()
        logger.record_tabular('epoch', epoch)
        for key, val in evaluator.logs('test'):
            logger.record_tabular(key, mpi_average(val))
        if n_cycles != 0:
            for key, val in rollout_worker.logs('train'):
                logger.record_tabular(key, mpi_average(val))
            for key, val in policy.logs(is_policy_training=(train_start_epoch <= epoch)):
                logger.record_tabular(key, mpi_average(val))

        logger.record_tabular('best_success_rate', best_success_rate)
        
        if rank == 0:
            logger.dump_tabular()

        # save the policy if it's better than the previous ones
        success_rate = mpi_average(evaluator.current_success_rate())
        if rank == 0 and success_rate >= best_success_rate and save_policies:
            best_success_rate = success_rate
            logger.info('New best success rate: {}. Saving policy to {} ...'.format(best_success_rate, best_policy_path))
            evaluator.save_policy(best_policy_path)
            evaluator.save_policy(latest_policy_path)
        if rank == 0 and policy_save_interval > 0 and epoch % policy_save_interval == 0 and save_policies:
            policy_path = periodic_policy_path.format(epoch)
            logger.info('Saving periodic policy to {} ...'.format(policy_path))
            evaluator.save_policy(policy_path)

        # make sure that different threads have different seeds
        local_uniform = np.random.uniform(size=(1,))
        root_uniform = local_uniform.copy()
        MPI.COMM_WORLD.Bcast(root_uniform, root=0)
        if rank != 0:
            assert local_uniform[0] != root_uniform[0]


def launch(
        run_group, env_name, n_epochs, train_start_epoch, num_cpu, seed, replay_strategy, policy_save_interval, clip_return, binding, logging,
        num_skills, version, n_cycles, note, skill_type, plot_freq, plot_repeats, n_random_trajectories,
        sk_r_scale, et_r_scale, sk_clip, et_clip, done_ground,
        max_path_length, hidden, layers, rollout_batch_size, n_batches, polyak, spectral_normalization,
        dual_reg, dual_init_lambda, dual_lam_opt, dual_slack, dual_dist,
        inner, algo, random_eps, noise_eps, lr, sk_lam_lr, buffer_size, algo_name,
        load_weight, override_params={}, save_policies=True,
):
    tf.compat.v1.disable_eager_execution()

    # Fork for multi-CPU MPI implementation.
    if num_cpu > 1:
        whoami = mpi_fork(num_cpu, binding)
        if whoami == 'parent':
            sys.exit(0)
        import baselines.common.tf_util as U
        U.single_threaded_session().__enter__()
    rank = MPI.COMM_WORLD.Get_rank()

    # Configure logging

    if logging:
        logdir = ''
        logdir += f'logs/{run_group}/'
        logdir += f'sd{seed:03d}_'
        if 'SLURM_JOB_ID' in os.environ:
            logdir += f's_{os.environ["SLURM_JOB_ID"]}.'
        if 'SLURM_PROCID' in os.environ:
            logdir += f'{os.environ["SLURM_PROCID"]}.'
        if 'SLURM_RESTART_COUNT' in os.environ:
            logdir += f'rs_{os.environ["SLURM_RESTART_COUNT"]}.'
        logdir += f'{g_start_time}_'
        logdir += str(env_name)
        logdir += '_ns' + str(num_skills)
        logdir += '_sn' + str(spectral_normalization)
        logdir += '_dr' + str(dual_reg)
        logdir += '_in' + str(inner)
        logdir += '_sk' + str(sk_r_scale)
        logdir += '_et' + str(et_r_scale)
    else:
        logdir = osp.join(tempfile.gettempdir(),
            datetime.datetime.now().strftime("openai-%Y-%m-%d-%H-%M-%S-%f"))

    if rank == 0:
        if logdir or logger.get_dir() is None:
            logger.configure(dir=logdir)
    else:
        logger.configure() # use temp folder for other rank
    logdir = logger.get_dir()
    assert logdir is not None
    os.makedirs(logdir, exist_ok=True)

    # Seed everything.
    rank_seed = seed + 1000000 * rank
    set_global_seeds(rank_seed)

    # Prepare params.
    params = config.DEFAULT_PARAMS
    params['env_name'] = env_name
    params['seed'] = seed
    params['replay_strategy'] = replay_strategy
    params['binding'] = binding
    params['max_timesteps'] = n_epochs * params['n_cycles'] *  params['n_batches'] * num_cpu
    params['version'] = version
    params['n_cycles'] = n_cycles
    params['num_cpu'] = num_cpu
    params['note'] = note or params['note']
    if note:
        with open('params/'+note+'.json', 'r') as file:
            override_params = json.loads(file.read())
            params.update(**override_params)

    ##########################################33
    params['num_skills'] = num_skills
    params['skill_type'] = skill_type
    params['plot_freq'] = plot_freq
    params['plot_repeats'] = plot_repeats
    params['n_random_trajectories'] = n_random_trajectories
    if sk_r_scale is not None:
        params['sk_r_scale'] = sk_r_scale
    if et_r_scale is not None:
        params['et_r_scale'] = et_r_scale
    params['sk_clip'] = sk_clip
    params['et_clip'] = et_clip
    params['done_ground'] = done_ground
    params['max_path_length'] = max_path_length
    params['hidden'] = hidden
    params['layers'] = layers
    params['rollout_batch_size'] = rollout_batch_size
    params['n_batches'] = n_batches
    params['polyak'] = polyak
    params['spectral_normalization'] = spectral_normalization
    params['dual_reg'] = dual_reg
    params['dual_init_lambda'] = dual_init_lambda
    params['dual_lam_opt'] = dual_lam_opt
    params['dual_slack'] = dual_slack
    params['dual_dist'] = dual_dist
    params['inner'] = inner
    params['algo'] = algo
    params['random_eps'] = random_eps
    params['noise_eps'] = noise_eps
    params['lr'] = lr
    params['sk_lam_lr'] = sk_lam_lr
    params['buffer_size'] = buffer_size
    params['algo_name'] = algo_name
    params['train_start_epoch'] = train_start_epoch

    if load_weight is not None:
        params['load_weight'] = load_weight

    if params['load_weight']:
        if type(params['load_weight']) is list:
            params['load_weight'] = params['load_weight'][seed]
        import glob
        base = os.path.splitext(params['load_weight'])[0]
        policy_path = base + '_weight.pkl'
        policy_path = glob.glob(policy_path)[0]
        policy_weight_file = open(policy_path, 'rb')
        pretrain_weights = pickle.load(policy_weight_file)
        policy_weight_file.close()
    else:
        pretrain_weights = None

    if env_name in config.DEFAULT_ENV_PARAMS:
        params.update(config.DEFAULT_ENV_PARAMS[env_name])  # merge env-specific parameters in
    with open(os.path.join(logger.get_dir(), 'params.json'), 'w') as f:
        json.dump(params, f)
    params = config.prepare_params(params)

    exp_name = logdir.split('/')[-1]

    if 'WANDB_API_KEY' in os.environ:
        wandb.init(project="", entity="", group=run_group, name=exp_name, config=params)  # Fill out this

    def make_env():
        if env_name == 'Maze':
            from envs.maze_env import MazeEnv
            env = MazeEnv(n=max_path_length)
        elif env_name == 'Kitchen':
            from d4rl_alt.kitchen.kitchen_envs import KitchenMicrowaveKettleLightTopLeftBurnerV0Custom
            from gym.wrappers.time_limit import TimeLimit
            env = KitchenMicrowaveKettleLightTopLeftBurnerV0Custom(control_mode='end_effector')
            max_episode_steps = max_path_length
            env = TimeLimit(env, max_episode_steps=max_episode_steps)
        else:
            env = gym.make(env_name)
            if 'max_path_length' in params:
                env = env.env
                from gym.wrappers.time_limit import TimeLimit
                max_episode_steps = params['max_path_length']
                env = TimeLimit(env, max_episode_steps=max_episode_steps)

        return env

    params['make_env'] = make_env
    ##########################################################

    config.log_params(params, logger=logger)

    dims = config.configure_dims(params)
    policy = config.configure_ddpg(dims=dims, params=params, pretrain_weights=pretrain_weights, clip_return=clip_return)

    render = False
    if params['collect_video']:
        render = 'rgb_array'

    rollout_params = {
        'exploit': False,
        'use_target_net': False,
        'use_demo_states': True,
        'compute_Q': False,
        'T': params['T'],
        'render': render,
    }

    eval_params = {
        'exploit': True,
        'use_target_net': params['test_with_polyak'],
        'use_demo_states': False,
        'compute_Q': True,
        'T': params['T'],
    }

    for name in ['T', 'rollout_batch_size', 'gamma', 'noise_eps', 'random_eps']:
        rollout_params[name] = params[name]
        eval_params[name] = params[name]


    rollout_worker = RolloutWorker(make_env, policy, dims, logger, **rollout_params)
    rollout_worker.seed(rank_seed)

    evaluator = RolloutWorker(make_env, policy, dims, logger, **eval_params)
    evaluator.seed(rank_seed)

    video_evaluator = RolloutWorker(make_env, policy, dims, logger, **dict(eval_params, rollout_batch_size=1))
    video_evaluator.seed(rank_seed)

    train(
        logdir=logdir, policy=policy, rollout_worker=rollout_worker, env_name=env_name,
        evaluator=evaluator, video_evaluator=video_evaluator, n_epochs=n_epochs, train_start_epoch=train_start_epoch, n_test_rollouts=params['n_test_rollouts'], n_cycles=params['n_cycles'], n_batches=params['n_batches'], policy_save_interval=policy_save_interval, save_policies=save_policies, num_cpu=num_cpu, collect_data=params['collect_data'], collect_video=params['collect_video'], goal_generation=params['goal_generation'], num_skills=params['num_skills'], use_skill_n=params['use_skill_n'], batch_size=params['_batch_size'], sk_r_scale=params['sk_r_scale'],
        skill_type=params['skill_type'], plot_freq=params['plot_freq'], plot_repeats=params['plot_repeats'], n_random_trajectories=params['n_random_trajectories'], sk_clip=params['sk_clip'], et_clip=params['et_clip'], done_ground=params['done_ground'],
    )


@click.command()
@click.option('--run_group', type=str, default='EXP')
@click.option('--env_name', type=click.Choice(['FetchPush-v1', 'FetchSlide-v1', 'FetchPickAndPlace-v1', 'Maze', 'Kitchen']))
@click.option('--n_epochs', type=int, default=50, help='the number of training epochs to run')
@click.option('--train_start_epoch', type=int, default=0)
@click.option('--num_cpu', type=int, default=1, help='the number of CPU cores to use (using MPI)')
@click.option('--seed', type=int, default=0, help='the random seed used to seed both the environment and the training code')
@click.option('--policy_save_interval', type=int, default=1, help='the interval with which policy pickles are saved. If set to 0, only the best and latest policy will be pickled.')
@click.option('--n_cycles', type=int, default=50, help='n_cycles')
@click.option('--replay_strategy', type=click.Choice(['future', 'final', 'none']), default='future', help='replay strategy to be used.')
@click.option('--clip_return', type=int, default=0, help='whether or not returns should be clipped')
@click.option('--binding', type=click.Choice(['none', 'core']), default='core', help='configure mpi using bind-to none or core.')
@click.option('--logging', type=bool, default=False, help='whether or not logging')
@click.option('--num_skills', type=int, default=5)
@click.option('--version', type=int, default=0, help='version')
@click.option('--note', type=str, default=None, help='unique notes')

@click.option('--skill_type', type=str, default='discrete')
@click.option('--plot_freq', type=int, default=1)
@click.option('--plot_repeats', type=int, default=1)
@click.option('--n_random_trajectories', type=int, default=200)
@click.option('--sk_r_scale', type=float, default=None)
@click.option('--et_r_scale', type=float, default=None)
@click.option('--sk_clip', type=int, default=1)
@click.option('--et_clip', type=int, default=1)
@click.option('--done_ground', type=int, default=0)
@click.option('--max_path_length', type=int, default=50)
@click.option('--hidden', type=int, default=256)
@click.option('--layers', type=int, default=3)
@click.option('--rollout_batch_size', type=int, default=2)
@click.option('--n_batches', type=int, default=40)
@click.option('--polyak', type=float, default=0.95)
@click.option('--spectral_normalization', type=int, default=0)
@click.option('--dual_reg', type=int, default=0)
@click.option('--dual_init_lambda', type=float, default=1)
@click.option('--dual_lam_opt', type=str, default='adam')
@click.option('--dual_slack', type=float, default=0.)
@click.option('--dual_dist', type=str, default='l2')
@click.option('--inner', type=int, default=0)
@click.option('--algo', type=str, default='csd')
@click.option('--random_eps', type=float, default=0.3)
@click.option('--noise_eps', type=float, default=0.2)
@click.option('--lr', type=float, default=0.001)
@click.option('--sk_lam_lr', type=float, default=0.001)
@click.option('--buffer_size', type=int, default=1000000)
@click.option('--algo_name', type=str, default=None)  # Only for logging, not used
@click.option('--load_weight', type=str, default=None)
def main(**kwargs):
    launch(**kwargs)

if __name__ == '__main__':
    main()
