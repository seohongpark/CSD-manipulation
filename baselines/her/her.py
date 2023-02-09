import numpy as np
import random
from baselines.common.schedules import PiecewiseSchedule
from scipy.stats import rankdata


def make_sample_her_transitions(replay_strategy, replay_k, reward_fun, et_w_schedule):

    if (replay_strategy == 'future') or (replay_strategy == 'final'):
        future_p = 1 - (1. / (1 + replay_k))
    else:
        future_p = 0
    et_w_scheduler = PiecewiseSchedule(endpoints=et_w_schedule)

    def _sample_her_transitions(ddpg, ir, episode_batch, batch_size_in_transitions, sk_r_scale, t):
        """episode_batch is {key: array(buffer_size x T x dim_key)}
        """
        T = episode_batch['u'].shape[1]
        rollout_batch_size = episode_batch['u'].shape[0]
        batch_size = batch_size_in_transitions

        # Select which episodes and time steps to use.
        episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)
        # t_samples = np.random.randint(T, size=batch_size)
        t_samples = []
        for i in range(batch_size):
            max_t = episode_batch['myv'][episode_idxs[i]].sum().astype(int)
            t_sample = np.random.randint(max_t)
            t_samples.append(t_sample)
        t_samples = np.array(t_samples)

        # calculate intrinsic rewards
        sk_trans = np.zeros([episode_idxs.shape[0], 1])
        if ir:
            o_curr = episode_batch['o'][episode_idxs, t_samples].copy()
            o_curr = np.reshape(o_curr, (o_curr.shape[0], 1, o_curr.shape[-1]))
            o_next = episode_batch['o'][episode_idxs, t_samples+1].copy()
            o_next = np.reshape(o_next, (o_next.shape[0], 1, o_next.shape[-1]))

            o = episode_batch['o'][episode_idxs, t_samples].copy()
            o2 = episode_batch['o_2'][episode_idxs, t_samples].copy()
            z = episode_batch['z'][episode_idxs, t_samples].copy()
            u = episode_batch['u'][episode_idxs, t_samples].copy()
            if sk_r_scale > 0:
                sk_r = ddpg.run_sk(o, z, o2, u)
                sk_trans = sk_r.copy()
        # #

        transitions = {}
        for key in episode_batch.keys():
            if not (key == 's' or key == 'p'):
                transitions[key] = episode_batch[key][episode_idxs, t_samples].copy()
            else:
                transitions[key] = episode_batch[key][episode_idxs].copy()
        transitions['s'] = transitions['s'].flatten().copy()

        her_indexes = np.where(np.random.uniform(size=batch_size) < future_p)
        future_offset = np.random.uniform(size=batch_size) * (T - t_samples)
        future_offset = future_offset.astype(int)
        future_t = (t_samples + 1 + future_offset)[her_indexes]

        if replay_strategy == 'final':
            future_t[:] = T

        future_ag = episode_batch['ag'][episode_idxs[her_indexes], future_t]
        transitions['g'][her_indexes] = future_ag

        info = {}
        for key, value in transitions.items():
            if key.startswith('info_'):
                info[key.replace('info_', '')] = value

        reward_params = {k: transitions[k] for k in ['ag_2', 'g']}
        reward_params['info'] = info
        # transitions['r'] = reward_fun(**reward_params)
        transitions['r'] = transitions['myr']

        transitions = {k: transitions[k].reshape(batch_size, *transitions[k].shape[1:])
                       for k in transitions.keys()}

        if ir:
            transitions['s'] = sk_trans.flatten().copy()

        transitions['s_w'] = 1.0
        transitions['r_w'] = 1.0
        transitions['e_w'] = et_w_scheduler.value(t)

        assert(transitions['u'].shape[0] == batch_size_in_transitions)

        return transitions

    return _sample_her_transitions
