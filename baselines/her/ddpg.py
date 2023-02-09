from collections import OrderedDict, defaultdict
import numpy as np
import tensorflow as tf
# from tensorflow.contrib.staging import StagingArea
from tensorflow.python.ops.data_flow_ops import StagingArea
from baselines import logger
from baselines.her.util import (
    import_function, store_args, flatten_grads, transitions_in_episode_batch, save_weight, load_weight)
from baselines.her.normalizer import Normalizer
from baselines.her.replay_buffer import ReplayBuffer
from baselines.common.mpi_adam import MpiAdam
from baselines.common.mpi_sgd import MpiSgd
import baselines.common.tf_util as U
import json
from collections import deque


def dims_to_shapes(input_dims):
    return {key: tuple([val]) if val > 0 else tuple() for key, val in input_dims.items()}


class DDPG(object):
    @store_args
    def __init__(
            self, input_dims, buffer_size, hidden, layers, network_class_actor_critic, network_class_discriminator,
            polyak, batch_size, Q_lr, pi_lr, sk_lr, r_scale, sk_r_scale, et_r_scale, norm_eps,
            norm_clip, max_u, action_l2, clip_obs, scope, T, rollout_batch_size, subtract_goals, relative_goals,
            clip_pos_returns, clip_return, sample_transitions, gamma, env_name, max_timesteps, pretrain_weights,
            finetune_pi, sac, reuse=False, history_len=10000,
            skill_type='discrete', sk_clip=1, et_clip=1, done_ground=0, obj_prior=0, spectral_normalization=0,
            dual_reg=0, dual_init_lambda=1., dual_lam_opt='adam', dual_slack=0., dual_dist='l2',
            inner=0, algo='csd', sk_lam_lr=0.001,
            **kwargs
    ):
        if self.clip_return is None:
            self.clip_return = np.inf

        self.create_actor_critic = import_function(self.network_class_actor_critic)
        self.create_discriminator = import_function(self.network_class_discriminator)

        input_shapes = dims_to_shapes(self.input_dims)
        self.dimo = self.input_dims['o']
        self.dimz = self.input_dims['z']
        self.dimg = self.input_dims['g']
        self.dimu = self.input_dims['u']
        self.skill_type = skill_type
        self.sk_clip = sk_clip
        self.et_clip = et_clip
        self.done_ground = done_ground
        self.obj_prior = obj_prior
        self.spectral_normalization = spectral_normalization
        self.dual_reg = dual_reg
        self.dual_init_lambda = dual_init_lambda
        self.dual_lam_opt = dual_lam_opt
        self.dual_slack = dual_slack
        self.dual_dist = dual_dist
        self.inner = inner
        self.algo = algo
        self.sk_lam_lr = sk_lam_lr

        self.env_name = env_name

        # Prepare staging area for feeding data to the model.
        stage_shapes = OrderedDict()
        for key in sorted(self.input_dims.keys()):
            if key.startswith('info_'):
                continue
            stage_shapes[key] = (None, *input_shapes[key])
        for key in ['o', 'g']:
            stage_shapes[key + '_2'] = stage_shapes[key]
        stage_shapes['r'] = (None,)
        stage_shapes['w'] = (None,)
        stage_shapes['s'] = (None,)
        stage_shapes['s_w'] = ()
        stage_shapes['r_w'] = ()
        stage_shapes['e_w'] = ()

        stage_shapes['myd'] = (None,)
        self.stage_shapes = stage_shapes

        # Create network.
        with tf.compat.v1.variable_scope(self.scope):
            self.staging_tf = StagingArea(
                dtypes=[tf.float32 for _ in self.stage_shapes.keys()],
                shapes=list(self.stage_shapes.values()))
            self.buffer_ph_tf = [
                tf.compat.v1.placeholder(tf.float32, shape=shape) for shape in self.stage_shapes.values()]
            self.stage_op = self.staging_tf.put(self.buffer_ph_tf)

            self._create_network(pretrain_weights, reuse=reuse)

        # Configure the replay buffer.
        buffer_shapes = {key: (self.T if key != 'o' else self.T+1, *input_shapes[key])
                         for key, val in input_shapes.items()}
        buffer_shapes['g'] = (buffer_shapes['g'][0], self.dimg)
        buffer_shapes['ag'] = (self.T+1, self.dimg)
        buffer_shapes['myr'] = (self.T,)
        buffer_shapes['myd'] = (self.T,)
        buffer_shapes['myv'] = (self.T,)
        buffer_size = (self.buffer_size // self.rollout_batch_size) * self.rollout_batch_size

        self.buffer = ReplayBuffer(buffer_shapes, buffer_size, self.T, self.sample_transitions)

        self.gl_r_history = deque(maxlen=history_len)
        self.sk_r_history = deque(maxlen=history_len)
        self.et_r_history = deque(maxlen=history_len)
        self.logp_current = 0
        self.finetune_pi = finetune_pi

        self.info_history = defaultdict(lambda: deque(maxlen=history_len))

    def _random_action(self, n):
        return np.random.uniform(low=-self.max_u, high=self.max_u, size=(n, self.dimu))

    def _preprocess_og(self, o, ag, g):
        if self.relative_goals:
            g_shape = g.shape
            g = g.reshape(-1, self.dimg)
            ag = ag.reshape(-1, self.dimg)
            g = self.subtract_goals(g, ag)
            g = g.reshape(*g_shape)
        o = np.clip(o, -self.clip_obs, self.clip_obs)
        g = np.clip(g, -self.clip_obs, self.clip_obs)
        return o, g

    def get_actions(self, o, z, ag, g, noise_eps=0., random_eps=0., use_target_net=False, compute_Q=False, exploit=False):
        o, g = self._preprocess_og(o, ag, g)
        policy = self.target if use_target_net else self.main
        # values to compute
        if self.sac:
            vals = [policy.mu_tf]
        else:
            vals = [policy.pi_tf]

        if compute_Q:
            vals += [policy.Q_pi_tf]

        feed = {
            policy.o_tf: o.reshape(-1, self.dimo),
            policy.z_tf: z.reshape(-1, self.dimz),
            policy.g_tf: g.reshape(-1, self.dimg),
            policy.u_tf: np.zeros((o.size // self.dimo, self.dimu), dtype=np.float32)
        }

        ret = self.sess.run(vals, feed_dict=feed)

        # action postprocessing
        u = ret[0]
        noise = noise_eps * self.max_u * np.random.randn(*u.shape)  # gaussian noise
        u += noise
        u = np.clip(u, -self.max_u, self.max_u)
        u += np.random.binomial(1, random_eps, u.shape[0]).reshape(-1, 1) * (self._random_action(u.shape[0]) - u)  # eps-greedy
        if u.shape[0] == 1:
            u = u[0]
        u = u.copy()
        ret[0] = u

        if len(ret) == 1:
            return ret[0]
        else:
            return ret

    def store_episode(self, episode_batch, update_stats=True):
        """
        episode_batch: array of batch_size x (T or T+1) x dim_key
                       'o' is of size T+1, others are of size T
        """
        
        episode_batch['s'] = np.empty([episode_batch['o'].shape[0], 1])
        # #

        self.buffer.store_episode(episode_batch, self)

        if update_stats:
            # add transitions to normalizer
            episode_batch['o_2'] = episode_batch['o'][:, 1:, :]
            episode_batch['ag_2'] = episode_batch['ag'][:, 1:, :]
            num_normalizing_transitions = transitions_in_episode_batch(episode_batch)
            
            transitions = self.sample_transitions(self, False, episode_batch, num_normalizing_transitions, 0, 0)

            o, o_2, g, ag = transitions['o'], transitions['o_2'], transitions['g'], transitions['ag']
            transitions['o'], transitions['g'] = self._preprocess_og(o, ag, g)

            self.o_stats.update(transitions['o'])
            self.g_stats.update(transitions['g'])

            self.o_stats.recompute_stats()
            self.g_stats.recompute_stats()

    def get_current_buffer_size(self):
        return self.buffer.get_current_size()

    def _sync_optimizers(self):
        self.Q_adam.sync()
        self.pi_adam.sync()
        self.sk_adam.sync()
        if self.dual_reg:
            self.sk_dual_opt.sync()
            if self.dual_dist != 'l2':
                self.sk_dist_adam.sync()

    def _grads_sk(self, o_s_batch, z_s_batch, o2_s_batch, u_s_batch):
        run_list = [self.main_ir.sk_tf, self.sk_grad_tf]
        if self.dual_reg:
            run_list.extend([self.main_ir.sk_lambda_tf, self.sk_dual_grad_tf])
        result = self.sess.run(run_list, feed_dict={
            self.main_ir.o_tf: o_s_batch, self.main_ir.z_tf: z_s_batch, self.main_ir.o2_tf: o2_s_batch,
            self.main_ir.u_tf: u_s_batch, self.main_ir.is_training: True,
        })

        return result

    def _grads_sk_dist(self, o_s_batch, z_s_batch, o2_s_batch, add_dict):
        feed_dict = {self.main_ir.o_tf: o_s_batch, self.main_ir.z_tf: z_s_batch, self.main_ir.o2_tf: o2_s_batch, self.main_ir.is_training: True}

        if self.dual_dist == 's2_from_s':
            sk_dist, sk_dist_grad, sk_cst_dist = self.sess.run([self.main_ir.sk_dist_tf, self.sk_dist_grad_tf, self.main_ir.cst_dist], feed_dict=feed_dict)
        self.info_history['sk_cst_dist'].extend(sk_cst_dist)
        self.info_history['sk_dist'].extend(sk_dist)

        return sk_dist, sk_dist_grad

    def _grads(self):
        critic_loss, actor_loss, Q_grad, pi_grad, neg_logp_pi, e_w, log_et_r_scale = self.sess.run([
            self.Q_loss_tf,
            self.pi_loss_tf,
            self.Q_grad_tf,
            self.pi_grad_tf,
            self.main.neg_logp_pi_tf,
            self.e_w_tf,
            self.log_et_r_scale_tf,
        ])
        return critic_loss, actor_loss, Q_grad, pi_grad, neg_logp_pi, e_w, log_et_r_scale

    def _update(self, Q_grad, pi_grad):
        self.Q_adam.update(Q_grad, self.Q_lr)
        self.pi_adam.update(pi_grad, self.pi_lr)

    def sample_batch(self, ir, t):

        transitions = self.buffer.sample(self, ir, self.batch_size, self.sk_r_scale, t)
        weights = np.ones_like(transitions['r']).copy()
        if ir:
            if self.sk_clip:
                self.sk_r_history.extend(((np.clip(self.sk_r_scale * transitions['s'], *(-1, 0)))*1.00).tolist())
            else:
                self.sk_r_history.extend(((self.sk_r_scale * transitions['s']) * 1.00).tolist())
            self.gl_r_history.extend(self.r_scale * transitions['r'])

        o, o_2, g = transitions['o'], transitions['o_2'], transitions['g']
        ag, ag_2 = transitions['ag'], transitions['ag_2']
        transitions['o'], transitions['g'] = self._preprocess_og(o, ag, g)
        transitions['o_2'], transitions['g_2'] = self._preprocess_og(o_2, ag_2, g)

        transitions['w'] = weights.flatten().copy() # note: ordered dict
        transitions_batch = [transitions[key] for key in self.stage_shapes.keys()]

        return transitions_batch

    def stage_batch(self, ir, t, batch=None):
        if batch is None:
            batch = self.sample_batch(ir, t)
        assert len(self.buffer_ph_tf) == len(batch)
        self.sess.run(self.stage_op, feed_dict=dict(zip(self.buffer_ph_tf, batch)))

    def run_sk(self, o, z, o2=None, u=None):
        feed_dict = {self.main_ir.o_tf: o, self.main_ir.z_tf: z, self.main_ir.o2_tf: o2, self.main_ir.u_tf: u, self.main_ir.is_training: True}
        if self.dual_reg:
            sk_r, cst_twoside, cst_oneside = self.sess.run([self.main_ir.sk_r_tf, self.main_ir.cst_twoside, self.main_ir.cst_oneside], feed_dict=feed_dict)
            self.info_history['cst_twoside'].extend(cst_twoside)
            self.info_history['cst_oneside'].extend(cst_oneside)
        else:
            sk_r = self.sess.run(self.main_ir.sk_r_tf, feed_dict=feed_dict)
        return sk_r

    def train_sk(self, o_s_batch, z_s_batch, o2_s_batch, u_s_batch, stage=True):
        result = self._grads_sk(o_s_batch, z_s_batch, o2_s_batch, u_s_batch)
        if self.dual_reg:
            sk, sk_grad, sk_lambda, sk_dual_grad = result
            self.sk_dual_opt.update(sk_dual_grad, self.sk_lam_lr)
        else:
            sk, sk_grad = result
        self.sk_adam.update(sk_grad, self.sk_lr)
        return -sk.mean()

    def train_sk_dist(self, o_s_batch, z_s_batch, o2_s_batch, add_dict, stage=True):
        sk_dist, sk_dist_grad = self._grads_sk_dist(o_s_batch, z_s_batch, o2_s_batch, add_dict)
        self.sk_dist_adam.update(sk_dist_grad, self.sk_lr)
        return -sk_dist.mean()

    def train(self, t, stage=True):
        if not self.buffer.current_size==0:
            if stage:
                self.stage_batch(ir=True, t=t)
            critic_loss, actor_loss, Q_grad, pi_grad, neg_logp_pi, e_w, log_et_r_scale = self._grads()
            self.info_history['critic_loss'].extend([critic_loss] * neg_logp_pi.shape[0])
            self.info_history['actor_loss'].extend([actor_loss] * neg_logp_pi.shape[0])
            self._update(Q_grad, pi_grad)
            et_r_scale = np.exp(log_et_r_scale)
            if self.et_clip:
                self.et_r_history.extend((( np.clip((et_r_scale * neg_logp_pi), *(-1, 0))) * e_w ).tolist())
            else:
                self.et_r_history.extend((( et_r_scale * neg_logp_pi) * e_w ).tolist())
            self.et_r_scale_current = et_r_scale
            self.logp_current = -neg_logp_pi.mean()
            return critic_loss, actor_loss

    def _init_target_net(self):
        self.sess.run(self.init_target_net_op)

    def update_target_net(self):
        self.sess.run(self.update_target_net_op)

    def clear_buffer(self):
        self.buffer.clear_buffer()

    def _vars(self, scope):
        res = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope + '/' + scope)
        assert len(res) > 0
        return res

    def _global_vars(self, scope):
        res = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope=self.scope + '/' + scope)
        return res

    def _create_network(self, pretrain_weights, reuse=False):
        if self.sac:
            logger.info("Creating a SAC agent with action space %d x %s..." % (self.dimu, self.max_u))
        else:
            logger.info("Creating a DDPG agent with action space %d x %s..." % (self.dimu, self.max_u))

        self.sess = tf.compat.v1.get_default_session()
        if self.sess is None:
            self.sess = tf.compat.v1.InteractiveSession()

        # running averages
        with tf.compat.v1.variable_scope('o_stats') as vs:
            if reuse:
                vs.reuse_variables()
            self.o_stats = Normalizer(self.dimo, self.norm_eps, self.norm_clip, sess=self.sess)
        with tf.compat.v1.variable_scope('g_stats') as vs:
            if reuse:
                vs.reuse_variables()
            self.g_stats = Normalizer(self.dimg, self.norm_eps, self.norm_clip, sess=self.sess)

        # mini-batch sampling.
        batch = self.staging_tf.get()
        batch_tf = OrderedDict([(key, batch[i])
                                for i, key in enumerate(self.stage_shapes.keys())])
        batch_tf['r'] = tf.reshape(batch_tf['r'], [-1, 1])
        batch_tf['w'] = tf.reshape(batch_tf['w'], [-1, 1])
        batch_tf['s'] = tf.reshape(batch_tf['s'], [-1, 1])
        batch_tf['myd'] = tf.reshape(batch_tf['myd'], [-1, 1])

        self.o_tau_tf = tf.compat.v1.placeholder(tf.float32, shape=(None, None, self.dimo))

        # networks
        with tf.compat.v1.variable_scope('main') as vs:
            if reuse:
                vs.reuse_variables()
            self.main = self.create_actor_critic(batch_tf, net_type='main', **self.__dict__)
            vs.reuse_variables()
        with tf.compat.v1.variable_scope('target') as vs:
            if reuse:
                vs.reuse_variables()
            target_batch_tf = batch_tf.copy()
            target_batch_tf['o'] = batch_tf['o_2']
            target_batch_tf['g'] = batch_tf['g_2']
            self.target = self.create_actor_critic(
                target_batch_tf, net_type='target', **self.__dict__)
            vs.reuse_variables()
        assert len(self._vars("main")) == len(self._vars("target"))

        # intrinsic reward (ir) network for mutual information
        with tf.compat.v1.variable_scope('ir') as vs:
            if reuse:
                vs.reuse_variables()
            self.main_ir = self.create_discriminator(batch_tf, net_type='ir', **self.__dict__)
            vs.reuse_variables()

        # loss functions
        sk_grads_tf = tf.gradients(ys=tf.reduce_mean(input_tensor=self.main_ir.sk_tf), xs=self._vars('ir/skill_ds'))
        assert len(self._vars('ir/skill_ds')) == len(sk_grads_tf)
        self.sk_grads_vars_tf = zip(sk_grads_tf, self._vars('ir/skill_ds'))  # Seems not used
        self.sk_grad_tf = flatten_grads(grads=sk_grads_tf, var_list=self._vars('ir/skill_ds'))
        self.sk_adam = MpiAdam(self._vars('ir/skill_ds'), scale_grad_by_procs=False)

        if self.dual_reg:
            sk_dual_grads_tf = tf.gradients(ys=tf.reduce_mean(input_tensor=self.main_ir.sk_lambda_tf), xs=self._vars('ir/skill_dual'))
            assert len(self._vars('ir/skill_dual')) == len(sk_dual_grads_tf)
            self.sk_dual_grad_tf = flatten_grads(grads=sk_dual_grads_tf, var_list=self._vars('ir/skill_dual'))
            if self.dual_lam_opt == 'adam':
                self.sk_dual_opt = MpiAdam(self._vars('ir/skill_dual'), scale_grad_by_procs=False)
            else:
                self.sk_dual_opt = MpiSgd(self._vars('ir/skill_dual'), scale_grad_by_procs=False)

            if self.dual_dist != 'l2':
                sk_dist_grads_tf = tf.gradients(ys=tf.reduce_mean(input_tensor=self.main_ir.sk_dist_tf), xs=self._vars('ir/skill_dist'))
                assert len(self._vars('ir/skill_dist')) == len(sk_dist_grads_tf)
                self.sk_dist_grad_tf = flatten_grads(grads=sk_dist_grads_tf, var_list=self._vars('ir/skill_dist'))
                self.sk_dist_adam = MpiAdam(self._vars('ir/skill_dist'), scale_grad_by_procs=False)

        target_Q_pi_tf = self.target.Q_pi_tf
        clip_range = (-self.clip_return, self.clip_return if self.clip_pos_returns else np.inf)

        self.e_w_tf = batch_tf['e_w']

        if not self.sac:
            self.main.neg_logp_pi_tf = tf.zeros(1)

        et_r_scale_init = tf.constant_initializer(np.log(self.et_r_scale))
        self.log_et_r_scale_tf = tf.compat.v1.get_variable('alpha/log_et_r_scale', (), tf.float32, initializer=et_r_scale_init)
        et_r_scale = tf.exp(self.log_et_r_scale_tf)
        target_tf = tf.clip_by_value(self.r_scale * batch_tf['r'] * batch_tf['r_w']
                                     + (tf.clip_by_value( self.sk_r_scale * batch_tf['s'], *(-1, 0)) if self.sk_clip else self.sk_r_scale * batch_tf['s']) * batch_tf['s_w']
                                     + (tf.clip_by_value( et_r_scale * self.main.neg_logp_pi_tf, *(-1, 0)) if self.et_clip else et_r_scale * self.main.neg_logp_pi_tf) * self.e_w_tf
                                     + (self.gamma * target_Q_pi_tf * (1 - batch_tf['myd']) if self.done_ground else self.gamma * target_Q_pi_tf), *clip_range)

        self.td_error_tf = tf.stop_gradient(target_tf) - self.main.Q_tf
        self.errors_tf = tf.square(self.td_error_tf)
        self.errors_tf = tf.reduce_mean(input_tensor=batch_tf['w'] * self.errors_tf)
        self.Q_loss_tf = tf.reduce_mean(input_tensor=self.errors_tf)

        self.pi_loss_tf = -tf.reduce_mean(input_tensor=self.main.Q_pi_tf)
        self.pi_loss_tf += self.action_l2 * tf.reduce_mean(input_tensor=tf.square(self.main.pi_tf / self.max_u))

        Q_grads_tf = tf.gradients(ys=self.Q_loss_tf, xs=self._vars('main/Q'))
        pi_grads_tf = tf.gradients(ys=self.pi_loss_tf, xs=self._vars('main/pi'))
        assert len(self._vars('main/Q')) == len(Q_grads_tf)
        assert len(self._vars('main/pi')) == len(pi_grads_tf)
        self.Q_grads_vars_tf = zip(Q_grads_tf, self._vars('main/Q'))
        self.pi_grads_vars_tf = zip(pi_grads_tf, self._vars('main/pi'))
        self.Q_grad_tf = flatten_grads(grads=Q_grads_tf, var_list=self._vars('main/Q'))
        self.pi_grad_tf = flatten_grads(grads=pi_grads_tf, var_list=self._vars('main/pi'))

        # optimizers
        self.Q_adam = MpiAdam(self._vars('main/Q'), scale_grad_by_procs=False)
        self.pi_adam = MpiAdam(self._vars('main/pi'), scale_grad_by_procs=False)

        self.main_vars = self._vars('main/Q') + self._vars('main/pi')
        self.target_vars = self._vars('target/Q') + self._vars('target/pi')

        # polyak averaging
        self.stats_vars = self._global_vars('o_stats') + self._global_vars('g_stats')
        self.init_target_net_op = list(
            map(lambda v: v[0].assign(v[1]), zip(self.target_vars, self.main_vars)))
        self.update_target_net_op = list(
            map(lambda v: v[0].assign(self.polyak * v[0] + (1. - self.polyak) * v[1]), zip(self.target_vars, self.main_vars)))

        # initialize all variables
        tf.compat.v1.variables_initializer(self._global_vars('')).run()
        if pretrain_weights:
            load_weight(self.sess, pretrain_weights, [''])

        self._sync_optimizers()
        # if pretrain_weights and self.finetune_pi:
        #     load_weight(self.sess, pretrain_weights, ['target'])
        # else:
        #     self._init_target_net()

    def logs(self, prefix='', is_policy_training=True):
        logs = []
        logs += [('stats_o/mean', np.mean(self.sess.run([self.o_stats.mean])))]
        logs += [('stats_o/std', np.mean(self.sess.run([self.o_stats.std])))]
        logs += [('stats_g/mean', np.mean(self.sess.run([self.g_stats.mean])))]
        logs += [('stats_g/std', np.mean(self.sess.run([self.g_stats.std])))]
        if is_policy_training:
            logs += [('sk_reward/mean', np.mean(self.sk_r_history))]
            logs += [('sk_reward/std', np.std(self.sk_r_history))]
            logs += [('sk_reward/max', np.max(self.sk_r_history))]
            logs += [('sk_reward/min', np.min(self.sk_r_history))]
            logs += [('et_reward/mean', np.mean(self.et_r_history))]
            logs += [('et_reward/std', np.std(self.et_r_history))]
            logs += [('et_reward/max', np.max(self.et_r_history))]
            logs += [('et_reward/min', np.min(self.et_r_history))]
            logs += [('et_train/logp', self.logp_current)]
            logs += [('et_train/et_r_scale', self.et_r_scale_current)]
            logs += [('gl_reward/mean', np.mean(self.gl_r_history))]
            logs += [('gl_reward/std', np.std(self.gl_r_history))]
            logs += [('gl_reward/max', np.max(self.gl_r_history))]
            logs += [('gl_reward/min', np.min(self.gl_r_history))]
            logs += [('loss/actor_loss', np.mean(self.info_history['actor_loss']))]
            logs += [('loss/critic_loss', np.mean(self.info_history['critic_loss']))]
            if self.dual_reg:
                logs += [('sk_dual/sk_lambda', np.exp(self.sess.run(self.main_ir.log_sk_lambda)))]
                logs += [('sk_dual/cst_twoside_mean', np.mean(self.info_history['cst_twoside']))]
                logs += [('sk_dual/cst_oneside_mean', np.mean(self.info_history['cst_oneside']))]
        if self.dual_reg and self.dual_dist != 'l2':
            logs += [('sk_dual/sk_cst_dist', np.mean(self.info_history['sk_cst_dist']))]
            logs += [('sk_dual/sk_dist', np.mean(self.info_history['sk_dist']))]

        if prefix is not '' and not prefix.endswith('/'):
            return [(prefix + '/' + key, val) for key, val in logs]
        else:
            return logs

    def __getstate__(self):
        """Our policies can be loaded from pkl, but after unpickling you cannot continue training.
        """
        excluded_subnames = ['_tf', '_op', '_vars', '_adam', '_sgd', 'buffer', 'sess', '_stats',
                             'main', 'target', 'lock', 'sample_transitions',
                             'stage_shapes', 'create_actor_critic', 'create_discriminator', '_history']

        state = {k: v for k, v in self.__dict__.items() if all([not subname in k for subname in excluded_subnames])}
        state['buffer_size'] = self.buffer_size
        state['tf'] = self.sess.run([x for x in self._global_vars('') if 'buffer' not in x.name])
        return state

    def __setstate__(self, state):
        if 'sample_transitions' not in state:
            # We don't need this for playing the policy.
            state['sample_transitions'] = None
        if 'env_name' not in state:
            state['env_name'] = 'FetchPickAndPlace-v1'
        if 'network_class_discriminator' not in state:
            state['network_class_discriminator'] = 'baselines.her.discriminator:Discriminator'
        if 'sk_r_scale' not in state:
            state['sk_r_scale'] = 1
        if 'sk_lr' not in state:
            state['sk_lr'] = 0.001
        if 'et_r_scale' not in state:
            state['et_r_scale'] = 1
        if 'finetune_pi' not in state:
            state['finetune_pi'] = None
        if 'load_weight' not in state:
            state['load_weight'] = None
        if 'pretrain_weights' not in state:
            state['pretrain_weights'] = None
        if 'sac' not in state:
            state['sac'] = None

        self.__init__(**state)
        # set up stats (they are overwritten in __init__)
        for k, v in state.items():
            if k[-6:] == '_stats':
                self.__dict__[k] = v
        # load TF variables
        vars = [x for x in self._global_vars('') if 'buffer' not in x.name]
        assert(len(vars) == len(state["tf"]))
        node = [tf.compat.v1.assign(var, val) for var, val in zip(vars, state["tf"])]
        self.sess.run(node)
