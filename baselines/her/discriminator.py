import tensorflow as tf

from baselines.her.normalizer import Normalizer
from baselines.her.util import store_args, nn, snn
import numpy as np


class Discriminator:
    @store_args
    def __init__(self, inputs_tf, dimo, dimz, dimg, dimu, max_u, o_stats, g_stats, hidden, layers, env_name, **kwargs):
        """The discriminator network and related training code.

        Args:
            inputs_tf (dict of tensors): all necessary inputs for the network: the
                observation (o), the goal (g), and the action (u)
            dimo (int): the dimension of the observations
            dimg (int): the dimension of the goals
            dimu (int): the dimension of the actions
            max_u (float): the maximum magnitude of actions; action outputs will be scaled
                accordingly
            o_stats (baselines.her.Normalizer): normalizer for observations
            g_stats (baselines.her.Normalizer): normalizer for goals
            hidden (int): number of hidden units that should be used in hidden layers
            layers (int): number of hidden layers
        """

        self.o_tf = tf.compat.v1.placeholder(tf.float32, shape=(None, self.dimo))
        self.o2_tf = tf.compat.v1.placeholder(tf.float32, shape=(None, self.dimo))
        self.z_tf = tf.compat.v1.placeholder(tf.float32, shape=(None, self.dimz))
        self.g_tf = tf.compat.v1.placeholder(tf.float32, shape=(None, self.dimg))
        self.u_tf = tf.compat.v1.placeholder(tf.float32, shape=(None, self.dimu))
        self.is_training = tf.compat.v1.placeholder(tf.bool, shape=())

        o_tau_tf = self.o_tau_tf

        o_tf = self.o_tf
        o2_tf = self.o2_tf
        obs_focus = o_tf
        obs2_focus = o2_tf

        if self.dual_reg:
            with tf.compat.v1.variable_scope('skill_dual', reuse=tf.compat.v1.AUTO_REUSE):
                sk_lambda_init = tf.constant_initializer(np.log(self.dual_init_lambda))
                self.log_sk_lambda = tf.compat.v1.get_variable('lam/log_sk_lambda', (), tf.float32, initializer=sk_lambda_init)
            with tf.compat.v1.variable_scope('skill_dist', reuse=tf.compat.v1.AUTO_REUSE):
                if self.dual_dist == 'l2':
                    pass
                elif self.dual_dist == 's2_from_s':
                    dims2 = obs2_focus.shape[1]
                    self.s2_mean_tf = snn(obs_focus, [int(self.hidden / 2)] * self.layers + [dims2], name='s2_mean', is_training=self.is_training)
                    self.s2_log_std_tf = snn(obs_focus, [int(self.hidden / 2)] * self.layers + [dims2], name='s2_log_std', is_training=self.is_training)
                    self.s2_clamped_log_std_tf = tf.clip_by_value(self.s2_log_std_tf, -13.8155, 1000)
                    self.s2_clamped_std_tf = tf.exp(self.s2_clamped_log_std_tf)
                    # Predict delta_s
                    self.sk_dist_tf = tf.math.reduce_sum(input_tensor=(tf.abs(self.s2_mean_tf - (obs2_focus - obs_focus)) / self.s2_clamped_std_tf) ** 2 + self.s2_clamped_std_tf, axis=1)

        with tf.compat.v1.variable_scope('skill_ds', reuse=tf.compat.v1.AUTO_REUSE):
            if self.skill_type == 'discrete':
                eye_z = tf.tile(tf.expand_dims(tf.eye(self.dimz), 0), [tf.shape(input=obs_focus)[0], 1, 1])
                self.mean_tf = snn(obs_focus, [int(self.hidden / 2)] * self.layers + [self.dimz], name='mean', sn=self.spectral_normalization, is_training=self.is_training)
                self.mean2_tf = snn(obs2_focus, [int(self.hidden / 2)] * self.layers + [self.dimz], name='mean', sn=self.spectral_normalization, is_training=self.is_training)
                self.mean_diff_tf = self.mean2_tf - self.mean_tf
                mean_diff_tf = tf.expand_dims(self.mean_diff_tf, 1)
                logits = tf.math.reduce_sum(input_tensor=eye_z * mean_diff_tf, axis=2)
                masks = self.z_tf * self.dimz / (self.dimz - 1) - 1 / (self.dimz - 1)
                self.sk_tf = -tf.reduce_sum(input_tensor=logits * masks, axis=1)
                self.sk_r_tf = -1 * self.sk_tf
            else:
                self.mean_tf = snn(obs_focus, [int(self.hidden / 2)] * self.layers + [self.dimz], name='mean', sn=self.spectral_normalization, is_training=self.is_training)
                self.mean2_tf = snn(obs2_focus, [int(self.hidden / 2)] * self.layers + [self.dimz], name='mean', sn=self.spectral_normalization, is_training=self.is_training)
                self.mean_diff_tf = self.mean2_tf - self.mean_tf
                self.sk_tf = -tf.math.reduce_sum(input_tensor=self.mean_diff_tf * self.z_tf, axis=1)
                self.sk_r_tf = -1 * self.sk_tf

            if self.dual_reg:
                x = obs_focus
                y = obs2_focus
                phi_x = self.mean_tf
                phi_y = self.mean2_tf

                if self.dual_dist == 'l2':
                    self.cst_dist = tf.reduce_mean(tf.square(y - x), axis=1)
                elif self.dual_dist == 's2_from_s':
                    self.scaling_factor = 1. / (self.s2_clamped_std_tf)
                    self.geo_mean = tf.exp(tf.reduce_mean(tf.math.log(self.scaling_factor), axis=1, keepdims=True))
                    self.normalized_scaling_factor = (self.scaling_factor / self.geo_mean) ** 2
                    self.cst_dist = tf.reduce_mean(tf.abs(self.s2_mean_tf - (obs2_focus - obs_focus)) ** 2 * self.normalized_scaling_factor, axis=1)

                self.phi_dist = tf.reduce_mean(tf.square(phi_y - phi_x), axis=1)

                self.cst_twoside = self.cst_dist - self.phi_dist
                self.cst_oneside = tf.minimum(self.dual_slack, self.cst_twoside)
                self.cst_penalty = -tf.stop_gradient(tf.exp(self.log_sk_lambda)) * self.cst_oneside
                self.sk_lambda_tf = self.log_sk_lambda * tf.stop_gradient(self.cst_oneside)
                self.sk_tf = self.sk_tf + self.cst_penalty
