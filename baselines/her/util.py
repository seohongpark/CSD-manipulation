import os
import subprocess
import sys
import importlib
import inspect
import functools

import tensorflow as tf
import numpy as np

from baselines.common import tf_util as U
import platform
import json
import math


def store_args(method):
    """Stores provided method args as instance attributes.
    """
    argspec = inspect.getfullargspec(method)
    defaults = {}
    if argspec.defaults is not None:
        defaults = dict(
            zip(argspec.args[-len(argspec.defaults):], argspec.defaults))
    if argspec.kwonlydefaults is not None:
        defaults.update(argspec.kwonlydefaults)
    arg_names = argspec.args[1:]

    @functools.wraps(method)
    def wrapper(*positional_args, **keyword_args):
        self = positional_args[0]
        # Get default arg values
        args = defaults.copy()
        # Add provided arg values
        for name, value in zip(arg_names, positional_args[1:]):
            args[name] = value
        args.update(keyword_args)
        self.__dict__.update(args)
        return method(*positional_args, **keyword_args)

    return wrapper


def import_function(spec):
    """Import a function identified by a string like "pkg.module:fn_name".
    """
    mod_name, fn_name = spec.split(':')
    module = importlib.import_module(mod_name)
    fn = getattr(module, fn_name)
    return fn


def flatten_grads(var_list, grads):
    """Flattens a variables and their gradients.
    """
    return tf.concat([tf.reshape(grad, [U.numel(v)])
                      for (v, grad) in zip(var_list, grads)], 0)


def l2_norm(v, eps=1e-12):
    return v / (tf.reduce_sum(input_tensor=v ** 2) ** 0.5 + eps)


def spectral_norm(w, iteration=1, name='', is_training=None):
    w_shape = w.shape.as_list()
    w = tf.reshape(w, [-1, w_shape[-1]])

    u = tf.compat.v1.get_variable(f"{name}_u", [1, w_shape[-1]], initializer=tf.compat.v1.truncated_normal_initializer(), trainable=False)

    u_hat = u
    v_hat = None
    for i in range(iteration):
        v_ = tf.matmul(u_hat, tf.transpose(a=w))
        v_hat = l2_norm(v_)

        u_ = tf.matmul(v_hat, w)
        u_hat = l2_norm(u_)

    sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(a=u_hat))
    w_norm = w / sigma
    # w_norm = tf.Print(w_norm, [u, u_hat, is_training], 'u_before')

    def assign_f():
        with tf.control_dependencies([u.assign(u_hat)]):
            return tf.reshape(w_norm, w_shape)
    def noassign_f():
        return tf.reshape(w_norm, w_shape)

    w_norm = tf.cond(pred=is_training, true_fn=assign_f, false_fn=noassign_f)
    # w_norm = tf.Print(w_norm, [u, u_hat, is_training], 'u_after')

    return w_norm


def fully_conneted(x, units, use_bias=True, sn=False, name='fully_0', is_training=None):
    x = tf.compat.v1.layers.flatten(x)
    shape = x.get_shape().as_list()
    channels = shape[-1]
    if sn:
        w = tf.compat.v1.get_variable(f"{name}_kernel", [channels, units], tf.float32, initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"))
        if use_bias:
            bias = tf.compat.v1.get_variable(f"{name}_bias", [units], initializer=tf.compat.v1.constant_initializer(0.0))
            x = tf.matmul(x, spectral_norm(w, name=name, is_training=is_training)) + bias
        else:
            x = tf.matmul(x, spectral_norm(w, name=name, is_training=is_training))
    else:
        x = tf.compat.v1.layers.dense(x, units=units, kernel_initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"), use_bias=use_bias, name=name)
    return x


def nn(input, layers_sizes, reuse=None, flatten=False, name=""):
    """Creates a simple neural network
    """
    for i, size in enumerate(layers_sizes):
        activation = tf.nn.relu if i < len(layers_sizes)-1 else None
        input = tf.compat.v1.layers.dense(inputs=input,
                                units=size,
                                kernel_initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"),
                                reuse=reuse,
                                name=name+'_'+str(i))
        if activation:
            input = activation(input)
    if flatten:
        assert layers_sizes[-1] == 1
        input = tf.reshape(input, [-1])
    return input


def snn(input, layers_sizes, flatten=False, name="", sn=False, is_training=None):
    for i, size in enumerate(layers_sizes):
        activation = tf.nn.relu if i < len(layers_sizes)-1 else None
        input = fully_conneted(input, units=size, name=f'{name}_fully_{i}', sn=sn, is_training=is_training)
        if activation:
            input = activation(input)
    if flatten:
        assert layers_sizes[-1] == 1
        input = tf.reshape(input, [-1])
    return input


def install_mpi_excepthook():
    import sys
    from mpi4py import MPI
    old_hook = sys.excepthook

    def new_hook(a, b, c):
        old_hook(a, b, c)
        sys.stdout.flush()
        sys.stderr.flush()
        MPI.COMM_WORLD.Abort()
    sys.excepthook = new_hook


def mpi_fork(n, binding="core"):
    """Re-launches the current script with workers
    Returns "parent" for original parent, "child" for MPI children
    """
    if n <= 1:
        return "child"
    if os.getenv("IN_MPI") is None:
        env = os.environ.copy()
        env.update(
            MKL_NUM_THREADS="1",
            OMP_NUM_THREADS="1",
            IN_MPI="1"
        )
        # "-bind-to core" is crucial for good performance
        if platform.system() == 'Darwin':
            args = [
                "mpirun",
                "-np",
                str(n),
                # "-allow-run-as-root",
                sys.executable
            ]
        else:
            args = [
            "mpirun",
            "--oversubscribe",
            "-np",
            str(n),
            "-bind-to",
            binding, # core or none
            "-allow-run-as-root",
            sys.executable
        ]
        args += sys.argv
        subprocess.check_call(args, env=env)
        return "parent"
    else:
        install_mpi_excepthook()
        return "child"


def convert_episode_to_batch_major(episode):
    """Converts an episode to have the batch dimension in the major (first)
    dimension.
    """
    episode_batch = {}
    for key in episode.keys():
        val = np.array(episode[key]).copy()
        # make inputs batch-major instead of time-major
        episode_batch[key] = val.swapaxes(0, 1)

    return episode_batch


def transitions_in_episode_batch(episode_batch):
    """Number of transitions in a given episode batch.
    """
    shape = episode_batch['u'].shape
    return shape[0] * shape[1]


def reshape_for_broadcasting(source, target):
    """Reshapes a tensor (source) to have the correct shape and dtype of the target
    before broadcasting it with MPI.
    """
    dim = len(target.get_shape())
    shape = ([1] * (dim-1)) + [-1]
    return tf.reshape(tf.cast(source, target.dtype), shape)

def make_dir(filename):
    folder = os.path.dirname(filename)
    if not os.path.exists(folder):
        os.makedirs(folder)

def save_video(ims, filename, lib='cv2'):
    make_dir(filename)
    fps = 30.0
    (height, width, _) = ims[0].shape
    if lib == 'cv2':
        import cv2
        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'MJPG') # MJPG, XVID
        writer = cv2.VideoWriter(filename, fourcc, fps, (width, height))
    elif lib == 'imageio':
        import imageio
        writer = imageio.get_writer(filename, fps=fps)
    for i in range(ims.shape[0]):
        if lib == 'cv2':
            # Fix color error by converting RGB to BGR
            writer.write(cv2.cvtColor(np.uint8(ims[i]), cv2.COLOR_RGB2BGR))
        elif lib == 'imageio':
            writer.append_data(ims[i])
    if lib == 'cv2':
        writer.release()
    elif lib == 'imageio':
        writer.close()

def dumpJson(dirname, episodes, epoch, rank):
    os = []
    for episode in episodes:
        episode['o'] = episode['o'].tolist()
        os.append(episode['o'])
    with open(dirname+'/rollout_{0}_{1}.txt'.format(epoch, rank), 'w') as file:
         file.write(json.dumps(os))

def loadJson(dirname, epoch, rank):
    filename = '/rollout_{0}_{1}.txt'.format(epoch, rank)
    with open(dirname+filename, 'r') as file:
        os = json.loads(file.read())
        return os

def save_weight(sess, collection=tf.compat.v1.GraphKeys.GLOBAL_VARIABLES):
    return {v.name: sess.run(v) for v in tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope='ddpg' + '/' + '')}

def load_weight(sess, data, include=[]):
    # include: ['stats','main','target','state_mi','skill_ds']
    for scope in include:
        for v in tf.compat.v1.global_variables():
            if (v.name in data.keys()) and (scope in v.name):
                if v.shape == data[v.name].shape:
                    sess.run(v.assign(data[v.name]))
                    print('load weight: ', v.name)
            
