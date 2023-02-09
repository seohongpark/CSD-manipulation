from matplotlib import figure
import pathlib
import numpy as np
from matplotlib.patches import Ellipse

from baselines import logger
from moviepy import editor as mpy


class FigManager:
    def __init__(self, label, epoch, eval_dir):
        self.label = label
        self.epoch = epoch
        self.fig = figure.Figure()
        self.ax = self.fig.add_subplot()
        self.eval_dir = eval_dir

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        plot_path = (pathlib.Path(self.eval_dir)
                     / 'plots'
                     / f'{self.label}_{self.epoch}.png')
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        self.fig.savefig(plot_path, dpi=300)
        logger.record_tabular(self.label, (plot_path, self.label))


def get_2d_colors(points, min_point, max_point):
    points = np.array(points)
    min_point = np.array(min_point)
    max_point = np.array(max_point)

    colors = (points - min_point) / (max_point - min_point)
    colors = np.hstack((
        colors,
        (2 - np.sum(colors, axis=1, keepdims=True)) / 2,
    ))
    colors = np.clip(colors, 0, 1)
    colors = np.c_[colors, np.full(len(colors), 0.8)]

    return colors


def get_option_colors(options):
    num_options = options.shape[0]
    dim_option = options.shape[1]

    if dim_option <= 2:
        # Use a predefined option color scheme
        if dim_option == 1:
            options = np.hstack((options, options))
        option_colors = get_2d_colors(options, (-4, -4), (4, 4))
    else:
        if dim_option > 3 and num_options >= 3:
            from sklearn import decomposition
            pca = decomposition.PCA(n_components=3)
            # Add random noises to break symmetry.
            pca_options = np.vstack((options, np.random.randn(dim_option, dim_option)))
            pca.fit(pca_options)
            option_colors = np.array(pca.transform(options))
        elif dim_option > 3 and num_options < 3:
            option_colors = options[:, :3]
        elif dim_option == 3:
            option_colors = options

        # max_colors = np.max(option_colors, axis=0)
        # min_colors = np.min(option_colors, axis=0)
        max_colors = np.array([4] * 3)
        min_colors = np.array([-4] * 3)
        if all((max_colors - min_colors) > 0):
            option_colors = (option_colors - min_colors) / (max_colors - min_colors)
        option_colors = np.clip(option_colors, 0, 1)

        option_colors = np.c_[option_colors, np.full(len(option_colors), 0.8)]

    return option_colors


def setup_evaluation(num_eval_options, num_eval_trajectories_per_option, num_skills):
    dim_option = num_skills
    if dim_option == 2:
        # If dim_option is 2, use predefined options for evaluation.
        eval_options = [[0, 0]]
        for dist in [1.5, 3.0, 4.5, 0.75, 2.25, 3.75]:
            for angle in [0, 4, 2, 6, 1, 5, 3, 7]:
                eval_options.append([dist * np.cos(angle * np.pi / 4), dist * np.sin(angle * np.pi / 4)])
        eval_options = eval_options[:num_eval_options]
        eval_options = np.array(eval_options)
    else:
        eval_options = [[0] * dim_option]
        for dist in [1.5, -1.5, 3.0, -3.0, 4.5, -4.5]:
            for dim in range(dim_option):
                cur_option = [0] * dim_option
                cur_option[dim] = dist
                eval_options.append(cur_option)
        eval_options = eval_options[:num_eval_options]
        eval_options = np.array(eval_options)

    eval_options = np.repeat(eval_options, num_eval_trajectories_per_option, axis=0)
    eval_option_colors = get_option_colors(eval_options)

    return eval_options, eval_option_colors


def plot_trajectory(trajectory, color, ax):
    ax.plot(trajectory[:, 0], trajectory[:, 1], color=color, linewidth=0.7)


def plot_trajectories(trajectories, colors, plot_axis, ax):
    """Plot trajectories onto given ax."""
    square_axis_limit = 0.0

    for trajectory, color in zip(trajectories, colors):
        trajectory = np.array(trajectory)
        plot_trajectory(trajectory, color, ax)
        square_axis_limit = max(square_axis_limit, np.max(np.abs(trajectory[:, :2])))
    square_axis_limit = square_axis_limit * 1.2
    if plot_axis == 'free':
        return
    if plot_axis is None:
        plot_axis = [-square_axis_limit, square_axis_limit, -square_axis_limit, square_axis_limit]
    if plot_axis is not None:
        ax.axis(plot_axis)
        ax.set_aspect('equal')
    else:
        ax.axis('scaled')


def draw_2d_gaussians(means, stddevs, colors, ax, fill=False, alpha=0.8, use_adaptive_axis=False, draw_unit_gaussian=True, plot_axis=None):
    means = np.clip(means, -1000, 1000)
    stddevs = np.clip(stddevs, -1000, 1000)
    square_axis_limit = 2.0
    if draw_unit_gaussian:
        ellipse = Ellipse(xy=(0, 0), width=2, height=2,
                          edgecolor='r', lw=1, facecolor='none', alpha=0.5)
        ax.add_patch(ellipse)
    for mean, stddev, color in zip(means, stddevs, colors):
        if len(mean) == 1:
            mean = np.concatenate([mean, [0.]])
            stddev = np.concatenate([stddev, [0.1]])
        ellipse = Ellipse(xy=mean, width=stddev[0] * 2, height=stddev[1] * 2,
                          edgecolor=color, lw=1, facecolor='none' if not fill else color, alpha=alpha)
        ax.add_patch(ellipse)
        square_axis_limit = max(
            square_axis_limit,
            np.abs(mean[0] + stddev[0]),
            np.abs(mean[0] - stddev[0]),
            np.abs(mean[1] + stddev[1]),
            np.abs(mean[1] - stddev[1]),
        )
    square_axis_limit = square_axis_limit * 1.2
    ax.axis('scaled')
    if plot_axis is None:
        if use_adaptive_axis:
            ax.set_xlim(-square_axis_limit, square_axis_limit)
            ax.set_ylim(-square_axis_limit, square_axis_limit)
        else:
            ax.set_xlim(-5, 5)
            ax.set_ylim(-5, 5)
    else:
        ax.axis(plot_axis)


def prepare_video(v, n_cols=None):
    orig_ndim = v.ndim
    if orig_ndim == 4:
        v = v[None, ]

    #b, t, c, h, w = v.shape
    _, t, c, h, w = v.shape

    if v.dtype == np.uint8:
        v = np.float32(v) / 255.

    def is_power2(num):
        return num != 0 and ((num & (num - 1)) == 0)

    # pad to nearest power of 2, all at once
    # if not is_power2(v.shape[0]):
    #     len_addition = int(2**v.shape[0].bit_length() - v.shape[0])
    #     v = np.concatenate(
    #         (v, np.zeros(shape=(len_addition, t, c, h, w))), axis=0)
    # n_rows = 2**((b.bit_length() - 1) // 2)

    if n_cols is None:
        if v.shape[0] <= 3:
            n_cols = v.shape[0]
        elif v.shape[0] <= 9:
            n_cols = 3
        elif v.shape[0] <= 18:
            n_cols = 6
        else:
            n_cols = 8
    if v.shape[0] % n_cols != 0:
        len_addition = n_cols - v.shape[0] % n_cols
        v = np.concatenate(
            (v, np.zeros(shape=(len_addition, t, c, h, w))), axis=0)
    n_rows = v.shape[0] // n_cols

    v = np.reshape(v, newshape=(n_rows, n_cols, t, c, h, w))
    v = np.transpose(v, axes=(2, 0, 4, 1, 5, 3))
    v = np.reshape(v, newshape=(t, n_rows * h, n_cols * w, c))

    return v


def save_video(path, tensor, fps=15, n_cols=None):
    def _to_uint8(t):
        # If user passes in uint8, then we don't need to rescale by 255
        if t.dtype != np.uint8:
            t = (t * 255.0).astype(np.uint8)
        return t
    if tensor.dtype in [np.object]:
        tensor = [_to_uint8(prepare_video(t, n_cols)) for t in tensor]
    else:
        tensor = prepare_video(tensor, n_cols)
        tensor = _to_uint8(tensor)

    # Encode sequence of images into gif string
    clip = mpy.ImageSequenceClip(list(tensor), fps=fps)

    plot_path = pathlib.Path(path)
    plot_path.parent.mkdir(parents=True, exist_ok=True)

    clip.write_videofile(str(plot_path), audio=False, verbose=False, logger=None)


def record_video(path, trajectories, n_cols=None):
    renders = []
    for trajectory in trajectories:
        render = trajectory.transpose(0, 3, 1, 2).astype(np.uint8)
        renders.append(render)
    max_length = max([len(render) for render in renders])
    for i, render in enumerate(renders):
        renders[i] = np.concatenate([render, np.zeros((max_length - render.shape[0], *render.shape[1:]), dtype=render.dtype)], axis=0)
    renders = np.array(renders)
    save_video(path, renders, n_cols=n_cols)


class RunningMeanStd(object):
    def __init__(self, epsilon=1e-5, shape=()):
        self.mean = np.zeros(shape, np.float32)
        self.var = np.ones(shape, np.float32)
        self.count = epsilon

    def update(self, arr: np.ndarray) -> None:
        batch_mean = np.mean(arr, axis=0)
        batch_var = np.var(arr, axis=0)
        batch_count = arr.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean: np.ndarray, batch_var: np.ndarray, batch_count: int) -> None:
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + np.square(delta) * self.count * batch_count / (self.count + batch_count)
        new_var = m_2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count
