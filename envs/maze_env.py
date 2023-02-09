# Copyright (c) 2019, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: MIT
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/MIT
from collections import defaultdict

import gym
import numpy as np

from envs.mazes import mazes_dict, make_crazy_maze, make_experiment_maze, make_hallway_maze, make_u_maze


class DummyGoal:
    pass


class MazeEnv(gym.Env):
    def __init__(self, n, maze_type='square', use_antigoal=False, ddiff=True, ignore_reset_start=False,
                 done_on_success=True, action_range_override=None, start_random_range_override=None,
                 obs_include_delta=False, keep_direction=False,
                 action_noise_std=None):
        self.n = n
        self._max_episode_steps = n
        self.env = DummyGoal()

        self._mazes = mazes_dict
        self.maze_type = maze_type.lower()

        self._ignore_reset_start = bool(ignore_reset_start)
        self._done_on_success = bool(done_on_success)

        self._obs_include_delta = obs_include_delta
        self._keep_direction = keep_direction
        self._action_noise_std = action_noise_std

        self._cur_direction = None

        # Generate a crazy maze specified by its size and generation seed
        if self.maze_type.startswith('crazy'):
            _, size, seed = self.maze_type.split('_')
            size = int(size)
            seed = int(seed)
            self._mazes[self.maze_type] = {'maze': make_crazy_maze(size, seed), 'action_range': 0.95}

        # Generate an "experiment" maze specified by its height, half-width, and size of starting section
        if self.maze_type.startswith('experiment'):
            _, h, half_w, sz0 = self.maze_type.split('_')
            h = int(h)
            half_w = int(half_w)
            sz0 = int(sz0)
            self._mazes[self.maze_type] = {'maze': make_experiment_maze(h, half_w, sz0), 'action_range': 0.25}

        if self.maze_type.startswith('corridor'):
            corridor_length = int(self.maze_type.split('_')[1])
            self._mazes[self.maze_type] = {'maze': make_hallway_maze(corridor_length), 'action_range': 0.95}

        if self.maze_type.startswith('umaze'):
            corridor_length = int(self.maze_type.split('_')[1])
            self._mazes[self.maze_type] = {'maze': make_u_maze(corridor_length), 'action_range': 0.95}

        assert self.maze_type in self._mazes
        self.min_x = self.maze.min_x
        self.max_x = self.maze.max_x
        self.min_y = self.maze.min_y
        self.max_y = self.maze.max_y
        self.min_point = np.array([self.min_x, self.min_y], dtype=np.float32)
        self.max_point = np.array([self.max_x, self.max_y], dtype=np.float32)

        if action_range_override is not None:
            self._mazes[self.maze_type]['action_range'] = action_range_override
        if start_random_range_override is not None:
            self.maze.start_random_range = start_random_range_override

        self.use_antigoal = bool(use_antigoal)
        self.ddiff = bool(ddiff)

        self._state = dict(s0=None, prev_state=None, state=None, goal=None, n=None, done=None, d_goal_0=None,
                           d_antigoal_0=None)

        self.dist_threshold = 0.15

        self.trajectory = []

        self.observation_space = gym.spaces.Dict({
            'observation': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(25,)),
            'achieved_goal': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3,)),
            'desired_goal': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3,)),
        })

        self.action_space = gym.spaces.Box(low=-self.action_range, high=self.action_range, shape=(2,))

        self.reset()

    @staticmethod
    def dist(goal, outcome):
        # return np.sum(np.abs(goal - outcome))
        return np.sqrt(np.sum((goal - outcome) ** 2))

    @property
    def maze(self):
        return self._mazes[self.maze_type]['maze']

    @property
    def action_range(self):
        return self._mazes[self.maze_type]['action_range']

    @property
    def state(self):
        return self._state['state'].reshape(-1)

    @property
    def goal(self):
        return self._state['goal'].reshape(-1)

    @property
    def antigoal(self):
        return self._state['antigoal'].reshape(-1)

    @property
    def reward(self):
        # r_sparse = -np.ones(1) + float(self.is_success)
        # r_dense = -self.dist(self.goal, self.state)
        # if self.use_antigoal:
        #     r_dense += self.dist(self.antigoal, self.state)
        # if not self.ddiff:
        #     reward = r_sparse + np.clip(r_dense, -np.inf, 0.0)
        # else:
        #     r_dense_prev = -self.dist(self.goal, self._state['prev_state'])
        #     if self.use_antigoal:
        #         r_dense_prev += self.dist(self.antigoal, self._state['prev_state'])
        #     r_dense -= r_dense_prev
        #     reward = r_sparse + r_dense
        reward = self.state[0] - self._state['prev_state'][0]
        return reward

    @property
    def achieved(self):
        return self.goal if self.is_success else self.state

    @property
    def is_done(self):
        return bool(self._state['done'])

    @property
    def is_success(self):
        d = self.dist(self.goal, self.state)
        return d <= self.dist_threshold

    @property
    def d_goal_0(self):
        return self._state['d_goal_0']

    @property
    def d_antigoal_0(self):
        return self._state['d_antigoal_0']

    @property
    def next_phase_reset(self):
        return {'state': self._state['s0'], 'goal': self.goal, 'antigoal': self.achieved}

    @property
    def sibling_reset(self):
        return {'state': self._state['s0'], 'goal': self.goal}

    def _get_mdp_state(self):
        observation = np.zeros(25)
        achieved_goal = np.zeros(3)
        desired_goal = np.zeros(3)
        observation[0:2] = self.state
        observation[3:5] = self.state
        return {
            'observation': observation,
            'achieved_goal': achieved_goal,
            'desired_goal': desired_goal,
        }

    def reset(self, state=None, goal=None, antigoal=None):
        # if state is None or self._ignore_reset_start:
        #     s_xy = self.maze.sample_start()
        # else:
        #     s_xy = state
        s_xy = np.zeros(2)
        s_xy = np.array(s_xy)
        if goal is None:
            if 'square' in self.maze_type:
                g_xy = self.maze.sample_goal(min_wall_dist=0.025 + self.dist_threshold)
            else:
                g_xy = self.maze.sample_goal()
        else:
            g_xy = goal
        g_xy = np.array(g_xy)

        if antigoal is None:
            ag_xy = g_xy
        else:
            ag_xy = antigoal

        if self._keep_direction:
            self._cur_direction = np.random.random() * 2 * np.pi

        self._state = {
            's0': s_xy,
            'prev_state': s_xy * np.ones_like(s_xy),
            'state': s_xy * np.ones_like(s_xy),
            'goal': g_xy,
            'antigoal': ag_xy,
            'n': 0,
            'done': False,
            'd_goal_0': self.dist(g_xy, s_xy),
            'd_antigoal_0': self.dist(g_xy, ag_xy),
        }

        self.trajectory = [self.state]

        return self._get_mdp_state()

    def step(self, action):
        obsbefore = self._get_mdp_state()

        action = action.copy() * 0.2

        if self._action_noise_std is not None:
            action = action + np.random.normal(scale=self._action_noise_std, size=action.shape)

        # Clip action
        for i in range(len(action)):
            action[i] = np.clip(action[i], -self.action_range, self.action_range)

        try:
            next_state = self._state['state'] + action
            # if self._keep_direction:
            #     r = (action[0] + self.action_range) / 2
            #     theta = (action[1] + self.action_range) / (2 * self.action_range) * np.pi - np.pi / 2
            #     self._cur_direction += theta
            #     x = r * np.cos(self._cur_direction)
            #     y = r * np.sin(self._cur_direction)
            #     next_state = self.maze.move(
            #         self._state['state'],
            #         np.array([x, y]),
            #     )
            # else:
            #     next_state = self.maze.move(
            #         self._state['state'],
            #         action
            #     )
            next_state = np.array(next_state)
        except:
            print('state', self._state['state'])
            print('action', action)
            raise
        self._state['prev_state'] = self._state['state']
        self._state['state'] = next_state
        self._state['n'] += 1
        # done = self._state['n'] >= self.n
        # if self._done_on_success:
        #     done = done or self.is_success
        done = False
        self._state['done'] = done

        self.trajectory.append(self.state)

        # self.render()

        return self._get_mdp_state(), self.reward, self.is_done, {
            'coordinates': self._state['prev_state'],
            'next_coordinates': self._state['state'],
        }

    def sample(self):
        return self.maze.sample()

    def render(self, *args):
        self.maze.plot(trajectory=self.trajectory)

    def render_trajectories(self, trajectories, colors, plot_axis, ax):
        """Plot multiple trajectories onto ax"""
        coordinates_trajectories = self._get_coordinates_trajectories(trajectories)
        self.maze.plot_trajectories(coordinates_trajectories, colors, plot_axis, ax)

    def _get_coordinates_trajectories(self, trajectories):
        coordinates_trajectories = []
        for trajectory in trajectories:
            if trajectory['env_infos']['coordinates'].ndim == 2:
                coordinates_trajectories.append(np.concatenate([
                    trajectory['env_infos']['coordinates'],
                    [trajectory['env_infos']['next_coordinates'][-1]]
                ]))
            elif trajectory['env_infos']['coordinates'].ndim > 2:
                # Nested array (due to the child policy)
                coordinates_trajectories.append(np.concatenate([
                    trajectory['env_infos']['coordinates'].reshape(-1, 2),
                    trajectory['env_infos']['next_coordinates'].reshape(-1, 2)[-1:]
                ]))

        return coordinates_trajectories

    def calc_eval_metrics(self, trajectories, is_option_trajectories):
        trajectory_eval_metrics = defaultdict(list)
        coordinates_trajectories = self._get_coordinates_trajectories(trajectories)

        for trajectory, coordinates_trajectory in zip(trajectories, coordinates_trajectories):
            # terminal distance
            trajectory_eval_metrics['TerminalDistance'].append(np.linalg.norm(
                coordinates_trajectory[0] - coordinates_trajectory[-1]
            ))

            # smoothed length
            smooth_window_size = 5
            num_smooth_samples = 6
            if len(coordinates_trajectory) >= smooth_window_size:
                smoothed_coordinates_trajectory = np.zeros((len(coordinates_trajectory) - smooth_window_size + 1, 2))
                for i in range(2):
                    smoothed_coordinates_trajectory[:, i] = np.convolve(
                        coordinates_trajectory[:, i], [1 / smooth_window_size] * smooth_window_size, mode='valid'
                    )
                idxs = np.round(np.linspace(0, len(smoothed_coordinates_trajectory) - 1, num_smooth_samples)).astype(int)
                smoothed_coordinates_trajectory = smoothed_coordinates_trajectory[idxs]
            else:
                smoothed_coordinates_trajectory = coordinates_trajectory
            sum_distances = 0
            for i in range(len(smoothed_coordinates_trajectory) - 1):
                sum_distances += np.linalg.norm(
                    smoothed_coordinates_trajectory[i] - smoothed_coordinates_trajectory[i + 1]
                )
            trajectory_eval_metrics['SmoothedLength'].append(sum_distances)

        # cell percentage
        num_grids = 10  # per one side
        grid_xs = np.linspace(self.min_x, self.max_x, num_grids + 1)
        grid_ys = np.linspace(self.min_y, self.max_y, num_grids + 1)
        is_exist = np.zeros((num_grids, num_grids))
        for coordinates_trajectory in coordinates_trajectories:
            for x, y in coordinates_trajectory:
                x_idx = np.searchsorted(grid_xs, x)  # binary search
                y_idx = np.searchsorted(grid_ys, y)
                x_idx = np.clip(x_idx, 1, num_grids) - 1
                y_idx = np.clip(y_idx, 1, num_grids) - 1
                is_exist[x_idx, y_idx] = 1
        is_exist = is_exist.flatten()
        cell_percentage = np.sum(is_exist) / len(is_exist)

        eval_metrics = {
            'MaxTerminalDistance': np.max(trajectory_eval_metrics['TerminalDistance']),
            'MeanTerminalDistance': np.mean(trajectory_eval_metrics['TerminalDistance']),
            'MaxSmoothedLength': np.max(trajectory_eval_metrics['SmoothedLength']),
            'MeanSmoothedLength': np.mean(trajectory_eval_metrics['SmoothedLength']),
            'CellPercentage': cell_percentage,
        }

        if is_option_trajectories:
            # option std
            option_terminals = defaultdict(list)
            for trajectory, coordinates_trajectory in zip(trajectories, coordinates_trajectories):
                option_terminals[
                    tuple(trajectory['agent_infos']['option'][0])
                ].append(coordinates_trajectory[-1])
            mean_option_terminals = [np.mean(terminals, axis=0) for terminals in option_terminals.values()]
            intra_option_std = np.mean([np.mean(np.std(terminals, axis=0)) for terminals in option_terminals.values()])
            inter_option_std = np.mean(np.std(mean_option_terminals, axis=0))

            eval_metrics['IntraOptionStd'] = intra_option_std
            eval_metrics['InterOptionStd'] = inter_option_std
            eval_metrics['InterIntraOptionStdDiff'] = inter_option_std - intra_option_std

        return eval_metrics
