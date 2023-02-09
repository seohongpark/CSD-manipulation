from d4rl.kitchen.kitchen_envs import KitchenBase
from dm_control.mujoco import engine
import numpy as np


class KitchenMicrowaveKettleLightSliderV0Custom(KitchenBase):
    TASK_ELEMENTS = ['microwave', 'kettle', 'light switch', 'slide cabinet']

    def render(self, mode='human', width=None, height=None):
        if width is None or height is None:
            return []
        camera = engine.MovableCamera(self.sim, width, height)
        camera.set_pose(distance=2.2, lookat=[-0.2, .5, 2.], azimuth=70, elevation=-35)
        img = camera.render()
        return img

    def _get_obs(self):
        t, qp, qv, obj_qp, obj_qv = self.robot.get_obs(
            self, robot_noise_ratio=self.robot_noise_ratio)

        self.obs_dict = {}
        self.obs_dict['t'] = t
        self.obs_dict['qp'] = qp
        self.obs_dict['qv'] = qv
        self.obs_dict['obj_qp'] = obj_qp
        self.obs_dict['obj_qv'] = obj_qv
        self.obs_dict['goal'] = self.goal
        return np.concatenate([self.obs_dict['qp'], self.obs_dict['obj_qp']])
