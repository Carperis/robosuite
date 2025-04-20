from mujoco import viewer
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R
from collections import deque

DEFAULT_FREE_CAM = {
    "lookat": [0, 0, 1],
    "distance": 2,
    "azimuth": 180,
    "elevation": -20,
}


class MjviewerRenderer:
    def __init__(self, env, camera_id=None, cam_config=None):
        if cam_config is None:
            cam_config = DEFAULT_FREE_CAM
        self.env = env
        self.camera_id = camera_id
        self.viewer = None
        self.camera_config = cam_config

        self.texts = deque(maxlen=20)
        self.max_frames = 20
        self.frames = []
        self.frames_to_keep = set()
        self.text_auto_clean = False

    def render(self):
        pass

    def set_camera(self, camera_id):
        self.camera_id = camera_id

    def update(self):
        if self.viewer is None:
            self.viewer = viewer.launch_passive(
                self.env.sim.model._model,
                self.env.sim.data._data,
                show_left_ui=False,
                show_right_ui=False,
            )

            self.viewer.opt.geomgroup[0] = 0

            if self.camera_config is not None:
                self.viewer.cam.lookat = self.camera_config["lookat"]
                self.viewer.cam.distance = self.camera_config["distance"]
                self.viewer.cam.azimuth = self.camera_config["azimuth"]
                self.viewer.cam.elevation = self.camera_config["elevation"]

            if self.camera_id is not None:
                if self.camera_id >= 0:
                    self.viewer.cam.type = 2
                    self.viewer.cam.fixedcamid = self.camera_id
                else:
                    self.viewer.cam.type = 0

        if self.viewer is not None:
            self.viewer.user_scn.ngeom = 0
        self._mjprint()
        self._mjshowframe()
        self.viewer.sync()
        if self.text_auto_clean:
            self.texts.clear()
            
        indices_to_remove = []
        for i, frame in enumerate(self.frames):
            if frame[3] is not None and frame[3] not in self.frames_to_keep or len(self.frames) > self.max_frames:
                indices_to_remove.append(i)
        for i in reversed(indices_to_remove):
            del self.frames[i]
            

    def reset(self):
        pass

    def close(self):

        self.sim = None
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def add_keypress_callback(self, keypress_callback):
        self.keypress_callback = keypress_callback

    def mjprint(self, text, auto_clean=False):
        """Prints text to the viewer window."""
        self.text_auto_clean = auto_clean
        texts = text.split("\n")
        self.texts.extend(texts)

    def _mjprint(self):
        """Prints text to the viewer window."""
        if self.viewer is not None:
            if self.env.render_camera:
                # cam_configs = self.env._cam_configs
                # curr_config = cam_configs[self.env.render_camera]
                # cam_pos = curr_config["pos"]
                # cam_quat = curr_config["quat"] # [w, x, y, z]
                # cam_quat = np.array([cam_quat[1], cam_quat[2], cam_quat[3], cam_quat[0]]) # [x, y, z, w]
                # cam_rot = R.from_quat(cam_quat).as_matrix()
                # cam_base_quat = self.env.sim.data.get_body_xquat("mobilebase0_support") # [w, x, y, z]
                # cam_base_quat = np.array([cam_base_quat[1], cam_base_quat[2], cam_base_quat[3], cam_base_quat[0]]) # [x, y, z, w]
                # cam_base_rot = R.from_quat(cam_base_quat).as_matrix()

                cam_rot = R.from_euler("xyz", [0, 0, 90], degrees=True).as_matrix()
                shift = np.array([0, -0.0, 0.0], dtype=np.float64)  # right side
                pos = self.env.sim.data.get_body_xpos("gripper0_right_eef") + cam_rot.dot(shift)
                next_line = np.array([0, 0, -0.02], dtype=np.float64)
            else:
                cam_rot = R.from_euler("xyz", [0, -self.viewer.cam.elevation, self.viewer.cam.azimuth], degrees=True).as_matrix()
                shift = np.array([0, -0.3, 0.3], dtype=np.float64) * self.viewer.cam.distance  # Top left corner
                shift[0] = self.viewer.cam.distance
                pos = self.viewer.cam.lookat + cam_rot.dot(shift)
                next_line = np.array([0, 0, -0.03], dtype=np.float64) * self.viewer.cam.distance

            for line in self.texts:
                # Get the next available geom from the viewer's scene.
                geom = self.viewer.user_scn.geoms[self.viewer.user_scn.ngeom]
                mujoco.mjv_initGeom(
                    geom,
                    type=mujoco.mjtGeom.mjGEOM_LABEL,
                    pos=pos,  # current text position (world frame)
                    mat=cam_rot.flatten(),  # use the orientation for the label
                    size=np.ones(3),
                    rgba=np.zeros(4),
                )
                # Set the label text
                geom.label = line
                self.viewer.user_scn.ngeom += 1

                # Update pos_i: transform a local offset into the world frame and add it
                pos = pos + cam_rot.dot(next_line)

    def mjshowframe(self, xyz, quat, size=0.1, name=None, keep=False):
        """
        Queue a coordinate frame at world-frame `xyz` with quaternion `quat` (x,y,z,w).
        Will be drawn on the next update().
        """
        if keep and name is not None:
            self.frames_to_keep.add(name)
        elif not keep and name in self.frames_to_keep:
            self.frames_to_keep.remove(name)
        self.frames.append((np.array(xyz, dtype=float), np.array(quat, dtype=float), float(size), name))

    def _mjshowframe(self):
        if self.viewer is not None:
            # draw each queued frame as three colored arrows
            for pos, quat, size, name in self.frames:
                # (x,y,z,w) → rotation matrix
                mat3 = R.from_quat(quat).as_matrix()
                flat = mat3.flatten()

                for axis, color in enumerate([[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]]):
                    geom = self.viewer.user_scn.geoms[self.viewer.user_scn.ngeom]
                    mujoco.mjv_initGeom(
                        geom,
                        type=mujoco.mjtGeom.mjGEOM_ARROW,
                        pos=pos,
                        mat=flat,
                        size=[0.003, 0.003, size],
                        rgba=color,
                    )
                    # adjust orientation for X and Y arrows
                    if axis == 0:  # X axis: rotate -90° about Y
                        extra = R.from_euler("y", -90, degrees=True).as_matrix()
                        new_flat = (mat3 @ extra).flatten()
                        geom.mat[:, :] = new_flat.reshape(3, 3)
                    elif axis == 1:  # Y axis: rotate +90° about X
                        extra = R.from_euler("x", 90, degrees=True).as_matrix()
                        new_flat = (mat3 @ extra).flatten()
                        geom.mat[:, :] = new_flat.reshape(3, 3)
                    # Z axis uses flat as set
                    if name:
                        geom.label = name
                    self.viewer.user_scn.ngeom += 1
