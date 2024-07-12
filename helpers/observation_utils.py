import numpy as np
from rlbench.backend.observation import Observation

from rlbench.backend.observation import BimanualObservation
from rlbench import CameraConfig, ObservationConfig
from pyrep.const import RenderMode
from typing import List

REMOVE_KEYS = [
    "joint_velocities",
    "joint_positions",
    "joint_forces",
    "gripper_open",
    "gripper_pose",
    "gripper_joint_positions",
    "gripper_touch_forces",
    "task_low_dim_state",
    "misc",
]


def extract_obs(
    obs: Observation,
    cameras,
    t: int = 0,
    prev_action=None,
    channels_last: bool = False,
    episode_length: int = 10,
    robot_name: str = ""
):
    if obs.is_bimanual:
        return extract_obs_bimanual(
            obs, cameras, t, prev_action, channels_last, episode_length, robot_name
        )
    else:
        return extract_obs_unimanual(
            obs, cameras, t, prev_action, channels_last, episode_length
        )


def extract_obs_unimanual(
    obs: Observation,
    cameras,
    t: int = 0,
    prev_action=None,
    channels_last: bool = False,
    episode_length: int = 10,
):
    obs.joint_velocities = None
    grip_mat = obs.gripper_matrix
    grip_pose = obs.gripper_pose
    joint_pos = obs.joint_positions
    obs.gripper_pose = None
    obs.gripper_matrix = None
    obs.joint_positions = None
    if obs.gripper_joint_positions is not None:
        obs.gripper_joint_positions = np.clip(obs.gripper_joint_positions, 0.0, 0.04)

    obs_dict = vars(obs)
    obs_dict = {k: v for k, v in obs_dict.items() if v is not None}
    robot_state = obs.get_low_dim_data()
    # remove low-level proprioception variables that are not needed
    obs_dict = {k: v for k, v in obs_dict.items() if k not in REMOVE_KEYS}

    if not channels_last:
        # swap channels from last dim to 1st dim
        obs_dict = {
            k: np.transpose(v, [2, 0, 1]) if v.ndim == 3 else np.expand_dims(v, 0)
            for k, v in obs.perception_data.items()
            if type(v) == np.ndarray or type(v) == list
        }
    else:
        # add extra dim to depth data
        obs_dict = {
            k: v if v.ndim == 3 else np.expand_dims(v, -1) for k, v in obs.perception_data.items()
        }
    obs_dict["low_dim_state"] = np.array(robot_state, dtype=np.float32)

    # binary variable indicating if collisions are allowed or not while planning paths to reach poses
    obs_dict["ignore_collisions"] = np.array([obs.ignore_collisions], dtype=np.float32)
    for k, v in [(k, v) for k, v in obs_dict.items() if "point_cloud" in k]:
        obs_dict[k] = v.astype(np.float32)

    for camera_name in cameras:
        obs_dict["%s_camera_extrinsics" % camera_name] = obs.misc[
            "%s_camera_extrinsics" % camera_name
        ]
        obs_dict["%s_camera_intrinsics" % camera_name] = obs.misc[
            "%s_camera_intrinsics" % camera_name
        ]

    # add timestep to low_dim_state
    time = (1.0 - (t / float(episode_length - 1))) * 2.0 - 1.0
    obs_dict["low_dim_state"] = np.concatenate(
        [obs_dict["low_dim_state"], [time]]
    ).astype(np.float32)

    obs.gripper_matrix = grip_mat
    obs.joint_positions = joint_pos
    obs.gripper_pose = grip_pose

    return obs_dict


def extract_obs_bimanual(
    obs: Observation,
    cameras,
    t: int = 0,
    prev_action = None,
    channels_last: bool = False,
    episode_length: int = 10,
    robot_name: str = "",
):
    obs.right.joint_velocities = None
    right_grip_mat = obs.right.gripper_matrix
    right_grip_pose = obs.right.gripper_pose
    right_joint_pos = obs.right.joint_positions
    obs.right.gripper_pose = None
    obs.right.gripper_matrix = None
    obs.right.joint_positions = None

    obs.left.joint_velocities = None
    left_grip_mat = obs.left.gripper_matrix
    left_grip_pose = obs.left.gripper_pose
    left_joint_pos = obs.left.joint_positions
    obs.left.gripper_pose = None
    obs.left.gripper_matrix = None
    obs.left.joint_positions = None

    if obs.right.gripper_joint_positions is not None:
        obs.right.gripper_joint_positions = np.clip(
            obs.right.gripper_joint_positions, 0.0, 0.04
        )
        obs.left.gripper_joint_positions = np.clip(
            obs.left.gripper_joint_positions, 0.0, 0.04
        )


    # fixme::
    obs_dict = vars(obs)
    obs_dict = {k: v for k, v in obs_dict.items() if v is not None}

    right_robot_state = obs.get_low_dim_data(obs.right)
    left_robot_state = obs.get_low_dim_data(obs.left)

    # remove low-level proprioception variables that are not needed
    obs_dict = {k: v for k, v in obs_dict.items() if k not in REMOVE_KEYS}

    if not channels_last:
        # swap channels from last dim to 1st dim
        obs_dict = {
            k: np.transpose(v, [2, 0, 1]) if v.ndim == 3 else np.expand_dims(v, 0)
            for k, v in obs.perception_data.items()
            if type(v) == np.ndarray or type(v) == list
        }
    else:
        # add extra dim to depth data
        obs_dict = {
            k: v if v.ndim == 3 else np.expand_dims(v, -1) for k, v in obs.perception_data.items()
        }

    if robot_name == "right":
        obs_dict["low_dim_state"] = right_robot_state.astype(np.float32)
        # binary variable indicating if collisions are allowed or not while planning paths to reach poses
        obs_dict["ignore_collisions"] = np.array(
            [obs.right.ignore_collisions], dtype=np.float32
        )
    elif robot_name == "left":
        obs_dict["low_dim_state"] = left_robot_state.astype(np.float32)
        obs_dict["ignore_collisions"] = np.array(
            [obs.left.ignore_collisions], dtype=np.float32
        )
    elif robot_name == "bimanual":
        obs_dict["right_low_dim_state"] = right_robot_state.astype(np.float32)
        obs_dict["left_low_dim_state"] = left_robot_state.astype(np.float32)
        obs_dict["right_ignore_collisions"] = np.array(
            [obs.right.ignore_collisions], dtype=np.float32
        )
        obs_dict["left_ignore_collisions"] = np.array(
            [obs.left.ignore_collisions], dtype=np.float32
        )

    for k, v in [(k, v) for k, v in obs_dict.items() if "point_cloud" in k]:
        # ..TODO:: switch to np.float16
        obs_dict[k] = v.astype(np.float32)

    for camera_name in cameras:
        obs_dict["%s_camera_extrinsics" % camera_name] = obs.misc[
            "%s_camera_extrinsics" % camera_name
        ]
        obs_dict["%s_camera_intrinsics" % camera_name] = obs.misc[
            "%s_camera_intrinsics" % camera_name
        ]

    # add timestep to low_dim_state
    time = (1.0 - (t / float(episode_length - 1))) * 2.0 - 1.0

    if "low_dim_state" in obs_dict:
        obs_dict["low_dim_state"] = np.concatenate(
            [obs_dict["low_dim_state"], [time]]
        ).astype(np.float32)
    else:
        obs_dict["right_low_dim_state"] = np.concatenate(
            [obs_dict["right_low_dim_state"], [time]]
        ).astype(np.float32)
        obs_dict["left_low_dim_state"] = np.concatenate(
            [obs_dict["left_low_dim_state"], [time]]
        ).astype(np.float32)

    obs.right.gripper_matrix = right_grip_mat
    obs.right.joint_positions = right_joint_pos
    obs.right.gripper_pose = right_grip_pose
    obs.left.gripper_matrix = left_grip_mat
    obs.left.joint_positions = left_joint_pos
    obs.left.gripper_pose = left_grip_pose

    return obs_dict


def create_obs_config(
    camera_names: List[str],
    camera_resolution: List[int],
    method_name: str,
    robot_name: str = "bimanual"
):
    unused_cams = CameraConfig()
    unused_cams.set_all(False)
    used_cams = CameraConfig(
        rgb=True,
        point_cloud=True,
        mask=False,
        depth=False,
        image_size=camera_resolution,
        render_mode=RenderMode.OPENGL,
    )

    camera_configs = {camera_name: used_cams for camera_name in camera_names}

    # Some of these obs are only used for keypoint detection.
    obs_config = ObservationConfig(
        camera_configs=camera_configs,
        joint_forces=False,
        joint_positions=True,
        joint_velocities=True,
        task_low_dim_state=False,
        gripper_touch_forces=False,
        gripper_pose=True,
        gripper_open=True,
        gripper_matrix=True,
        gripper_joint_positions=True,
        robot_name=robot_name
    )
    return obs_config
