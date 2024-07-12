
import logging
from typing import List

import numpy as np
from rlbench.backend.observation import Observation
from rlbench.observation_config import ObservationConfig
import rlbench.utils as rlbench_utils
from rlbench.demo import Demo
from yarr.replay_buffer.replay_buffer import ReplayBuffer

from helpers import demo_loading_utils, utils
from helpers import observation_utils
from helpers.clip.core.clip import tokenize


from yarr.replay_buffer.prioritized_replay_buffer import ObservationElement
from yarr.replay_buffer.replay_buffer import ReplayElement
from yarr.replay_buffer.task_uniform_replay_buffer import TaskUniformReplayBuffer


import torch
from torch.multiprocessing import Process, Value, Manager
from helpers.clip.core.clip import build_model, load_clip
from omegaconf import DictConfig


REWARD_SCALE = 100.0
LOW_DIM_SIZE = 4


def create_replay(cfg, replay_path):
    
    if cfg.method.robot_name == "bimanual":
        return create_bimanual_replay(            
            cfg.replay.batch_size,
            cfg.replay.timesteps,
            cfg.replay.prioritisation,
            cfg.replay.task_uniform,
            replay_path if cfg.replay.use_disk else None,
            cfg.rlbench.cameras,
            cfg.method.voxel_sizes,
            cfg.rlbench.camera_resolution,
        )
    else:
        return create_unimanual_replay(            
            cfg.replay.batch_size,
            cfg.replay.timesteps,
            cfg.replay.prioritisation,
            cfg.replay.task_uniform,
            replay_path if cfg.replay.use_disk else None,
            cfg.rlbench.cameras,
            cfg.method.voxel_sizes,
            cfg.rlbench.camera_resolution,
        )



def create_bimanual_replay(
    batch_size: int,
    timesteps: int,
    prioritisation: bool,
    task_uniform: bool,
    save_dir: str,
    cameras: list,
    voxel_sizes,
    image_size=[128, 128],
    replay_size=3e5,
):
    trans_indicies_size = 3 * len(voxel_sizes)
    rot_and_grip_indicies_size = 3 + 1
    gripper_pose_size = 7
    ignore_collisions_size = 1
    max_token_seq_len = 77
    lang_feat_dim = 1024
    lang_emb_dim = 512

    # low_dim_state
    observation_elements = []
    observation_elements.append(
        ObservationElement("right_low_dim_state", (LOW_DIM_SIZE,), np.float32)
    )
    observation_elements.append(
        ObservationElement("left_low_dim_state", (LOW_DIM_SIZE,), np.float32)
    )

    # rgb, depth, point cloud, intrinsics, extrinsics
    for cname in cameras:
        observation_elements.append(
            # color, height, width
            ObservationElement(
                "%s_rgb" % cname,
                (
                    3,
                    image_size[1],
                    image_size[0],
                ),
                np.float32,
            )
        )
        observation_elements.append(
            ObservationElement("%s_point_cloud" % cname, (3, image_size[1], image_size[0]), np.float16)
        )  # see pyrep/objects/vision_sensor.py on how pointclouds are extracted from depth frames
        observation_elements.append(
            ObservationElement(
                "%s_camera_extrinsics" % cname,
                (
                    4,
                    4,
                ),
                np.float32,
            )
        )
        observation_elements.append(
            ObservationElement(
                "%s_camera_intrinsics" % cname,
                (
                    3,
                    3,
                ),
                np.float32,
            )
        )

    # discretized translation, discretized rotation, discrete ignore collision, 6-DoF gripper pose, and pre-trained language embeddings
    for robot_name in ["right", "left"]:
        observation_elements.extend(
            [
                ReplayElement(
                    f"{robot_name}_trans_action_indicies",
                    (trans_indicies_size,),
                    np.int32,
                ),
                ReplayElement(
                    f"{robot_name}_rot_grip_action_indicies",
                    (rot_and_grip_indicies_size,),
                    np.int32,
                ),
                ReplayElement(
                    f"{robot_name}_ignore_collisions",
                    (ignore_collisions_size,),
                    np.int32,
                ),
                ReplayElement(
                    f"{robot_name}_gripper_pose", (gripper_pose_size,), np.float32
                ),
            ]
        )

    observation_elements.extend(
        [
            ReplayElement("lang_goal_emb", (lang_feat_dim,), np.float32),
            ReplayElement(
                "lang_token_embs",
                (
                    max_token_seq_len,
                    lang_emb_dim,
                ),
                np.float32,
            ),  # extracted from CLIP's language encoder
            ReplayElement("task", (), str),
            ReplayElement(
                "lang_goal", (1,), object
            ),  # language goal string for debugging and visualization
        ]
    )

    extra_replay_elements = [
        ReplayElement("demo", (), bool),
    ]

    replay_buffer = TaskUniformReplayBuffer(
        save_dir=save_dir,
        batch_size=batch_size,
        timesteps=timesteps,
        replay_capacity=int(replay_size),
        action_shape=(8 * 2,),
        action_dtype=np.float32,
        reward_shape=(),
        reward_dtype=np.float32,
        update_horizon=1,
        observation_elements=observation_elements,
        extra_replay_elements=extra_replay_elements,
    )
    return replay_buffer




def create_unimanual_replay(
    batch_size: int,
    timesteps: int,
    prioritisation: bool,
    task_uniform: bool,
    save_dir: str,
    cameras: list,
    voxel_sizes,
    image_size=[128, 128],
    replay_size=3e5,
):
    trans_indicies_size = 3 * len(voxel_sizes)
    rot_and_grip_indicies_size = 3 + 1
    gripper_pose_size = 7
    ignore_collisions_size = 1
    max_token_seq_len = 77
    lang_feat_dim = 1024
    lang_emb_dim = 512

    # low_dim_state
    observation_elements = []
    observation_elements.append(
        ObservationElement("low_dim_state", (LOW_DIM_SIZE,), np.float32)
    )

    # rgb, depth, point cloud, intrinsics, extrinsics
    for cname in cameras:
        observation_elements.append(
            ObservationElement(
                "%s_rgb" % cname,
                (
                    3,
                    *image_size,
                ),
                np.float32,
            )
        )
        observation_elements.append(
            ObservationElement("%s_point_cloud" % cname, (3, *image_size), np.float32)
        )  # see pyrep/objects/vision_sensor.py on how pointclouds are extracted from depth frames
        observation_elements.append(
            ObservationElement(
                "%s_camera_extrinsics" % cname,
                (
                    4,
                    4,
                ),
                np.float32,
            )
        )
        observation_elements.append(
            ObservationElement(
                "%s_camera_intrinsics" % cname,
                (
                    3,
                    3,
                ),
                np.float32,
            )
        )

    # discretized translation, discretized rotation, discrete ignore collision, 6-DoF gripper pose, and pre-trained language embeddings
    observation_elements.extend(
        [
            ReplayElement("trans_action_indicies", (trans_indicies_size,), np.int32),
            ReplayElement(
                "rot_grip_action_indicies", (rot_and_grip_indicies_size,), np.int32
            ),
            ReplayElement("ignore_collisions", (ignore_collisions_size,), np.int32),
            ReplayElement("gripper_pose", (gripper_pose_size,), np.float32),
            ReplayElement("lang_goal_emb", (lang_feat_dim,), np.float32),
            ReplayElement(
                "lang_token_embs",
                (
                    max_token_seq_len,
                    lang_emb_dim,
                ),
                np.float32,
            ),  # extracted from CLIP's language encoder
            ReplayElement("task", (), str),
            ReplayElement(
                "lang_goal", (1,), object
            ),  # language goal string for debugging and visualization
        ]
    )

    extra_replay_elements = [
        ReplayElement("demo", (), bool),
    ]

    replay_buffer = TaskUniformReplayBuffer(
        save_dir=save_dir,
        batch_size=batch_size,
        timesteps=timesteps,
        replay_capacity=int(replay_size),
        action_shape=(8,),
        action_dtype=np.float32,
        reward_shape=(),
        reward_dtype=np.float32,
        update_horizon=1,
        observation_elements=observation_elements,
        extra_replay_elements=extra_replay_elements,
    )
    return replay_buffer



def _get_action(
    obs_tp1: Observation,
    obs_tm1: Observation,
    rlbench_scene_bounds: List[float],  # metric 3D bounds of the scene
    voxel_sizes: List[int],
    bounds_offset: List[float],
    rotation_resolution: int,
    crop_augmentation: bool,
):
    quat = utils.normalize_quaternion(obs_tp1.gripper_pose[3:])
    if quat[-1] < 0:
        quat = -quat
    disc_rot = utils.quaternion_to_discrete_euler(quat, rotation_resolution)
    disc_rot = utils.correct_rotation_instability(disc_rot, rotation_resolution)

    attention_coordinate = obs_tp1.gripper_pose[:3]
    trans_indicies, attention_coordinates = [], []
    bounds = np.array(rlbench_scene_bounds)
    ignore_collisions = int(obs_tm1.ignore_collisions)
    for depth, vox_size in enumerate(
        voxel_sizes
    ):  # only single voxelization-level is used in PerAct
        if depth > 0:
            if crop_augmentation:
                shift = bounds_offset[depth - 1] * 0.75
                attention_coordinate += np.random.uniform(-shift, shift, size=(3,))
            bounds = np.concatenate(
                [
                    attention_coordinate - bounds_offset[depth - 1],
                    attention_coordinate + bounds_offset[depth - 1],
                ]
            )
        index = utils.point_to_voxel_index(obs_tp1.gripper_pose[:3], vox_size, bounds)
        trans_indicies.extend(index.tolist())
        res = (bounds[3:] - bounds[:3]) / vox_size
        attention_coordinate = bounds[:3] + res * index
        attention_coordinates.append(attention_coordinate)

    rot_and_grip_indicies = disc_rot.tolist()
    grip = float(obs_tp1.gripper_open)
    rot_and_grip_indicies.extend([int(obs_tp1.gripper_open)])
    return (
        trans_indicies,
        rot_and_grip_indicies,
        ignore_collisions,
        np.concatenate([obs_tp1.gripper_pose, np.array([grip])]),
        attention_coordinates,
    )


def _add_keypoints_to_replay(
    cfg: DictConfig,
    task: str,
    replay: ReplayBuffer,
    inital_obs: Observation,
    demo: Demo,
    episode_keypoints: List[int],

    description: str = "",
    clip_model=None,
    device="cpu",
):

    cameras = cfg.rlbench.cameras
    rlbench_scene_bounds = cfg.rlbench.scene_bounds
    voxel_sizes = cfg.method.voxel_sizes
    bounds_offset = cfg.method.bounds_offset
    rotation_resolution = cfg.method.rotation_resolution
    crop_augmentation = cfg.method.crop_augmentation
    robot_name = cfg.method.robot_name

    prev_action = None
    obs = inital_obs
    
    for k, keypoint in enumerate(episode_keypoints):
        obs_tp1 = demo[keypoint]
        obs_tm1 = demo[max(0, keypoint - 1)]

        if obs_tp1.is_bimanual and robot_name == "bimanual":
            #assert isinstance(obs_tp1, BimanualObservation)
            (
                right_trans_indicies,
                right_rot_grip_indicies,
                right_ignore_collisions,
                right_action,
                right_attention_coordinates,
            ) = _get_action(
                obs_tp1.right,
                obs_tm1.right,
                rlbench_scene_bounds,
                voxel_sizes,
                bounds_offset,
                rotation_resolution,
                crop_augmentation,
            )

            (
                left_trans_indicies,
                left_rot_grip_indicies,
                left_ignore_collisions,
                left_action,
                left_attention_coordinates,
            ) = _get_action(
                obs_tp1.left,
                obs_tm1.left,
                rlbench_scene_bounds,
                voxel_sizes,
                bounds_offset,
                rotation_resolution,
                crop_augmentation,
            )

            action = np.append(right_action, left_action)

            right_ignore_collisions = np.array([right_ignore_collisions])
            left_ignore_collisions = np.array([left_ignore_collisions])
            
        elif robot_name == "unimanual":
            (
                trans_indicies,
                rot_grip_indicies,
                ignore_collisions,
                action,
                attention_coordinates,
            ) = _get_action(
                obs_tp1,
                obs_tm1,
                rlbench_scene_bounds,
                voxel_sizes,
                bounds_offset,
                rotation_resolution,
                crop_augmentation,
            )
            gripper_pose = obs_tp1.gripper_pose
        elif obs_tp1.is_bimanual and robot_name == "right":
            (
                trans_indicies,
                rot_grip_indicies,
                ignore_collisions,
                action,
                attention_coordinates,
            ) = _get_action(
                obs_tp1.right,
                obs_tm1.right,
                rlbench_scene_bounds,
                voxel_sizes,
                bounds_offset,
                rotation_resolution,
                crop_augmentation,
            )
            gripper_pose = obs_tp1.right.gripper_pose
        elif obs_tp1.is_bimanual and robot_name == "left":
            (
                trans_indicies,
                rot_grip_indicies,
                ignore_collisions,
                action,
                attention_coordinates,
            ) = _get_action(
                obs_tp1.left,
                obs_tm1.left,
                rlbench_scene_bounds,
                voxel_sizes,
                bounds_offset,
                rotation_resolution,
                crop_augmentation,
            )
            gripper_pose = obs_tp1.left.gripper_pose
        else:
            logging.error("Invalid robot name %s", cfg.method.robot_name)
            raise Exception("Invalid robot name.")

        terminal = k == len(episode_keypoints) - 1
        reward = float(terminal) * REWARD_SCALE if terminal else 0

        obs_dict = observation_utils.extract_obs(
            obs,
            t=k,
            prev_action=prev_action,
            cameras=cameras,
            episode_length=cfg.rlbench.episode_length,
            robot_name=robot_name
        )
        tokens = tokenize([description]).numpy()
        token_tensor = torch.from_numpy(tokens).to(device)
        sentence_emb, token_embs = clip_model.encode_text_with_embeddings(token_tensor)
        obs_dict["lang_goal_emb"] = sentence_emb[0].float().detach().cpu().numpy()
        obs_dict["lang_token_embs"] = token_embs[0].float().detach().cpu().numpy()

        prev_action = np.copy(action)

        others = {"demo": True}
        if robot_name == "bimanual":

            final_obs = {
                "right_trans_action_indicies": right_trans_indicies,
                "right_rot_grip_action_indicies": right_rot_grip_indicies,
                "right_gripper_pose": obs_tp1.right.gripper_pose,
                "left_trans_action_indicies": left_trans_indicies,
                "left_rot_grip_action_indicies": left_rot_grip_indicies,
                "left_gripper_pose": obs_tp1.left.gripper_pose,
                "task": task,
                "lang_goal": np.array([description], dtype=object),
            }
        else:
            final_obs = {
                "trans_action_indicies": trans_indicies,
                "rot_grip_action_indicies": rot_grip_indicies,
                "gripper_pose": gripper_pose,
                "task": task,
                "lang_goal": np.array([description], dtype=object),
            }

        others.update(final_obs)
        others.update(obs_dict)

        timeout = False
        replay.add(action, reward, terminal, timeout, **others)
        obs = obs_tp1

    # final step
    obs_dict_tp1 = observation_utils.extract_obs(
        obs_tp1,
        t=k + 1,
        prev_action=prev_action,
        cameras=cameras,
        episode_length=cfg.rlbench.episode_length,
        robot_name=cfg.method.robot_name
    )
    obs_dict_tp1["lang_goal_emb"] = sentence_emb[0].float().detach().cpu().numpy()
    obs_dict_tp1["lang_token_embs"] = token_embs[0].float().detach().cpu().numpy()

    obs_dict_tp1.pop("wrist_world_to_cam", None)
    obs_dict_tp1.update(final_obs)
    replay.add_final(**obs_dict_tp1)


def fill_replay(
    cfg: DictConfig,
    obs_config: ObservationConfig,
    rank: int,
    replay: ReplayBuffer,
    task: str,
    clip_model=None,
    device="cpu",
):

    num_demos=cfg.rlbench.demos
    demo_augmentation=cfg.method.demo_augmentation
    demo_augmentation_every_n=cfg.method.demo_augmentation_every_n
    keypoint_method=cfg.method.keypoint_method


    if clip_model is None:
        model, _ = load_clip("RN50", jit=False, device=device)
        clip_model = build_model(model.state_dict())
        clip_model.to(device)
        del model

    logging.debug("Filling %s replay ..." % task)
    for d_idx in range(num_demos):
        # load demo from disk
        demo = rlbench_utils.get_stored_demos(
            amount=1,
            image_paths=False,
            dataset_root=cfg.rlbench.demo_path,
            variation_number=-1,
            task_name=task,
            obs_config=obs_config,
            random_selection=False,
            from_episode_number=d_idx,
        )[0]

        descs = demo._observations[0].misc["descriptions"]

        # extract keypoints (a.k.a keyframes)
        episode_keypoints = demo_loading_utils.keypoint_discovery(
            demo, method=keypoint_method
        )

        if rank == 0:
            logging.info(
                f"Loading Demo({d_idx}) - found {len(episode_keypoints)} keypoints - {task}"
            )

        for i in range(len(demo) - 1):
            if not demo_augmentation and i > 0:
                break
            if i % demo_augmentation_every_n != 0:
                continue

            obs = demo[i]
            desc = descs[0]
            # if our starting point is past one of the keypoints, then remove it
            while len(episode_keypoints) > 0 and i >= episode_keypoints[0]:
                episode_keypoints = episode_keypoints[1:]
            if len(episode_keypoints) == 0:
                break
            _add_keypoints_to_replay(
                cfg,
                task,
                replay,
                obs,
                demo,
                episode_keypoints,
                description=desc,
                clip_model=clip_model,
                device=device,
            )
    logging.debug("Replay %s filled with demos." % task)


def fill_multi_task_replay(
    cfg: DictConfig,
    obs_config: ObservationConfig,
    rank: int,
    replay: ReplayBuffer,
    tasks: List[str],
    clip_model=None,
):

    tasks = cfg.rlbench.tasks

    manager = Manager()
    store = manager.dict()

    # create a MP dict for storing indicies
    # TODO(mohit): this shouldn't be initialized here
    del replay._task_idxs
    task_idxs = manager.dict()
    replay._task_idxs = task_idxs
    replay._create_storage(store)
    replay.add_count = Value("i", 0)

    # fill replay buffer in parallel across tasks
    max_parallel_processes = cfg.replay.max_parallel_processes
    processes = []
    n = np.arange(len(tasks))
    split_n = utils.split_list(n, max_parallel_processes)
    for split in split_n:
        for e_idx, task_idx in enumerate(split):
            task = tasks[int(task_idx)]
            model_device = torch.device(
                "cuda:%s" % (e_idx % torch.cuda.device_count())
                if torch.cuda.is_available()
                else "cpu"
            )
            p = Process(
                target=fill_replay,
                args=(
                    cfg,
                    obs_config,
                    rank,
                    replay,
                    task,
                    clip_model,
                    model_device
                ),
            )

            p.start()
            processes.append(p)

        for p in processes:
            p.join()
