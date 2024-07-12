# Adapted from ARM
# Source: https://github.com/stepjam/ARM
# License: https://github.com/stepjam/ARM/LICENSE

import logging
from typing import List

import numpy as np
from omegaconf import DictConfig
from rlbench.backend.observation import Observation
from rlbench.observation_config import ObservationConfig
import rlbench.utils as rlbench_utils
from rlbench.demo import Demo
from yarr.replay_buffer.prioritized_replay_buffer import \
    PrioritizedReplayBuffer, ObservationElement
from yarr.replay_buffer.replay_buffer import ReplayElement, ReplayBuffer
from yarr.replay_buffer.uniform_replay_buffer import UniformReplayBuffer
from yarr.replay_buffer.task_uniform_replay_buffer import TaskUniformReplayBuffer

from helpers import utils
from helpers import observation_utils
from agents.act_bc_lang.act_bc_lang_agent import ActBCLangAgent
from helpers.custom_rlbench_env import CustomRLBenchEnv
from helpers.preprocess_agent import PreprocessAgent
from agents.act_bc_lang.act_policy import ACTPolicy, CNNMLPPolicy

import torch
from torch.multiprocessing import Process, Value, Manager
from helpers.clip.core.clip import build_model, load_clip, tokenize

LOW_DIM_SIZE = 8


def create_replay(batch_size: int, timesteps: int,
                  prioritisation: bool, task_uniform: bool,
                  save_dir: str, cameras: list,
                  image_size=[128, 128],
                  replay_size=3e5,
                  prev_action_horizon: int = 1,
                  next_action_horizon: int = 1):
    
    lang_feat_dim = 1024

    # low_dim_state
    observation_elements = []
    observation_elements.append(
        ObservationElement('low_dim_state', (LOW_DIM_SIZE,), np.float32))

    # action sequences
    action_seq_sizes = {'right_prev_joint_positions': 7 ,
                        'right_prev_gripper_joint_positions': 2,
                        'right_prev_gripper_poses': 7,

                        'right_next_joint_positions': 7,
                        'right_next_gripper_joint_positions': 2,
                        'right_next_gripper_poses': 7,

                        'left_prev_joint_positions': 7 ,                        
                        'left_prev_gripper_joint_positions': 2,
                        'left_prev_gripper_poses': 7,

                        'left_next_joint_positions': 7,
                        'left_next_gripper_joint_positions': 2,
                        'left_next_gripper_poses': 7}


    for seq_name, seq_size in action_seq_sizes.items():
        horizon = prev_action_horizon if 'prev' in seq_name else next_action_horizon
        observation_elements.append(
            ObservationElement(seq_name, (horizon, seq_size,), np.float32))

    # action is_pad
    observation_elements.append(
        ObservationElement('is_pad', (next_action_horizon,), np.int32))


    # rgb, depth, point cloud, intrinsics, extrinsics
    for cname in cameras:
        observation_elements.append(
            ObservationElement('%s_rgb' % cname, (3, *image_size,), np.float32))
        observation_elements.append(
            ObservationElement('%s_point_cloud' % cname, (3, *image_size),
                               np.float32))  # see pyrep/objects/vision_sensor.py on how pointclouds are extracted from depth frames
        observation_elements.append(
            ObservationElement('%s_camera_extrinsics' % cname, (4, 4,), np.float32))
        observation_elements.append(
            ObservationElement('%s_camera_intrinsics' % cname, (3, 3,), np.float32))

    observation_elements.extend([
        ReplayElement('lang_goal_emb', (lang_feat_dim,),
                      np.float32),
        ReplayElement('task', (),
                      str),
        ReplayElement('lang_goal', (1,),
                      object),  # language goal string for debugging and visualization
    ])

    extra_replay_elements = [
        ReplayElement('demo', (), bool),
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
        extra_replay_elements=extra_replay_elements
    )
    return replay_buffer


def _get_action(obs_tp1: Observation):
    quat = utils.normalize_quaternion(obs_tp1.gripper_pose[3:])
    if quat[-1] < 0:
        quat = -quat
    return np.concatenate([obs_tp1.gripper_pose[:3], quat,
                           [float(obs_tp1.gripper_open)]])


def _get_action_seq(demo: Demo,
                    timestep: int,
                    prev_action_horizon: int,
                    next_action_horizon: int,
                    robot_name: str):

    action_seq = {
        'right_prev_joint_positions': [],
        'right_prev_gripper_joint_positions': [],
        'right_prev_gripper_poses': [],
        'left_prev_joint_positions': [],
        'left_prev_gripper_joint_positions': [],
        'left_prev_gripper_poses': [],

        'right_next_joint_positions': [],
        'right_next_gripper_joint_positions': [],
        'right_next_gripper_poses': [],
        'left_next_joint_positions': [],
        'left_next_gripper_joint_positions': [],
        'left_next_gripper_poses': [],

        'is_pad': [],
    }

    for prev_t in list(reversed(range(prev_action_horizon))):
        t = timestep - prev_t
        t = max(0, t)
        obs = demo[t]

        action_seq['right_prev_joint_positions'].append(obs.right.joint_positions)                
        action_seq['right_prev_gripper_joint_positions'].append(obs.right.gripper_joint_positions)
        action_seq['right_prev_gripper_poses'].append(obs.right.gripper_pose)        
        action_seq['left_prev_joint_positions'].append(obs.left.joint_positions)
        action_seq['left_prev_gripper_joint_positions'].append(obs.left.gripper_joint_positions)        
        action_seq['left_prev_gripper_poses'].append(obs.left.gripper_pose)
       

    action_seq['is_pad'] = np.zeros(next_action_horizon)
    for idx, next_t in enumerate(range(0, next_action_horizon)):
        t = timestep + next_t
        t = min(t, len(demo)-1)
        obs = demo[t]

        if timestep + next_t > len(demo)-1:
            action_seq['is_pad'][idx] = 1
   
        action_seq['right_next_joint_positions'].append(obs.right.joint_positions)
        action_seq['right_next_gripper_joint_positions'].append(obs.right.gripper_joint_positions)
        action_seq['right_next_gripper_poses'].append(obs.right.gripper_pose)      
        action_seq['left_next_joint_positions'].append(obs.left.joint_positions)
        action_seq['left_next_gripper_joint_positions'].append(obs.left.gripper_joint_positions)
        action_seq['left_next_gripper_poses'].append(obs.left.gripper_pose)             
       

    # convert to numpy arrays
    return {k: np.array(v) for k, v in action_seq.items()}



def _add_keypoints_to_replay(
        step: int,
        cfg: DictConfig,
        task: str,
        replay: ReplayBuffer,
        inital_obs: Observation,
        demo: Demo,
        description: str = '',
        clip_model = None,
        device = 'cpu'):
    

    cameras = cfg.rlbench.cameras
    robot_name = cfg.method.robot_name

    prev_action = None
    obs = inital_obs
    all_actions = []
    k = step
    k_tp1 = min(k+1, len(demo) - 1)
    obs_tp1 = demo[k_tp1]

    if obs_tp1.is_bimanual and robot_name == "bimanual":
        right_action = _get_action(obs_tp1.right)
        left_action = _get_action(obs_tp1.left)
        action = np.append(right_action, left_action)
    elif robot_name == "unimanual":
        action = _get_action(obs_tp1)
    elif obs_tp1.is_bimanual and robot_name == "right":
        action = _get_action(obs_tp1.right)
    elif obs_tp1.is_bimanual and  robot_name == "left":
        action = _get_action(obs_tp1.left)
    else:
        logging.error("Invalid robot name %s", cfg.method.robot_name)
        raise Exception("Invalid robot name.")


    all_actions.append(action)

    terminal = (k == len(demo) - 1)
    reward = float(terminal) if terminal else 0

    obs_dict = observation_utils.extract_obs(obs, t=k, prev_action=prev_action,
                                 cameras=cameras, episode_length=cfg.rlbench.episode_length, robot_name=robot_name)
    

    if  obs_tp1.is_bimanual and robot_name == "bimanual":
        obs_dict['low_dim_state'] = np.concatenate([obs_dict['right_low_dim_state'], obs_dict['left_low_dim_state']])
        del obs_dict['right_low_dim_state']
        del obs_dict['left_low_dim_state']
        del obs_dict['right_ignore_collisions']
        del obs_dict['left_ignore_collisions']
    else:
        del obs_dict['ignore_collisions']


    tokens = tokenize([description]).numpy()
    token_tensor = torch.from_numpy(tokens).to(device)
    lang_feats, lang_embs = clip_model.encode_text_with_embeddings(token_tensor)
    obs_dict['lang_goal_emb'] = lang_feats[0].float().detach().cpu().numpy()

    final_obs = {
        'task': task,
        'lang_goal': np.array([description], dtype=object),
    }

    action_seq = _get_action_seq(demo,
                                 step,
                                 cfg.method.prev_action_horizon,
                                 cfg.method.next_action_horizon,
                                 robot_name)
    obs_dict.update(action_seq)

    prev_action = np.copy(action)
    others = {'demo': True}
    others.update(final_obs)
    others.update(obs_dict)
    timeout = False
    replay.add(action, reward, terminal, timeout, **others)

    return all_actions


def fill_replay(cfg: DictConfig,
                obs_config: ObservationConfig,
                rank: int,
                replay: ReplayBuffer,
                task: str,
                num_demos: int,
                demo_augmentation: bool,
                demo_augmentation_every_n: int,
                cameras: List[str],
                clip_model = None,
                device = 'cpu'):

    if clip_model is None:
        model, _ = load_clip('RN50', jit=False, device=device)
        clip_model = build_model(model.state_dict())
        clip_model.to(device)
        del model

    logging.debug('Filling %s replay ...' % task)
    all_actions = []
    for d_idx in range(num_demos):
        # load demo from disk
        demo = rlbench_utils.get_stored_demos(
            amount=1, image_paths=False,
            dataset_root=cfg.rlbench.demo_path,
            variation_number=-1, task_name=task,
            obs_config=obs_config,
            random_selection=False,
            from_episode_number=d_idx)[0]

        descs = demo._observations[0].misc['descriptions']

        if rank == 0:
            logging.info(f"Loading Demo({d_idx})")

        for i in range(len(demo) - 1):
            obs = demo[i]
            desc = descs[0]

            # stopped = np.allclose(obs.joint_velocities, 0, atol=0.1)
            # if stopped:
            #     continue

            all_actions.extend(_add_keypoints_to_replay(
                i, cfg, task, replay, obs, demo,
                description=desc, clip_model=clip_model, device=device))
    logging.debug('Replay filled with demos.')
    return all_actions


def fill_multi_task_replay(cfg: DictConfig,
                           obs_config: ObservationConfig,
                           rank: int,
                           replay: ReplayBuffer,
                           tasks: List[str],
                           num_demos: int,
                           demo_augmentation: bool,
                           demo_augmentation_every_n: int,
                           cameras: List[str],
                           clip_model = None):
    manager = Manager()
    store = manager.dict()

    # create a MP dict for storing indicies
    # TODO(mohit): this shouldn't be initialized here
    del replay._task_idxs
    task_idxs = manager.dict()
    replay._task_idxs = task_idxs
    replay._create_storage(store)
    replay.add_count = Value('i', 0)

    # fill replay buffer in parallel across tasks
    max_parallel_processes = cfg.replay.max_parallel_processes
    processes = []
    n = np.arange(len(tasks))
    split_n = utils.split_list(n, max_parallel_processes)
    for split in split_n:
        for e_idx, task_idx in enumerate(split):
            task = tasks[int(task_idx)]
            model_device = torch.device('cuda:%s' % (e_idx % torch.cuda.device_count())
                                        if torch.cuda.is_available() else 'cpu')
            p = Process(target=fill_replay,
                        args=(cfg,
                              obs_config,
                              rank,
                              replay,
                              task,
                              num_demos,
                              demo_augmentation,
                              demo_augmentation_every_n,
                              cameras,
                              clip_model,
                              model_device))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

    logging.debug('Replay filled with multi demos.')


def create_agent(cfg: DictConfig):
    actor_net = ACTPolicy(cfg.method)

    bc_agent = ActBCLangAgent(
        actor_network=actor_net,
        camera_names=cfg.rlbench.cameras,
        lr=cfg.method.lr,
        weight_decay=cfg.method.weight_decay,
        grad_clip=cfg.method.grad_clip,
        episode_length=cfg.rlbench.episode_length,
        train_demo_path=cfg.method.train_demo_path,
        task_name=cfg.rlbench.tasks[0])

    return PreprocessAgent(pose_agent=bc_agent,
                           norm_type='imagenet')
