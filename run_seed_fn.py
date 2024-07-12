import os
import pickle
import gc
from typing import List

import hydra
import numpy as np
import torch
from omegaconf import DictConfig

from rlbench import CameraConfig, ObservationConfig
from yarr.replay_buffer.wrappers.pytorch_replay_buffer import PyTorchReplayBuffer
from yarr.runners.offline_train_runner import OfflineTrainRunner
from yarr.utils.stat_accumulator import SimpleAccumulator

from helpers.custom_rlbench_env import CustomRLBenchEnv, CustomMultiTaskRLBenchEnv
import torch.distributed as dist

from agents import agent_factory
from agents import replay_utils

import peract_config
from functools import partial

def run_seed(
    rank,
    cfg: DictConfig,
    obs_config: ObservationConfig,
    seed,
    world_size,
) -> None:
    

    peract_config.config_logging()
    
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    tasks = cfg.rlbench.tasks
    cams = cfg.rlbench.cameras

    task_folder = "multi" if len(tasks) > 1 else tasks[0] 
    replay_path = os.path.join(
        cfg.replay.path, task_folder, cfg.method.name, "seed%d" % seed
    )

    agent = agent_factory.create_agent(cfg)

    if not agent:
        print("Unable to create agent")
        return

    if cfg.method.name == "ARM":
        raise NotImplementedError("ARM is not supported yet")
    elif cfg.method.name == "BC_LANG":
        from agents.baselines import bc_lang

        assert cfg.ddp.num_devices == 1, "BC_LANG only supports single GPU training"
        replay_buffer = bc_lang.launch_utils.create_replay(
            cfg.replay.batch_size,
            cfg.replay.timesteps,
            cfg.replay.prioritisation,
            cfg.replay.task_uniform,
            replay_path if cfg.replay.use_disk else None,
            cams,
            cfg.rlbench.camera_resolution,
        )

        bc_lang.launch_utils.fill_multi_task_replay(
            cfg,
            obs_config,
            rank,
            replay_buffer,
            tasks,
            cfg.rlbench.demos,
            cfg.method.demo_augmentation,
            cfg.method.demo_augmentation_every_n,
            cams,
        )


    elif cfg.method.name == "VIT_BC_LANG":
        from agents.baselines import vit_bc_lang

        assert cfg.ddp.num_devices == 1, "VIT_BC_LANG only supports single GPU training"
        replay_buffer = vit_bc_lang.launch_utils.create_replay(
            cfg.replay.batch_size,
            cfg.replay.timesteps,
            cfg.replay.prioritisation,
            cfg.replay.task_uniform,
            replay_path if cfg.replay.use_disk else None,
            cams,
            cfg.rlbench.camera_resolution,
        )

        vit_bc_lang.launch_utils.fill_multi_task_replay(
            cfg,
            obs_config,
            rank,
            replay_buffer,
            tasks,
            cfg.rlbench.demos,
            cfg.method.demo_augmentation,
            cfg.method.demo_augmentation_every_n,
            cams,
        )

    elif cfg.method.name.startswith("ACT_BC_LANG"):
        from agents import act_bc_lang

        assert cfg.ddp.num_devices == 1, "ACT_BC_LANG only supports single GPU training"
        replay_buffer = act_bc_lang.launch_utils.create_replay(
            cfg.replay.batch_size,
            cfg.replay.timesteps,
            cfg.replay.prioritisation,
            cfg.replay.task_uniform,
            replay_path if cfg.replay.use_disk else None,
            cams,
            cfg.rlbench.camera_resolution,
            replay_size=3e5,
            prev_action_horizon=cfg.method.prev_action_horizon,
            next_action_horizon=cfg.method.next_action_horizon
        )

        act_bc_lang.launch_utils.fill_multi_task_replay(
            cfg,
            obs_config,
            rank,
            replay_buffer,
            tasks,
            cfg.rlbench.demos,
            cfg.method.demo_augmentation,
            cfg.method.demo_augmentation_every_n,
            cams,
        )

    elif cfg.method.name == "C2FARM_LINGUNET_BC":
        from agents import c2farm_lingunet_bc

        replay_buffer = c2farm_lingunet_bc.launch_utils.create_replay(
            cfg.replay.batch_size,
            cfg.replay.timesteps,
            cfg.replay.prioritisation,
            cfg.replay.task_uniform,
            replay_path if cfg.replay.use_disk else None,
            cams,
            cfg.method.voxel_sizes,
            cfg.rlbench.camera_resolution,
        )

        c2farm_lingunet_bc.launch_utils.fill_multi_task_replay(
            cfg,
            obs_config,
            rank,
            replay_buffer,
            tasks,
            cfg.rlbench.demos,
            cfg.method.demo_augmentation,
            cfg.method.demo_augmentation_every_n,
            cams,
            cfg.rlbench.scene_bounds,
            cfg.method.voxel_sizes,
            cfg.method.bounds_offset,
            cfg.method.rotation_resolution,
            cfg.method.crop_augmentation,
            keypoint_method=cfg.method.keypoint_method,
        )


    elif cfg.method.name.startswith("BIMANUAL_PERACT") or cfg.method.name.startswith("RVT") or cfg.method.name.startswith("PERACT_BC"):

        replay_buffer = replay_utils.create_replay(cfg, replay_path)
        
        replay_utils.fill_multi_task_replay(
            cfg,
            obs_config,
            rank,
            replay_buffer,
            tasks
        )

    elif cfg.method.name == "PERACT_RL":
        raise NotImplementedError("PERACT_RL is not supported yet")

    else:
        raise ValueError("Method %s does not exists." % cfg.method.name)

    wrapped_replay = PyTorchReplayBuffer(
        replay_buffer, num_workers=cfg.framework.num_workers
    )
    stat_accum = SimpleAccumulator(eval_video_fps=30)

    cwd = os.getcwd()
    weightsdir = os.path.join(cwd, "seed%d" % seed, "weights")
    logdir = os.path.join(cwd, "seed%d" % seed)

    train_runner = OfflineTrainRunner(
        agent=agent,
        wrapped_replay_buffer=wrapped_replay,
        train_device=rank,
        stat_accumulator=stat_accum,
        iterations=cfg.framework.training_iterations,
        logdir=logdir,
        logging_level=cfg.framework.logging_level,
        log_freq=cfg.framework.log_freq,
        weightsdir=weightsdir,
        num_weights_to_keep=cfg.framework.num_weights_to_keep,
        save_freq=cfg.framework.save_freq,
        tensorboard_logging=cfg.framework.tensorboard_logging,
        csv_logging=cfg.framework.csv_logging,
        load_existing_weights=cfg.framework.load_existing_weights,
        rank=rank,
        world_size=world_size,
    )

    train_runner._on_thread_start = partial(peract_config.config_logging, cfg.framework.logging_level)
    
    train_runner.start()

    del train_runner
    del agent
    gc.collect()
    torch.cuda.empty_cache()
