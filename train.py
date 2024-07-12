from typing import List
import logging
import os
import sys
from datetime import datetime

import peract_config

import hydra
from omegaconf import DictConfig, OmegaConf, ListConfig

import run_seed_fn
from helpers.observation_utils import create_obs_config

import torch.multiprocessing as mp


@hydra.main(config_name="config", config_path="conf")
def main(cfg: DictConfig) -> None:

    cfg_yaml = OmegaConf.to_yaml(cfg)
    logging.info("\n" + cfg_yaml)

    peract_config.on_config(cfg)

    cfg.rlbench.cameras = (
        cfg.rlbench.cameras
        if isinstance(cfg.rlbench.cameras, ListConfig)
        else [cfg.rlbench.cameras]
    )

    # sanity check if rgb is not used as camera name   
    for camera_name in cfg.rlbench.cameras:
        assert("rgb" not in camera_name)

    obs_config = create_obs_config(
        cfg.rlbench.cameras, cfg.rlbench.camera_resolution, cfg.method.name
    )

    cwd = os.getcwd()
    logging.info("CWD:" + os.getcwd())

    if cfg.framework.start_seed >= 0:
        # seed specified
        start_seed = cfg.framework.start_seed
    elif (
        cfg.framework.start_seed == -1
        and len(list(filter(lambda x: "seed" in x, os.listdir(cwd)))) > 0
    ):
        # unspecified seed; use largest existing seed plus one
        largest_seed = max(
            [
                int(n.replace("seed", ""))
                for n in list(filter(lambda x: "seed" in x, os.listdir(cwd)))
            ]
        )
        start_seed = largest_seed + 1
    else:
        # start with seed 0
        start_seed = 0

    seed_folder = os.path.join(os.getcwd(), "seed%d" % start_seed)
    os.makedirs(seed_folder, exist_ok=True)

    start_time = datetime.now()
    with open(os.path.join(seed_folder, "config.yaml"), "w") as f:
        f.write(cfg_yaml)


    # check if previous checkpoints already exceed the number of desired training iterations
    # if so, exit the script
    latest_weight = 0
    weights_folder = os.path.join(seed_folder, "weights")
    if os.path.isdir(weights_folder) and len(os.listdir(weights_folder)) > 0:
        weights = os.listdir(weights_folder)
        latest_weight = sorted(map(int, weights))[-1]
        if latest_weight >= cfg.framework.training_iterations:
            logging.info(
                "Agent was already trained for %d iterations. Exiting." % latest_weight
            )
            sys.exit(0)


    with open(os.path.join(seed_folder, "training.log"), "a") as f:

        f.write(f"# Starting training from weights: {latest_weight} to {cfg.framework.training_iterations}")
        f.write(f"# Training started on: {start_time.isoformat()}")
        f.write(os.linesep)


    # run train jobs with multiple seeds (sequentially)
    for seed in range(start_seed, start_seed + cfg.framework.seeds):
        
        logging.info("Starting seed %d." % seed)

        world_size = cfg.ddp.num_devices
        mp.spawn(
            run_seed_fn.run_seed,
            args=(
                cfg,
                obs_config,
                seed,
                world_size,
            ),
            nprocs=world_size,
            join=True,
        )

    end_time = datetime.now()
    duration = (end_time - start_time)
    with open(os.path.join(seed_folder, "training.log"), "a") as f:
        f.write(f"# Training finished on: {end_time.isoformat()}")
        f.write(f"# Took {duration.total_seconds()}")
        f.write(os.linesep)
        f.write(os.linesep)

if __name__ == "__main__":
    peract_config.on_init()
    main()
