"""
System configuration for peract 
"""
import os
import logging

import torch.multiprocessing as mp


def config_logging(logging_level=logging.INFO, reset=False):

    if reset:
        root = logging.getLogger()
        list(map(root.removeHandler, root.handlers))
        list(map(root.removeFilter, root.filters))
        
    from rich.logging import RichHandler
    logging.basicConfig(level=logging_level, handlers=[RichHandler()])


def on_init():

    config_logging(logging.INFO)
    
    logging.debug("Configuring environment.")
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    mp.set_start_method("spawn", force=True)
    mp.set_sharing_strategy("file_system")


def on_config(cfg):    
    
    os.environ["MASTER_ADDR"] = str(cfg.ddp.master_addr)
    os.environ["MASTER_PORT"] = str(cfg.ddp.master_port)
