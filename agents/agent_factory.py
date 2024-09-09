import os
import logging

from omegaconf import DictConfig


from yarr.agents.agent import BimanualAgent
from yarr.agents.agent import LeaderFollowerAgent
from yarr.agents.agent import Agent


supported_agents = {
    "leader_follower": ("PERACT_BC", "RVT"),
    "independent": ("PERACT_BC", "RVT"),
    "bimanual": ("BIMANUAL_PERACT", "ACT_BC_LANG"),
    "unimanual": (),
}


def create_agent(cfg: DictConfig) -> Agent:
    method_name = cfg.method.name
    agent_type = cfg.method.agent_type

    logging.info("Using method %s with type %s", method_name, agent_type)

    assert method_name in supported_agents[agent_type]

    agent_fn = agent_fn_by_name(method_name)

    if agent_type == "leader_follower":
        checkpoint_name_prefix = cfg.framework.checkpoint_name_prefix
        cfg.method.robot_name = "right"
        cfg.framework.checkpoint_name_prefix = (
            f"{checkpoint_name_prefix}_{method_name.lower()}_leader"
        )
        leader_agent = agent_fn(cfg)

        cfg.method.robot_name = "left"
        cfg.framework.checkpoint_name_prefix = (
            f"{checkpoint_name_prefix}_{method_name.lower()}_follower"
        )
        cfg.method.low_dim_size = (
            cfg.method.low_dim_size + 8
        )  # also add the action size
        follower_agent = agent_fn(cfg)

        cfg.method.robot_name = "bimanual"

        return LeaderFollowerAgent(leader_agent, follower_agent)

    elif agent_type == "independent":
        checkpoint_name_prefix = cfg.framework.checkpoint_name_prefix
        cfg.method.robot_name = "right"
        cfg.framework.checkpoint_name_prefix = (
            f"{checkpoint_name_prefix}_{method_name.lower()}_right"
        )
        right_agent = agent_fn(cfg)

        cfg.method.robot_name = "left"
        cfg.framework.checkpoint_name_prefix = (
            f"{checkpoint_name_prefix}_{method_name.lower()}_left"
        )
        left_agent = agent_fn(cfg)

        cfg.method.robot_name = "bimanual"

        return BimanualAgent(right_agent, left_agent)
    elif agent_type == "bimanual" or agent_type == "unimanual":
        return agent_fn(cfg)
    else:
        raise Exception("invalid agent type")


def agent_fn_by_name(method_name: str) -> Agent:
    if method_name == "ARM":
        from agents import arm

        raise NotImplementedError("ARM not yet supported for eval.py")
    elif method_name == "BC_LANG":
        from agents.baselines import bc_lang

        return bc_lang.launch_utils.create_agent
    elif method_name == "VIT_BC_LANG":
        from agents.baselines import vit_bc_lang

        return vit_bc_lang.launch_utils.create_agent
    elif method_name == "C2FARM_LINGUNET_BC":
        from agents import c2farm_lingunet_bc

        return c2farm_lingunet_bc.launch_utils.create_agent
    elif method_name.startswith("PERACT_BC"):
        from agents import peract_bc

        return peract_bc.launch_utils.create_agent
    elif method_name.startswith("BIMANUAL_PERACT"):
        from agents import bimanual_peract

        return bimanual_peract.launch_utils.create_agent
    elif method_name.startswith("RVT"):
        from agents import rvt

        return rvt.launch_utils.create_agent
    elif method_name.startswith("ACT_BC_LANG"):
        from agents import act_bc_lang

        return act_bc_lang.launch_utils.create_agent
    elif method_name == "PERACT_RL":
        raise NotImplementedError("PERACT_RL not yet supported for eval.py")

    else:
        raise ValueError("Method %s does not exists." % method_name)
