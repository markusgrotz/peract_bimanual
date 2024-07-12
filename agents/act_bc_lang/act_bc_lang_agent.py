import copy
import logging
from functools import lru_cache
import pickle
import os
from typing import List
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from yarr.agents.agent import Agent, Summary, ActResult, \
    ScalarSummary, HistogramSummary

from helpers import utils
from helpers.utils import stack_on_channel

from helpers.clip.core.clip import build_model, load_clip

NAME = 'ActBCLangAgent'


class ActBCLangAgent(Agent):

    def __init__(self,
                 actor_network: nn.Module,
                 camera_names: List[str],
                 lr: float = 0.01,
                 weight_decay: float = 1e-5,
                 grad_clip: float = 20.0,
                 episode_length: int = 400, train_demo_path=None, task_name=None):
        self._camera_names = camera_names
        self._actor = actor_network
        self._lr = lr
        self._weight_decay = weight_decay
        self._grad_clip = grad_clip
        self._episode_length = episode_length
        self.train_demo_path = train_demo_path
        self.task_name = task_name

    def build(self, training: bool, device: torch.device = None):
        if device is None:
            device = torch.device('cpu')
        self._actor = self._actor.to(device).train(training)
        self._actor_optimizer = self._actor.configure_optimizers()

        self._device = device

    def reset(self):
        super(ActBCLangAgent, self).reset()

        self._timestep = 0
        # .. input_dim = input_dim * 2 for bimanual
        self._all_time_actions = torch.zeros([self._episode_length,
                                              self._episode_length+self._actor.model.num_queries,
                                              self._actor.model.input_dim]).to(self._device)
        self._all_actions = None

    def _grad_step(self, loss, opt, model_params=None, clip=None):
        opt.zero_grad()
        loss.backward()
        if clip is not None and model_params is not None:
            nn.utils.clip_grad_value_(model_params, clip)
        opt.step()


  
    @lru_cache()
    def train_stats(self):

        right_joint_positions = []
        left_joint_positions = []

        right_gripper_positions = []
        left_gripper_positions = []

        episodes_dir = f"{self.train_demo_path}/{self.task_name}/all_variations/episodes/"

        for episode in os.listdir(episodes_dir):
            with open(os.path.join(episodes_dir, episode, "low_dim_obs.pkl"), "br") as f:
                d = pickle.load(f)

            for o in d:
                right_joint_positions.append(o.right.joint_positions)
                left_joint_positions.append(o.left.joint_positions)

                right_gripper_positions.append([o.right.gripper_joint_positions[0]])
                left_gripper_positions.append([o.left.gripper_joint_positions[0]])

        right_joint_positions = np.asarray(right_joint_positions, dtype=np.float32)
        left_joint_positions = np.asarray(left_joint_positions, dtype=np.float32)


        right_gripper_positions = np.asarray(right_gripper_positions, dtype=np.float32)
        left_gripper_positions = np.asarray(left_gripper_positions, dtype=np.float32)

        stats = {
            "right_joints_mean": right_joint_positions.mean(axis=0),
            "right_joints_std": right_joint_positions.std(axis=0),

            "left_joints_mean": left_joint_positions.mean(axis=0),
            "left_joints_std": left_joint_positions.std(axis=0),

            "right_gripper_mean": right_gripper_positions.mean(axis=0),
            "right_gripper_std": right_gripper_positions.std(axis=0),

            "left_gripper_mean":  left_gripper_positions.mean(axis=0),
            "left_gripper_std": left_gripper_positions.std(axis=0)
        }

        return {k: torch.from_numpy(v).to(self._device) for k,v in stats.items()}

    

    def normalize_z(self, data, mean, std):
        return (data - mean) / std

    def unnormalize_z(self, data, mean, std):
        return data * std + mean
    

    def preprocess_qpos(self, observation: dict):

        stats = self.train_stats()

        right_qrev = self.normalize_z(observation['right_joint_positions'][:,0], stats["right_joints_mean"], stats["right_joints_std"])
        right_qgripper = self.normalize_z(observation['right_gripper_joint_positions'][:,0], stats["right_gripper_mean"], stats["right_gripper_std"])
        left_qrev = self.normalize_z(observation['left_joint_positions'][:,0], stats["left_joints_mean"], stats["left_joints_std"] )
        left_qgripper = self.normalize_z(observation['left_gripper_joint_positions'][:,0], stats["left_gripper_mean"], stats["left_gripper_std"])
        qpos = torch.cat([right_qrev, right_qgripper[:,0].unsqueeze(-1), left_qrev, left_qgripper[:,0].unsqueeze(-1)], dim=-1)

        return qpos
    


    def preprocess_action(self, replay_sample: dict):

        stats = self.train_stats()

        right_qrev = self.normalize_z(replay_sample['right_prev_joint_positions'][:,0], stats["right_joints_mean"], stats["right_joints_std"])
        right_qgripper = self.normalize_z(replay_sample['right_prev_gripper_joint_positions'][:,0], stats["right_gripper_mean"], stats["right_gripper_std"])
        left_qrev = self.normalize_z(replay_sample['left_prev_joint_positions'][:,0], stats["left_joints_mean"], stats["left_joints_std"] )
        left_qgripper = self.normalize_z(replay_sample['left_prev_gripper_joint_positions'][:,0], stats["left_gripper_mean"], stats["left_gripper_std"])
        qpos = torch.cat([right_qrev, right_qgripper[:,0].unsqueeze(-1), left_qrev, left_qgripper[:,0].unsqueeze(-1)], dim=-1)

        right_action_rev = self.normalize_z(replay_sample['right_next_joint_positions'], stats["right_joints_mean"], stats["right_joints_std"])
        right_action_gripper = self.normalize_z(replay_sample['right_next_gripper_joint_positions'], stats["right_gripper_mean"], stats["right_gripper_std"])
        left_action_rev = self.normalize_z(replay_sample['left_next_joint_positions'], stats["left_joints_mean"], stats["left_joints_std"] )
        left_action_gripper = self.normalize_z(replay_sample['left_next_gripper_joint_positions'], stats["left_gripper_mean"], stats["left_gripper_std"])
        action_seq = torch.cat([right_action_rev, right_action_gripper[:,:,0].unsqueeze(-1), left_action_rev, left_action_gripper[:,:,0].unsqueeze(-1)], dim=-1)

        return qpos, action_seq

    def preprocess_images(self, replay_sample: dict):
        stacked_rgb = []
        stacked_point_cloud = []

        for camera in self._camera_names:
            rgb = replay_sample['%s_rgb' % camera]
            rgb = rgb if rgb.dim() == 4 else rgb[:,0]
            stacked_rgb.append(rgb)

            point_cloud = replay_sample['%s_point_cloud' % camera]
            point_cloud = point_cloud if point_cloud.dim() == 4 else point_cloud[:,0]
            stacked_point_cloud.append(point_cloud)

        stacked_rgb = torch.stack(stacked_rgb, dim=1)
        stacked_point_cloud = torch.stack(stacked_point_cloud, dim=1)

        return stacked_rgb, stacked_point_cloud

    def update(self, step: int, replay_sample: dict) -> dict:
        lang_goal_emb = replay_sample['lang_goal_emb'] # TODO use language
        robot_state = replay_sample['low_dim_state']

        # preprocess input
        qpos, action_seq = self.preprocess_action(replay_sample)
        stacked_rgb, stacked_point_cloud = self.preprocess_images(replay_sample)
        is_pad = replay_sample['is_pad'].bool()

        # forward pass
        loss_dict = self._actor(qpos, stacked_rgb, action_seq, is_pad)

        # gradient step
        loss = loss_dict['total_losses']
        loss.backward()
        self._actor_optimizer.step()
        self._actor_optimizer.zero_grad()

        self._summaries = {
            'loss': loss_dict['total_losses'],
            'l1': loss_dict['l1'],
            'right_l1': loss_dict['right_l1'],
            'left_l1': loss_dict['left_l1'],
            'kl': loss_dict['kl'],
        }

        return loss_dict

    def _normalize_quat(self, x):
        return x / x.square().sum(dim=1).sqrt().unsqueeze(-1)

    def _normalize_revolute_joints(self, x):
        # normalize joint angles
        # input ranges from -pi to pi
        # out ranges from 0 to 1
        return (x + np.pi) / (2 * np.pi)

    def _unnormalize_revolute_joints(self, x):
        # map input with range 0 to 1 to -pi to pi
        x = (x - 0.5) * 2.0 * np.pi
        x = torch.clamp(x, -np.pi, np.pi)
        return x

    def _normalize_gripper_joints(self, x):
        gripper_min = 0
        gripper_max = 0.04
        # normalize gripper joint angles between 0 and 1, the input ranges from 0 to 0.04
        return ((x - gripper_min) / (gripper_max - gripper_min))

    def _unnormalize_gripper_joints(self, x):
        gripper_min = 0
        gripper_max = 0.04
        
        x = x * (gripper_max - gripper_min) + gripper_min
        x = torch.clamp(x, gripper_min, gripper_max)
        return torch.unsqueeze(x, dim=0)




    def act(self, step: int, observation: dict,
            deterministic=False) -> ActResult:
        # lang_goal_tokens = observation.get('lang_goal_tokens', None).long()
        # with torch.no_grad():
        #     lang_goal_tokens = lang_goal_tokens.to(device=self._device)
        #     lang_goal_emb, _ = self._clip_rn50.encode_text_with_embeddings(lang_goal_tokens[0])
        #     lang_goal_emb = lang_goal_emb.to(device=self._device)

        action_horizon = self._actor.model.num_queries
        query_freq = 1



        stats = self.train_stats()

        if self._timestep % query_freq == 0:
            with torch.no_grad():
                # preprocess input
                qpos = self.preprocess_qpos(observation)
                stacked_rgb, stacked_point_cloud = self.preprocess_images(observation)

                # forward pass
                self._all_actions = self._actor(qpos, stacked_rgb, actions=None, is_pad=None)

        # temporal aggregation
        t = self._timestep

        self._all_time_actions[[t], t:t + action_horizon] = self._all_actions
        actions_for_curr_step = self._all_time_actions[:, t]
        actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
        actions_for_curr_step = actions_for_curr_step[actions_populated]
        k = 0.01
        exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
        exp_weights = exp_weights / exp_weights.sum()
        exp_weights = torch.from_numpy(exp_weights).to(self._device).unsqueeze(dim=1)
        raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
        raw_action = raw_action[0]


        right_a_rev = self.unnormalize_z(raw_action[0:7], stats["right_joints_mean"], stats["right_joints_std"])
        right_a_gripper = self.unnormalize_z(raw_action[7], stats["right_gripper_mean"], stats["right_gripper_std"])
        
        left_a_rev = self.unnormalize_z(raw_action[8:15], stats["left_joints_mean"], stats["left_joints_std"] )
        left_a_gripper = self.unnormalize_z(raw_action[15], stats["left_gripper_mean"], stats["left_gripper_std"])

        raw_action = torch.cat([right_a_rev, right_a_gripper, left_a_rev, left_a_gripper], dim=-1)

        self._timestep += 1

        return ActResult(raw_action.detach().cpu().numpy())

    def update_summaries(self) -> List[Summary]:
        summaries = []
        for n, v in self._summaries.items():
            summaries.append(ScalarSummary('%s/%s' % (NAME, n), v))

        # for tag, param in self._actor.named_parameters():
        #     summaries.append(
        #
        #     summaries.append(
        #         HistogramSummary('%s/weight/%s' % (NAME, tag), param.data))

        return summaries

    def act_summaries(self) -> List[Summary]:
        return []

    def load_weights(self, savedir: str):
        self._actor.load_state_dict(
            torch.load(os.path.join(savedir, 'bc_actor.pt'),
                       map_location=torch.device('cpu')))
        print('Loaded weights from %s' % savedir)

    def save_weights(self, savedir: str):
        torch.save(self._actor.state_dict(),
                   os.path.join(savedir, 'bc_actor.pt'))
