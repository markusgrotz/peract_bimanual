from typing import List

import torch
from yarr.agents.agent import Agent, ActResult, Summary

import numpy as np

from helpers import utils
from agents.bimanual_peract.qattention_peract_bc_agent import QAttentionPerActBCAgent

NAME = "QAttentionStackAgent"


class QAttentionStackAgent(Agent):
    def __init__(
        self,
        qattention_agents: List[QAttentionPerActBCAgent],
        rotation_resolution: float,
        camera_names: List[str],
        rotation_prediction_depth: int = 0,
    ):
        super(QAttentionStackAgent, self).__init__()
        self._qattention_agents = qattention_agents
        self._rotation_resolution = rotation_resolution
        self._camera_names = camera_names
        self._rotation_prediction_depth = rotation_prediction_depth

    def build(self, training: bool, device=None) -> None:
        self._device = device
        if self._device is None:
            self._device = torch.device("cpu")
        for qa in self._qattention_agents:
            qa.build(training, device)

    def update(self, step: int, replay_sample: dict) -> dict:
        priorities = 0
        total_losses = 0.0
        for qa in self._qattention_agents:
            update_dict = qa.update(step, replay_sample)
            replay_sample.update(update_dict)
            total_losses += update_dict["total_loss"]
        return {
            "total_losses": total_losses,
        }

    def act(self, step: int, observation: dict, deterministic=False) -> ActResult:
        observation_elements = {}
        (
            right_translation_results,
            right_rot_grip_results,
            right_ignore_collisions_results,
        ) = ([], [], [])
        (
            left_translation_results,
            left_rot_grip_results,
            left_ignore_collisions_results,
        ) = ([], [], [])

        infos = {}
        for depth, qagent in enumerate(self._qattention_agents):
            act_results = qagent.act(step, observation, deterministic)
            right_attention_coordinate = (
                act_results.observation_elements["right_attention_coordinate"]
                .cpu()
                .numpy()
            )
            left_attention_coordinate = (
                act_results.observation_elements["left_attention_coordinate"]
                .cpu()
                .numpy()
            )
            observation_elements[
                "right_attention_coordinate_layer_%d" % depth
            ] = right_attention_coordinate[0]
            observation_elements[
                "left_attention_coordinate_layer_%d" % depth
            ] = left_attention_coordinate[0]

            (
                right_translation_idxs,
                right_rot_grip_idxs,
                right_ignore_collisions_idxs,
                left_translation_idxs,
                left_rot_grip_idxs,
                left_ignore_collisions_idxs,
            ) = act_results.action

            right_translation_results.append(right_translation_idxs)
            if right_rot_grip_idxs is not None:
                right_rot_grip_results.append(right_rot_grip_idxs)
            if right_ignore_collisions_idxs is not None:
                right_ignore_collisions_results.append(right_ignore_collisions_idxs)

            left_translation_results.append(left_translation_idxs)
            if left_rot_grip_idxs is not None:
                left_rot_grip_results.append(left_rot_grip_idxs)
            if left_ignore_collisions_idxs is not None:
                left_ignore_collisions_results.append(left_ignore_collisions_idxs)

            observation[
                "right_attention_coordinate"
            ] = act_results.observation_elements["right_attention_coordinate"]
            observation["left_attention_coordinate"] = act_results.observation_elements[
                "left_attention_coordinate"
            ]

            observation["prev_layer_voxel_grid"] = act_results.observation_elements[
                "prev_layer_voxel_grid"
            ]
            observation["prev_layer_bounds"] = act_results.observation_elements[
                "prev_layer_bounds"
            ]

            for n in self._camera_names:
                extrinsics = observation["%s_camera_extrinsics" % n][0, 0].cpu().numpy()
                intrinsics = observation["%s_camera_intrinsics" % n][0, 0].cpu().numpy()
                px, py = utils.point_to_pixel_index(
                    right_attention_coordinate[0], extrinsics, intrinsics
                )
                pc_t = torch.tensor(
                    [[[py, px]]], dtype=torch.float32, device=self._device
                )
                observation[f"right_{n}_pixel_coord"] = pc_t
                observation_elements[f"right_{n}_pixel_coord"] = [py, px]

                px, py = utils.point_to_pixel_index(
                    left_attention_coordinate[0], extrinsics, intrinsics
                )
                pc_t = torch.tensor(
                    [[[py, px]]], dtype=torch.float32, device=self._device
                )
                observation[f"left_{n}_pixel_coord"] = pc_t
                observation_elements[f"left_{n}_pixel_coord"] = [py, px]
            infos.update(act_results.info)

        right_rgai = torch.cat(right_rot_grip_results, 1)[0].cpu().numpy()
        # ..todo:: utils.correct_rotation_instability does nothing so we can ignore it
        # right_rgai = utils.correct_rotation_instability(right_rgai, self._rotation_resolution)
        right_ignore_collisions = (
            torch.cat(right_ignore_collisions_results, 1)[0].cpu().numpy()
        )
        right_trans_action_indicies = (
            torch.cat(right_translation_results, 1)[0].cpu().numpy()
        )

        observation_elements[
            "right_trans_action_indicies"
        ] = right_trans_action_indicies[:3]
        observation_elements["right_rot_grip_action_indicies"] = right_rgai[:4]

        left_rgai = torch.cat(left_rot_grip_results, 1)[0].cpu().numpy()
        left_ignore_collisions = (
            torch.cat(left_ignore_collisions_results, 1)[0].cpu().numpy()
        )
        left_trans_action_indicies = (
            torch.cat(left_translation_results, 1)[0].cpu().numpy()
        )

        observation_elements["left_trans_action_indicies"] = left_trans_action_indicies[
            3:
        ]
        observation_elements["left_rot_grip_action_indicies"] = left_rgai[4:]

        continuous_action = np.concatenate(
            [
                right_attention_coordinate[0],
                utils.discrete_euler_to_quaternion(
                    right_rgai[-4:-1], self._rotation_resolution
                ),
                right_rgai[-1:],
                right_ignore_collisions,
                left_attention_coordinate[0],
                utils.discrete_euler_to_quaternion(
                    left_rgai[-4:-1], self._rotation_resolution
                ),
                left_rgai[-1:],
                left_ignore_collisions,
            ]
        )
        return ActResult(
            continuous_action, observation_elements=observation_elements, info=infos
        )

    def update_summaries(self) -> List[Summary]:
        summaries = []
        for qa in self._qattention_agents:
            summaries.extend(qa.update_summaries())
        return summaries

    def act_summaries(self) -> List[Summary]:
        s = []
        for qa in self._qattention_agents:
            s.extend(qa.act_summaries())
        return s

    def load_weights(self, savedir: str):
        for qa in self._qattention_agents:
            qa.load_weights(savedir)

    def save_weights(self, savedir: str):
        for qa in self._qattention_agents:
            qa.save_weights(savedir)
