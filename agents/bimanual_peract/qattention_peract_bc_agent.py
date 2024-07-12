import copy
import logging
import os
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from pytorch3d import transforms as torch3d_tf
from yarr.agents.agent import (
    Agent,
    ActResult,
    ScalarSummary,
    HistogramSummary,
    ImageSummary,
    Summary,
)

from helpers import utils
from helpers.utils import visualise_voxel, stack_on_channel
from voxel.voxel_grid import VoxelGrid
from voxel.augmentation import apply_se3_augmentation
from einops import rearrange
from helpers.clip.core.clip import build_model, load_clip

import transformers
from helpers.optim.lamb import Lamb

from torch.nn.parallel import DistributedDataParallel as DDP

NAME = "QAttentionAgent"


class QFunction(nn.Module):
    def __init__(
        self,
        perceiver_encoder: nn.Module,
        voxelizer: VoxelGrid,
        bounds_offset: float,
        rotation_resolution: float,
        device,
        training,
    ):
        super(QFunction, self).__init__()
        self._rotation_resolution = rotation_resolution
        self._voxelizer = voxelizer
        self._bounds_offset = bounds_offset
        self._qnet = perceiver_encoder.to(device)

        # distributed training
        if training:
            self._qnet = DDP(self._qnet, device_ids=[device])

    def _argmax_3d(self, tensor_orig):
        b, c, d, h, w = tensor_orig.shape  # c will be one
        idxs = tensor_orig.view(b, c, -1).argmax(-1)
        indices = torch.cat([((idxs // h) // d), (idxs // h) % w, idxs % w], 1)
        return indices

    def choose_highest_action(self, q_trans, q_rot_grip, q_collision):
        coords = self._argmax_3d(q_trans)
        rot_and_grip_indicies = None
        ignore_collision = None
        if q_rot_grip is not None:
            q_rot = torch.stack(
                torch.split(
                    q_rot_grip[:, :-2], int(360 // self._rotation_resolution), dim=1
                ),
                dim=1,
            )
            rot_and_grip_indicies = torch.cat(
                [
                    q_rot[:, 0:1].argmax(-1),
                    q_rot[:, 1:2].argmax(-1),
                    q_rot[:, 2:3].argmax(-1),
                    q_rot_grip[:, -2:].argmax(-1, keepdim=True),
                ],
                -1,
            )
            ignore_collision = q_collision[:, -2:].argmax(-1, keepdim=True)
        return coords, rot_and_grip_indicies, ignore_collision

    def forward(
        self,
        rgb_pcd,
        proprio,
        pcd,
        lang_goal_emb,
        lang_token_embs,
        bounds=None,
        prev_bounds=None,
        prev_layer_voxel_grid=None,
    ):
        # rgb_pcd will be list of list (list of [rgb, pcd])
        b = rgb_pcd[0][0].shape[0]
        pcd_flat = torch.cat([p.permute(0, 2, 3, 1).reshape(b, -1, 3) for p in pcd], 1)

        # flatten RGBs and Pointclouds
        rgb = [rp[0] for rp in rgb_pcd]
        feat_size = rgb[0].shape[1]
        flat_imag_features = torch.cat(
            [p.permute(0, 2, 3, 1).reshape(b, -1, feat_size) for p in rgb], 1
        )

        # construct voxel grid
        voxel_grid = self._voxelizer.coords_to_bounding_voxel_grid(
            pcd_flat, coord_features=flat_imag_features, coord_bounds=bounds
        )

        # swap to channels fist
        voxel_grid = voxel_grid.permute(0, 4, 1, 2, 3).detach()

        # batch bounds if necessary
        if bounds.shape[0] != b:
            bounds = bounds.repeat(b, 1)

        # forward pass
        split_pred = self._qnet(
            voxel_grid,
            proprio,
            lang_goal_emb,
            lang_token_embs,
            prev_layer_voxel_grid,
            bounds,
            prev_bounds,
        )

        return split_pred, voxel_grid


class QAttentionPerActBCAgent(Agent):
    def __init__(
        self,
        layer: int,
        coordinate_bounds: list,
        perceiver_encoder: nn.Module,
        camera_names: list,
        batch_size: int,
        voxel_size: int,
        bounds_offset: float,
        voxel_feature_size: int,
        image_crop_size: int,
        num_rotation_classes: int,
        rotation_resolution: float,
        lr: float = 0.0001,
        lr_scheduler: bool = False,
        training_iterations: int = 100000,
        num_warmup_steps: int = 20000,
        trans_loss_weight: float = 1.0,
        rot_loss_weight: float = 1.0,
        grip_loss_weight: float = 1.0,
        collision_loss_weight: float = 1.0,
        include_low_dim_state: bool = False,
        image_resolution: list = None,
        lambda_weight_l2: float = 0.0,
        transform_augmentation: bool = True,
        transform_augmentation_xyz: list = [0.0, 0.0, 0.0],
        transform_augmentation_rpy: list = [0.0, 0.0, 180.0],
        transform_augmentation_rot_resolution: int = 5,
        optimizer_type: str = "adam",
        num_devices: int = 1,
    ):
        self._layer = layer
        self._coordinate_bounds = coordinate_bounds
        self._perceiver_encoder = perceiver_encoder
        self._voxel_feature_size = voxel_feature_size
        self._bounds_offset = bounds_offset
        self._image_crop_size = image_crop_size
        self._lr = lr
        self._lr_scheduler = lr_scheduler
        self._training_iterations = training_iterations
        self._num_warmup_steps = num_warmup_steps
        self._trans_loss_weight = trans_loss_weight
        self._rot_loss_weight = rot_loss_weight
        self._grip_loss_weight = grip_loss_weight
        self._collision_loss_weight = collision_loss_weight
        self._include_low_dim_state = include_low_dim_state
        self._image_resolution = image_resolution or [128, 128]
        self._voxel_size = voxel_size
        self._camera_names = camera_names
        self._num_cameras = len(camera_names)
        self._batch_size = batch_size
        self._lambda_weight_l2 = lambda_weight_l2
        self._transform_augmentation = transform_augmentation
        self._transform_augmentation_xyz = torch.from_numpy(
            np.array(transform_augmentation_xyz)
        )
        self._transform_augmentation_rpy = transform_augmentation_rpy
        self._transform_augmentation_rot_resolution = (
            transform_augmentation_rot_resolution
        )
        self._optimizer_type = optimizer_type
        self._num_devices = num_devices
        self._num_rotation_classes = num_rotation_classes
        self._rotation_resolution = rotation_resolution

        self._cross_entropy_loss = nn.CrossEntropyLoss(reduction="none")
        self._name = NAME + "_layer" + str(self._layer)

    def build(self, training: bool, device: torch.device = None):
        self._training = training

        if device is None:
            device = torch.device("cpu")

        self._device = device

        self._voxelizer = VoxelGrid(
            coord_bounds=self._coordinate_bounds,
            voxel_size=self._voxel_size,
            device=device,
            batch_size=self._batch_size if training else 1,
            feature_size=self._voxel_feature_size,
            max_num_coords=np.prod(self._image_resolution) * self._num_cameras,
        )

        self._q = (
            QFunction(
                self._perceiver_encoder,
                self._voxelizer,
                self._bounds_offset,
                self._rotation_resolution,
                device,
                training,
            )
            .to(device)
            .train(training)
        )

        grid_for_crop = (
            torch.arange(0, self._image_crop_size, device=device)
            .unsqueeze(0)
            .repeat(self._image_crop_size, 1)
            .unsqueeze(-1)
        )
        self._grid_for_crop = torch.cat(
            [grid_for_crop.transpose(1, 0), grid_for_crop], dim=2
        ).unsqueeze(0)

        self._coordinate_bounds = torch.tensor(
            self._coordinate_bounds, device=device
        ).unsqueeze(0)

        if self._training:
            # optimizer
            if self._optimizer_type == "lamb":
                self._optimizer = Lamb(
                    self._q.parameters(),
                    lr=self._lr,
                    weight_decay=self._lambda_weight_l2,
                    betas=(0.9, 0.999),
                    adam=False,
                )
            elif self._optimizer_type == "adam":
                self._optimizer = torch.optim.Adam(
                    self._q.parameters(),
                    lr=self._lr,
                    weight_decay=self._lambda_weight_l2,
                )
            else:
                raise Exception("Unknown optimizer type")

            # learning rate scheduler
            if self._lr_scheduler:
                self._scheduler = (
                    transformers.get_cosine_with_hard_restarts_schedule_with_warmup(
                        self._optimizer,
                        num_warmup_steps=self._num_warmup_steps,
                        num_training_steps=self._training_iterations,
                        num_cycles=self._training_iterations // 10000,
                    )
                )

            # one-hot zero tensors
            self._action_trans_one_hot_zeros = torch.zeros(
                (
                    self._batch_size,
                    1,
                    self._voxel_size,
                    self._voxel_size,
                    self._voxel_size,
                ),
                dtype=int,
                device=device,
            )
            self._action_rot_x_one_hot_zeros = torch.zeros(
                (self._batch_size, self._num_rotation_classes), dtype=int, device=device
            )
            self._action_rot_y_one_hot_zeros = torch.zeros(
                (self._batch_size, self._num_rotation_classes), dtype=int, device=device
            )
            self._action_rot_z_one_hot_zeros = torch.zeros(
                (self._batch_size, self._num_rotation_classes), dtype=int, device=device
            )
            self._action_grip_one_hot_zeros = torch.zeros(
                (self._batch_size, 2), dtype=int, device=device
            )
            self._action_ignore_collisions_one_hot_zeros = torch.zeros(
                (self._batch_size, 2), dtype=int, device=device
            )

            # print total params
            logging.info(
                "# Q Params: %d"
                % sum(
                    p.numel()
                    for name, p in self._q.named_parameters()
                    if p.requires_grad and "clip" not in name
                )
            )
        else:
            for param in self._q.parameters():
                param.requires_grad = False

            # load CLIP for encoding language goals during evaluation
            model, _ = load_clip("RN50", jit=False)
            self._clip_rn50 = build_model(model.state_dict())
            self._clip_rn50 = self._clip_rn50.float().to(device)
            self._clip_rn50.eval()
            del model

            self._voxelizer.to(device)
            self._q.to(device)

    def _extract_crop(self, pixel_action, observation):
        # Pixel action will now be (B, 2)
        # observation = stack_on_channel(observation)
        h = observation.shape[-1]
        top_left_corner = torch.clamp(
            pixel_action - self._image_crop_size // 2, 0, h - self._image_crop_size
        )
        grid = self._grid_for_crop + top_left_corner.unsqueeze(1)
        grid = ((grid / float(h)) * 2.0) - 1.0  # between -1 and 1
        # Used for cropping the images across a batch
        # swap fro y x, to x, y
        grid = torch.cat((grid[:, :, :, 1:2], grid[:, :, :, 0:1]), dim=-1)
        crop = F.grid_sample(observation, grid, mode="nearest", align_corners=True)
        return crop

    def _preprocess_inputs(self, replay_sample):
        obs = []
        pcds = []
        self._crop_summary = []
        for n in self._camera_names:
            rgb = replay_sample["%s_rgb" % n]
            pcd = replay_sample["%s_point_cloud" % n]

            obs.append([rgb, pcd])
            pcds.append(pcd)
        return obs, pcds

    def _act_preprocess_inputs(self, observation):
        obs, pcds = [], []
        for n in self._camera_names:
            rgb = observation["%s_rgb" % n]
            pcd = observation["%s_point_cloud" % n]

            obs.append([rgb, pcd])
            pcds.append(pcd)
        return obs, pcds

    def _get_value_from_voxel_index(self, q, voxel_idx):
        b, c, d, h, w = q.shape
        q_trans_flat = q.view(b, c, d * h * w)
        flat_indicies = (
            voxel_idx[:, 0] * d * h + voxel_idx[:, 1] * h + voxel_idx[:, 2]
        )[:, None].int()
        highest_idxs = flat_indicies.unsqueeze(-1).repeat(1, c, 1)
        chosen_voxel_values = q_trans_flat.gather(2, highest_idxs)[
            ..., 0
        ]  # (B, trans + rot + grip)
        return chosen_voxel_values

    def _get_value_from_rot_and_grip(self, rot_grip_q, rot_and_grip_idx):
        q_rot = torch.stack(
            torch.split(
                rot_grip_q[:, :-2], int(360 // self._rotation_resolution), dim=1
            ),
            dim=1,
        )  # B, 3, 72
        q_grip = rot_grip_q[:, -2:]
        rot_and_grip_values = torch.cat(
            [
                q_rot[:, 0].gather(1, rot_and_grip_idx[:, 0:1]),
                q_rot[:, 1].gather(1, rot_and_grip_idx[:, 1:2]),
                q_rot[:, 2].gather(1, rot_and_grip_idx[:, 2:3]),
                q_grip.gather(1, rot_and_grip_idx[:, 3:4]),
            ],
            -1,
        )
        return rot_and_grip_values

    def _celoss(self, pred, labels):
        return self._cross_entropy_loss(pred, labels.argmax(-1))

    def _softmax_q_trans(self, q):
        q_shape = q.shape
        return F.softmax(q.reshape(q_shape[0], -1), dim=1).reshape(q_shape)

    def _softmax_q_rot_grip(self, q_rot_grip):
        q_rot_x_flat = q_rot_grip[
            :, 0 * self._num_rotation_classes : 1 * self._num_rotation_classes
        ]
        q_rot_y_flat = q_rot_grip[
            :, 1 * self._num_rotation_classes : 2 * self._num_rotation_classes
        ]
        q_rot_z_flat = q_rot_grip[
            :, 2 * self._num_rotation_classes : 3 * self._num_rotation_classes
        ]
        q_grip_flat = q_rot_grip[:, 3 * self._num_rotation_classes :]

        q_rot_x_flat_softmax = F.softmax(q_rot_x_flat, dim=1)
        q_rot_y_flat_softmax = F.softmax(q_rot_y_flat, dim=1)
        q_rot_z_flat_softmax = F.softmax(q_rot_z_flat, dim=1)
        q_grip_flat_softmax = F.softmax(q_grip_flat, dim=1)

        return torch.cat(
            [
                q_rot_x_flat_softmax,
                q_rot_y_flat_softmax,
                q_rot_z_flat_softmax,
                q_grip_flat_softmax,
            ],
            dim=1,
        )

    def _softmax_ignore_collision(self, q_collision):
        q_collision_softmax = F.softmax(q_collision, dim=1)
        return q_collision_softmax

    def update(self, step: int, replay_sample: dict) -> dict:
        right_action_trans = replay_sample["right_trans_action_indicies"][
            :, self._layer * 3 : self._layer * 3 + 3
        ].int()
        right_action_rot_grip = replay_sample["right_rot_grip_action_indicies"].int()
        right_action_gripper_pose = replay_sample["right_gripper_pose"]
        right_action_ignore_collisions = replay_sample["right_ignore_collisions"].int()

        left_action_trans = replay_sample["left_trans_action_indicies"][
            :, self._layer * 3 : self._layer * 3 + 3
        ].int()
        left_action_rot_grip = replay_sample["left_rot_grip_action_indicies"].int()
        left_action_gripper_pose = replay_sample["left_gripper_pose"]
        left_action_ignore_collisions = replay_sample["left_ignore_collisions"].int()

        lang_goal_emb = replay_sample["lang_goal_emb"].float()
        lang_token_embs = replay_sample["lang_token_embs"].float()
        prev_layer_voxel_grid = replay_sample.get("prev_layer_voxel_grid", None)
        prev_layer_bounds = replay_sample.get("prev_layer_bounds", None)
        device = self._device

        bounds = self._coordinate_bounds.to(device)
        if self._layer > 0:
            right_cp = replay_sample[
                "right_attention_coordinate_layer_%d" % (self._layer - 1)
            ]

            left_cp = replay_sample[
                "left_attention_coordinate_layer_%d" % (self._layer - 1)
            ]

            right_bounds = torch.cat(
                [right_cp - self._bounds_offset, right_cp + self._bounds_offset], dim=1
            )
            left_bounds = torch.cat(
                [left_cp - self._bounds_offset, left_cp + self._bounds_offset], dim=1
            )

        else:
            right_bounds = bounds
            left_bounds = bounds

        right_proprio = None
        left_proprio = None
        if self._include_low_dim_state:
            right_proprio = replay_sample["right_low_dim_state"]
            left_proprio = replay_sample["left_low_dim_state"]

        # ..TODO::
        # Can we add the coordinates of both robots?
        #

        obs, pcd = self._preprocess_inputs(replay_sample)

        # batch size
        bs = pcd[0].shape[0]

        # We can move the point cloud w.r.t to the other robot's cooridinate system
        # similar to apply_se3_augmentation
        #

        # SE(3) augmentation of point clouds and actions
        if self._transform_augmentation:
            from voxel import augmentation

            (
                right_action_trans,
                right_action_rot_grip,
                left_action_trans,
                left_action_rot_grip,
                pcd,
            ) = augmentation.bimanual_apply_se3_augmentation(
                pcd,
                right_action_gripper_pose,
                right_action_trans,
                right_action_rot_grip,
                left_action_gripper_pose,
                left_action_trans,
                left_action_rot_grip,
                bounds,
                self._layer,
                self._transform_augmentation_xyz,
                self._transform_augmentation_rpy,
                self._transform_augmentation_rot_resolution,
                self._voxel_size,
                self._rotation_resolution,
                self._device,
            )
        else:
            right_action_trans = right_action_trans.int()
            left_action_trans = left_action_trans.int()

        proprio = torch.cat((right_proprio, left_proprio), dim=1)

        right_action = (
            right_action_trans,
            right_action_rot_grip,
            right_action_ignore_collisions,
        )
        left_action = (
            left_action_trans,
            left_action_rot_grip,
            left_action_ignore_collisions,
        )
        # forward pass
        q, voxel_grid = self._q(
            obs,
            proprio,
            pcd,
            lang_goal_emb,
            lang_token_embs,
            bounds,
            prev_layer_bounds,
            prev_layer_voxel_grid,
        )

        (
            right_q_trans,
            right_q_rot_grip,
            right_q_collision,
            left_q_trans,
            left_q_rot_grip,
            left_q_collision,
        ) = q

        # argmax to choose best action
        (
            right_coords,
            right_rot_and_grip_indicies,
            right_ignore_collision_indicies,
        ) = self._q.choose_highest_action(
            right_q_trans, right_q_rot_grip, right_q_collision
        )

        (
            left_coords,
            left_rot_and_grip_indicies,
            left_ignore_collision_indicies,
        ) = self._q.choose_highest_action(
            left_q_trans, left_q_rot_grip, left_q_collision
        )


        right_q_trans_loss, right_q_rot_loss, right_q_grip_loss, right_q_collision_loss = 0.0, 0.0, 0.0, 0.0
        left_q_trans_loss, left_q_rot_loss, left_q_grip_loss, left_q_collision_loss = 0.0, 0.0, 0.0, 0.0

        # translation one-hot
        right_action_trans_one_hot = self._action_trans_one_hot_zeros.clone().detach()
        left_action_trans_one_hot = self._action_trans_one_hot_zeros.clone().detach()
        for b in range(bs):
            right_gt_coord = right_action_trans[b, :].int()
            right_action_trans_one_hot[
                b, :, right_gt_coord[0], right_gt_coord[1], right_gt_coord[2]
            ] = 1
            left_gt_coord = left_action_trans[b, :].int()
            left_action_trans_one_hot[
                b, :, left_gt_coord[0], left_gt_coord[1], left_gt_coord[2]
            ] = 1

        # translation loss
        right_q_trans_flat = right_q_trans.view(bs, -1)
        right_action_trans_one_hot_flat = right_action_trans_one_hot.view(bs, -1)
        right_q_trans_loss = self._celoss(
            right_q_trans_flat, right_action_trans_one_hot_flat
        )
        left_q_trans_flat = left_q_trans.view(bs, -1)
        left_action_trans_one_hot_flat = left_action_trans_one_hot.view(bs, -1)
        left_q_trans_loss = self._celoss(
            left_q_trans_flat, left_action_trans_one_hot_flat
        )

        q_trans_loss = right_q_trans_loss + left_q_trans_loss

        with_rot_and_grip = (
            len(right_rot_and_grip_indicies) > 0 and len(left_rot_and_grip_indicies) > 0
        )
        if with_rot_and_grip:
            # rotation, gripper, and collision one-hots
            right_action_rot_x_one_hot = self._action_rot_x_one_hot_zeros.clone()
            right_action_rot_y_one_hot = self._action_rot_y_one_hot_zeros.clone()
            right_action_rot_z_one_hot = self._action_rot_z_one_hot_zeros.clone()
            right_action_grip_one_hot = self._action_grip_one_hot_zeros.clone()
            right_action_ignore_collisions_one_hot = (
                self._action_ignore_collisions_one_hot_zeros.clone()
            )

            left_action_rot_x_one_hot = self._action_rot_x_one_hot_zeros.clone()
            left_action_rot_y_one_hot = self._action_rot_y_one_hot_zeros.clone()
            left_action_rot_z_one_hot = self._action_rot_z_one_hot_zeros.clone()
            left_action_grip_one_hot = self._action_grip_one_hot_zeros.clone()
            left_action_ignore_collisions_one_hot = (
                self._action_ignore_collisions_one_hot_zeros.clone()
            )

            for b in range(bs):
                right_gt_rot_grip = right_action_rot_grip[b, :].int()
                right_action_rot_x_one_hot[b, right_gt_rot_grip[0]] = 1
                right_action_rot_y_one_hot[b, right_gt_rot_grip[1]] = 1
                right_action_rot_z_one_hot[b, right_gt_rot_grip[2]] = 1
                right_action_grip_one_hot[b, right_gt_rot_grip[3]] = 1

                right_gt_ignore_collisions = right_action_ignore_collisions[b, :].int()
                right_action_ignore_collisions_one_hot[
                    b, right_gt_ignore_collisions[0]
                ] = 1

                left_gt_rot_grip = left_action_rot_grip[b, :].int()
                left_action_rot_x_one_hot[b, left_gt_rot_grip[0]] = 1
                left_action_rot_y_one_hot[b, left_gt_rot_grip[1]] = 1
                left_action_rot_z_one_hot[b, left_gt_rot_grip[2]] = 1
                left_action_grip_one_hot[b, left_gt_rot_grip[3]] = 1

                left_gt_ignore_collisions = left_action_ignore_collisions[b, :].int()
                left_action_ignore_collisions_one_hot[
                    b, left_gt_ignore_collisions[0]
                ] = 1

            # flatten predictions
            right_q_rot_x_flat = right_q_rot_grip[
                :, 0 * self._num_rotation_classes : 1 * self._num_rotation_classes
            ]
            right_q_rot_y_flat = right_q_rot_grip[
                :, 1 * self._num_rotation_classes : 2 * self._num_rotation_classes
            ]
            right_q_rot_z_flat = right_q_rot_grip[
                :, 2 * self._num_rotation_classes : 3 * self._num_rotation_classes
            ]
            right_q_grip_flat = right_q_rot_grip[:, 3 * self._num_rotation_classes :]
            right_q_ignore_collisions_flat = right_q_collision

            left_q_rot_x_flat = left_q_rot_grip[
                :, 0 * self._num_rotation_classes : 1 * self._num_rotation_classes
            ]
            left_q_rot_y_flat = left_q_rot_grip[
                :, 1 * self._num_rotation_classes : 2 * self._num_rotation_classes
            ]
            left_q_rot_z_flat = left_q_rot_grip[
                :, 2 * self._num_rotation_classes : 3 * self._num_rotation_classes
            ]
            left_q_grip_flat = left_q_rot_grip[:, 3 * self._num_rotation_classes :]
            left_q_ignore_collisions_flat = left_q_collision


            # rotation loss
            right_q_rot_loss += self._celoss(right_q_rot_x_flat, right_action_rot_x_one_hot)
            right_q_rot_loss += self._celoss(right_q_rot_y_flat, right_action_rot_y_one_hot)
            right_q_rot_loss += self._celoss(right_q_rot_z_flat, right_action_rot_z_one_hot)

            left_q_rot_loss += self._celoss(left_q_rot_x_flat, left_action_rot_x_one_hot)
            left_q_rot_loss += self._celoss(left_q_rot_y_flat, left_action_rot_y_one_hot)
            left_q_rot_loss += self._celoss(left_q_rot_z_flat, left_action_rot_z_one_hot)

            # gripper loss
            right_q_grip_loss += self._celoss(right_q_grip_flat, right_action_grip_one_hot)
            left_q_grip_loss += self._celoss(left_q_grip_flat, left_action_grip_one_hot)

            # collision loss
            right_q_collision_loss += self._celoss(
                right_q_ignore_collisions_flat, right_action_ignore_collisions_one_hot
            )
            left_q_collision_loss += self._celoss(
                left_q_ignore_collisions_flat, left_action_ignore_collisions_one_hot
            )


        q_trans_loss = right_q_trans_loss + left_q_trans_loss
        q_rot_loss = right_q_rot_loss + left_q_rot_loss
        q_grip_loss = right_q_grip_loss + left_q_grip_loss
        q_collision_loss = right_q_collision_loss + left_q_collision_loss
        
        combined_losses = (
            (q_trans_loss * self._trans_loss_weight)
            + (q_rot_loss * self._rot_loss_weight)
            + (q_grip_loss * self._grip_loss_weight)
            + (q_collision_loss * self._collision_loss_weight)
        )
        total_loss = combined_losses.mean()

        self._optimizer.zero_grad()
        total_loss.backward()
        self._optimizer.step()

        self._summaries = {
            "losses/total_loss": total_loss,
            "losses/trans_loss": q_trans_loss.mean(),
            "losses/rot_loss": q_rot_loss.mean() if with_rot_and_grip else 0.0,
            "losses/grip_loss": q_grip_loss.mean() if with_rot_and_grip else 0.0,

            "losses/right/trans_loss": q_trans_loss.mean(),
            "losses/right/rot_loss": q_rot_loss.mean() if with_rot_and_grip else 0.0,
            "losses/right/grip_loss": q_grip_loss.mean() if with_rot_and_grip else 0.0,
            "losses/right/collision_loss": q_collision_loss.mean() if with_rot_and_grip else 0.0,

            "losses/left/trans_loss": q_trans_loss.mean(),
            "losses/left/rot_loss": q_rot_loss.mean() if with_rot_and_grip else 0.0,
            "losses/left/grip_loss": q_grip_loss.mean() if with_rot_and_grip else 0.0,
            "losses/left/collision_loss": q_collision_loss.mean() if with_rot_and_grip else 0.0,

            "losses/collision_loss": q_collision_loss.mean()
            if with_rot_and_grip
            else 0.0,
        }

        if self._lr_scheduler:
            self._scheduler.step()
            self._summaries["learning_rate"] = self._scheduler.get_last_lr()[0]

        self._vis_voxel_grid = voxel_grid[0]
        self._right_vis_translation_qvalue = self._softmax_q_trans(right_q_trans[0])
        self._right_vis_max_coordinate = right_coords[0]
        self._right_vis_gt_coordinate = right_action_trans[0]

        self._left_vis_translation_qvalue = self._softmax_q_trans(left_q_trans[0])
        self._left_vis_max_coordinate = left_coords[0]
        self._left_vis_gt_coordinate = left_action_trans[0]

        # Note: PerAct doesn't use multi-layer voxel grids like C2FARM
        # stack prev_layer_voxel_grid(s) from previous layers into a list
        if prev_layer_voxel_grid is None:
            prev_layer_voxel_grid = [voxel_grid]
        else:
            prev_layer_voxel_grid = prev_layer_voxel_grid + [voxel_grid]

        # stack prev_layer_bound(s) from previous layers into a list
        if prev_layer_bounds is None:
            prev_layer_bounds = [self._coordinate_bounds.repeat(bs, 1)]
        else:
            prev_layer_bounds = prev_layer_bounds + [bounds]

        return {
            "total_loss": total_loss,
            "prev_layer_voxel_grid": prev_layer_voxel_grid,
            "prev_layer_bounds": prev_layer_bounds,
        }

    def act(self, step: int, observation: dict, deterministic=False) -> ActResult:
        deterministic = True
        bounds = self._coordinate_bounds
        prev_layer_voxel_grid = observation.get("prev_layer_voxel_grid", None)
        prev_layer_bounds = observation.get("prev_layer_bounds", None)
        lang_goal_tokens = observation.get("lang_goal_tokens", None).long()

        # extract CLIP language embs
        with torch.no_grad():
            lang_goal_tokens = lang_goal_tokens.to(device=self._device)
            (
                lang_goal_emb,
                lang_token_embs,
            ) = self._clip_rn50.encode_text_with_embeddings(lang_goal_tokens[0])

        # voxelization resolution
        res = (bounds[:, 3:] - bounds[:, :3]) / self._voxel_size
        max_rot_index = int(360 // self._rotation_resolution)
        right_proprio = None
        left_proprio = None

        if self._include_low_dim_state:
            right_proprio = observation["right_low_dim_state"]
            left_proprio = observation["left_low_dim_state"]
            right_proprio = right_proprio[0].to(self._device)
            left_proprio = left_proprio[0].to(self._device)

        obs, pcd = self._act_preprocess_inputs(observation)

        # correct batch size and device
        obs = [[o[0][0].to(self._device), o[1][0].to(self._device)] for o in obs]

        pcd = [p[0].to(self._device) for p in pcd]
        lang_goal_emb = lang_goal_emb.to(self._device)
        lang_token_embs = lang_token_embs.to(self._device)
        bounds = torch.as_tensor(bounds, device=self._device)
        prev_layer_voxel_grid = (
            prev_layer_voxel_grid.to(self._device)
            if prev_layer_voxel_grid is not None
            else None
        )
        prev_layer_bounds = (
            prev_layer_bounds.to(self._device)
            if prev_layer_bounds is not None
            else None
        )

        proprio = torch.cat((right_proprio, left_proprio), dim=1)

        # inference
        (
            right_q_trans,
            right_q_rot_grip,
            right_q_ignore_collisions,
            left_q_trans,
            left_q_rot_grip,
            left_q_ignore_collisions,
        ), vox_grid = self._q(
            obs,
            proprio,
            pcd,
            lang_goal_emb,
            lang_token_embs,
            bounds,
            prev_layer_bounds,
            prev_layer_voxel_grid,
        )

        # softmax Q predictions
        right_q_trans = self._softmax_q_trans(right_q_trans)
        left_q_trans = self._softmax_q_trans(left_q_trans)

        if right_q_rot_grip is not None:
            right_q_rot_grip = self._softmax_q_rot_grip(right_q_rot_grip)

        if left_q_rot_grip is not None:
            left_q_rot_grip = self._softmax_q_rot_grip(left_q_rot_grip)

        if right_q_ignore_collisions is not None:
            right_q_ignore_collisions = self._softmax_ignore_collision(
                right_q_ignore_collisions
            )

        if left_q_ignore_collisions is not None:
            left_q_ignore_collisions = self._softmax_ignore_collision(
                left_q_ignore_collisions
            )

        # argmax Q predictions
        (
            right_coords,
            right_rot_and_grip_indicies,
            right_ignore_collisions,
        ) = self._q.choose_highest_action(
            right_q_trans, right_q_rot_grip, right_q_ignore_collisions
        )
        (
            left_coords,
            left_rot_and_grip_indicies,
            left_ignore_collisions,
        ) = self._q.choose_highest_action(
            left_q_trans, left_q_rot_grip, left_q_ignore_collisions
        )

        if right_q_rot_grip is not None:
            right_rot_grip_action = right_rot_and_grip_indicies
        if right_q_ignore_collisions is not None:
            right_ignore_collisions_action = right_ignore_collisions.int()

        if left_q_rot_grip is not None:
            left_rot_grip_action = left_rot_and_grip_indicies
        if left_q_ignore_collisions is not None:
            left_ignore_collisions_action = left_ignore_collisions.int()

        right_coords = right_coords.int()
        left_coords = left_coords.int()

        right_attention_coordinate = bounds[:, :3] + res * right_coords + res / 2
        left_attention_coordinate = bounds[:, :3] + res * left_coords + res / 2

        # stack prev_layer_voxel_grid(s) into a list
        # NOTE: PerAct doesn't used multi-layer voxel grids like C2FARM
        if prev_layer_voxel_grid is None:
            prev_layer_voxel_grid = [vox_grid]
        else:
            prev_layer_voxel_grid = prev_layer_voxel_grid + [vox_grid]

        if prev_layer_bounds is None:
            prev_layer_bounds = [bounds]
        else:
            prev_layer_bounds = prev_layer_bounds + [bounds]

        observation_elements = {
            "right_attention_coordinate": right_attention_coordinate,
            "left_attention_coordinate": left_attention_coordinate,
            "prev_layer_voxel_grid": prev_layer_voxel_grid,
            "prev_layer_bounds": prev_layer_bounds,
        }
        info = {
            "voxel_grid_depth%d" % self._layer: vox_grid,
            "right_q_depth%d" % self._layer: right_q_trans,
            "right_voxel_idx_depth%d" % self._layer: right_coords,
            "left_q_depth%d" % self._layer: left_q_trans,
            "left_voxel_idx_depth%d" % self._layer: left_coords,
        }
        self._act_voxel_grid = vox_grid[0]
        self._right_act_max_coordinate = right_coords[0]
        self._right_act_qvalues = right_q_trans[0].detach()
        self._left_act_max_coordinate = left_coords[0]
        self._left_act_qvalues = left_q_trans[0].detach()

        action = (
            right_coords,
            right_rot_grip_action,
            right_ignore_collisions,
            left_coords,
            left_rot_grip_action,
            left_ignore_collisions,
        )

        return ActResult(action, observation_elements=observation_elements, info=info)

    def update_summaries(self) -> List[Summary]:
        voxel_grid = self._vis_voxel_grid.detach().cpu().numpy()
        summaries = []
        summaries.append(
            ImageSummary(
                "%s/right_update_qattention" % self._name,
                transforms.ToTensor()(
                    visualise_voxel(
                        voxel_grid,
                        self._right_vis_translation_qvalue.detach().cpu().numpy(),
                        self._right_vis_max_coordinate.detach().cpu().numpy(),
                        self._right_vis_gt_coordinate.detach().cpu().numpy(),
                    )
                ),
            )
        )
        summaries.append(
            ImageSummary(
                "%s/left_update_qattention" % self._name,
                transforms.ToTensor()(
                    visualise_voxel(
                        voxel_grid,
                        self._left_vis_translation_qvalue.detach().cpu().numpy(),
                        self._left_vis_max_coordinate.detach().cpu().numpy(),
                        self._left_vis_gt_coordinate.detach().cpu().numpy(),
                    )
                ),
            )
        )
        for n, v in self._summaries.items():
            summaries.append(ScalarSummary("%s/%s" % (self._name, n), v))

        for name, crop in self._crop_summary:
            crops = (torch.cat(torch.split(crop, 3, dim=1), dim=3) + 1.0) / 2.0
            summaries.extend([ImageSummary("%s/crops/%s" % (self._name, name), crops)])

        for tag, param in self._q.named_parameters():
            # assert not torch.isnan(param.grad.abs() <= 1.0).all()
            summaries.append(
                HistogramSummary("%s/gradient/%s" % (self._name, tag), param.grad)
            )
            summaries.append(
                HistogramSummary("%s/weight/%s" % (self._name, tag), param.data)
            )

        return summaries

    def act_summaries(self) -> List[Summary]:
        voxel_grid = self._act_voxel_grid.cpu().numpy()
        right_q_attention = self._right_act_qvalues.cpu().numpy()
        right_highlight_coordinate = self._right_act_max_coordinate.cpu().numpy()
        right_visualization = visualise_voxel(
            voxel_grid, right_q_attention, right_highlight_coordinate
        )

        left_q_attention = self._left_act_qvalues.cpu().numpy()
        left_highlight_coordinate = self._left_act_max_coordinate.cpu().numpy()
        left_visualization = visualise_voxel(
            voxel_grid, left_q_attention, left_highlight_coordinate
        )

        return [
            ImageSummary(
                f"{self._name}/right_act_Qattention",
                transforms.ToTensor()(right_visualization),
            ),
            ImageSummary(
                f"{self._name}/left_act_Qattention",
                transforms.ToTensor()(left_visualization),
            ),
        ]

    def load_weights(self, savedir: str):
        device = (
            self._device
            if not self._training
            else torch.device("cuda:%d" % self._device)
        )
        weight_file = os.path.join(savedir, "%s.pt" % self._name)
        state_dict = torch.load(weight_file, map_location=device)

        # load only keys that are in the current model
        merged_state_dict = self._q.state_dict()
        for k, v in state_dict.items():
            if not self._training:
                k = k.replace("_qnet.module", "_qnet")
            if k in merged_state_dict:
                merged_state_dict[k] = v
            else:
                if "_voxelizer" not in k:
                    logging.warning("key %s not found in checkpoint" % k)
        if not self._training:
            # reshape voxelizer weights
            b = merged_state_dict["_voxelizer._ones_max_coords"].shape[0]
            merged_state_dict["_voxelizer._ones_max_coords"] = merged_state_dict[
                "_voxelizer._ones_max_coords"
            ][0:1]
            flat_shape = merged_state_dict["_voxelizer._flat_output"].shape[0]
            merged_state_dict["_voxelizer._flat_output"] = merged_state_dict[
                "_voxelizer._flat_output"
            ][0 : flat_shape // b]
            merged_state_dict["_voxelizer._tiled_batch_indices"] = merged_state_dict[
                "_voxelizer._tiled_batch_indices"
            ][0:1]
            merged_state_dict["_voxelizer._index_grid"] = merged_state_dict[
                "_voxelizer._index_grid"
            ][0:1]
        self._q.load_state_dict(merged_state_dict)
        print("loaded weights from %s" % weight_file)

    def save_weights(self, savedir: str):
        torch.save(self._q.state_dict(), os.path.join(savedir, "%s.pt" % self._name))
