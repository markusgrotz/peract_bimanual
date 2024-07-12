import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms

from agents.act_bc_lang.detr.build import build_ACT_model_and_optimizer, build_CNNMLP_model_and_optimizer


class ACTPolicy(nn.Module):
    def __init__(self, args):
        super().__init__()
        model, optimizer = build_ACT_model_and_optimizer(args)
        self.model = model # CVAE decoder
        self.optimizer = optimizer
        self.kl_weight = args.kl_weight
        print(f'KL Weight {self.kl_weight}')

    def forward(self, qpos, image, actions=None, is_pad=None):
        env_state = None

        if actions is not None: # training time
            actions = actions[:, :self.model.num_queries]
            is_pad = is_pad[:, :self.model.num_queries]

            a_hat, is_pad_hat, (mu, logvar) = self.model(qpos, image, env_state, actions, is_pad)
            total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
            loss_dict = dict()

            right_actions_joints, right_a_hat_joints = actions[:, :, 0:8], a_hat[:, :, 0:8]
            right_actions_gripper, right_a_hat_gripper = actions[:, :, 7], a_hat[:, :, 7]
            left_actions_joints, left_a_hat_joints = actions[:, :, 8:16], a_hat[:, :, 8:16]
            left_actions_gripper, left_a_hat_gripper = actions[:, :, 15], a_hat[:, :, 15]

            # use L1 loss for joints
            right_l1_loss = F.l1_loss(right_a_hat_joints, right_actions_joints, reduction='none')
            right_l1 = (right_l1_loss * ~is_pad.unsqueeze(-1)).mean()

            left_l1_loss = F.l1_loss(left_a_hat_joints, left_actions_joints, reduction='none')
            left_l1 = (left_l1_loss * ~is_pad.unsqueeze(-1)).mean()


            l1 = right_l1 + left_l1


            right_gripper_l1_loss = F.l1_loss(right_a_hat_gripper, right_actions_gripper, reduction='none')
            right_gripper_l1_loss = (right_gripper_l1_loss * ~is_pad).mean()

            left_gripper_l1_loss = F.l1_loss(left_a_hat_gripper, left_actions_gripper, reduction='none')
            left_gripper_l1_loss = (left_gripper_l1_loss * ~is_pad).mean()

            gripper_l1 = right_gripper_l1_loss + left_gripper_l1_loss
            loss_dict['right_l1'] = right_l1
            loss_dict['left_l1'] = left_l1

            loss_dict['l1'] = l1
            loss_dict['gripper_l1'] = gripper_l1

            loss_dict['kl'] = total_kld[0]
            loss_dict['total_losses'] = loss_dict['l1'] + loss_dict['kl'] * self.kl_weight
            return loss_dict
        else: # inference time
            a_hat, _, (_, _) = self.model(qpos, image, env_state) # no action, sample from prior
            return a_hat

    def configure_optimizers(self):
        return self.optimizer


class CNNMLPPolicy(nn.Module):
    def __init__(self, args):
        super().__init__()
        model, optimizer = build_CNNMLP_model_and_optimizer(args)
        self.model = model # decoder
        self.optimizer = optimizer

    def forward(self, qpos, image, actions=None, is_pad=None):
        env_state = None # TODO

        if actions is not None: # training time
            actions = actions[:, 0]
            a_hat = self.model(qpos, image, env_state, actions)
            mse = F.mse_loss(actions, a_hat)
            loss_dict = dict()
            loss_dict['mse'] = mse
            loss_dict['loss'] = loss_dict['mse']
            return loss_dict
        else: # inference time
            a_hat = self.model(qpos, image, env_state) # no action, sample from prior
            return a_hat

    def configure_optimizers(self):
        return self.optimizer

def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld
