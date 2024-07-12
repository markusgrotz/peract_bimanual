# Perceiver IO implementation adpated for manipulation
# Source: https://github.com/lucidrains/perceiver-pytorch
# License: https://github.com/lucidrains/perceiver-pytorch/blob/main/LICENSE

import torch
from torch import nn

from einops import rearrange
from einops import repeat

from perceiver_pytorch.perceiver_pytorch import cache_fn
from perceiver_pytorch.perceiver_pytorch import PreNorm, FeedForward, Attention

from helpers.network_utils import (
    DenseBlock,
    SpatialSoftmax3D,
    Conv3DBlock,
    Conv3DUpsampleBlock,
)


# PerceiverIO adapted for 6-DoF manipulation
class PerceiverVoxelLangEncoder(nn.Module):
    def __init__(
        self,
        depth,  # number of self-attention layers
        iterations,  # number cross-attention iterations (PerceiverIO uses just 1)
        voxel_size,  # N voxels per side (size: N*N*N)
        initial_dim,  # 10 dimensions - dimension of the input sequence to be encoded
        low_dim_size,  # 4 dimensions - proprioception: {gripper_open, left_finger, right_finger, timestep}
        layer=0,
        num_rotation_classes=72,  # 5 degree increments (5*72=360) for each of the 3-axis
        num_grip_classes=2,  # open or not open
        num_collision_classes=2,  # collisions allowed or not allowed
        input_axis=3,  # 3D tensors have 3 axes
        num_latents=512,  # number of latent vectors
        im_channels=64,  # intermediate channel size
        latent_dim=512,  # dimensions of latent vectors
        cross_heads=1,  # number of cross-attention heads
        latent_heads=8,  # number of latent heads
        cross_dim_head=64,
        latent_dim_head=64,
        activation="relu",
        weight_tie_layers=False,
        pos_encoding_with_lang=True,
        input_dropout=0.1,
        attn_dropout=0.1,
        decoder_dropout=0.0,
        lang_fusion_type="seq",
        voxel_patch_size=9,
        voxel_patch_stride=8,
        no_skip_connection=False,
        no_perceiver=False,
        no_language=False,
        final_dim=64,
    ):
        super().__init__()
        self.depth = depth
        self.layer = layer
        self.init_dim = int(initial_dim)
        self.iterations = iterations
        self.input_axis = input_axis
        self.voxel_size = voxel_size
        self.low_dim_size = low_dim_size
        self.im_channels = im_channels
        self.pos_encoding_with_lang = pos_encoding_with_lang
        self.lang_fusion_type = lang_fusion_type
        self.voxel_patch_size = voxel_patch_size
        self.voxel_patch_stride = voxel_patch_stride
        self.num_rotation_classes = num_rotation_classes
        self.num_grip_classes = num_grip_classes
        self.num_collision_classes = num_collision_classes
        self.final_dim = final_dim
        self.input_dropout = input_dropout
        self.attn_dropout = attn_dropout
        self.decoder_dropout = decoder_dropout
        self.no_skip_connection = no_skip_connection
        self.no_perceiver = no_perceiver
        self.no_language = no_language

        # patchified input dimensions
        spatial_size = voxel_size // self.voxel_patch_stride  # 100/5 = 20

        # 64 voxel features + 64 proprio features (+ 64 lang goal features if concattenated)
        self.input_dim_before_seq = (
            self.im_channels * 3
            if self.lang_fusion_type == "concat"
            else self.im_channels * 2
        )

        # CLIP language feature dimensions
        lang_feat_dim, lang_emb_dim, lang_max_seq_len = 1024, 512, 77

        # learnable positional encoding
        if self.pos_encoding_with_lang:
            self.pos_encoding = nn.Parameter(
                torch.randn(
                    1, lang_max_seq_len + spatial_size**3, self.input_dim_before_seq
                )
            )
        else:
            # assert self.lang_fusion_type == 'concat', 'Only concat is supported for pos encoding without lang.'
            self.pos_encoding = nn.Parameter(
                torch.randn(
                    1,
                    spatial_size,
                    spatial_size,
                    spatial_size,
                    self.input_dim_before_seq,
                )
            )

        # voxel input preprocessing 1x1 conv encoder
        self.input_preprocess = Conv3DBlock(
            self.init_dim,
            self.im_channels,
            kernel_sizes=1,
            strides=1,
            norm=None,
            activation=activation,
        )

        # patchify conv
        self.patchify = Conv3DBlock(
            self.input_preprocess.out_channels,
            self.im_channels,
            kernel_sizes=self.voxel_patch_size,
            strides=self.voxel_patch_stride,
            norm=None,
            activation=activation,
        )

        # language preprocess
        if self.lang_fusion_type == "concat":
            self.lang_preprocess = nn.Linear(lang_feat_dim, self.im_channels)
        elif self.lang_fusion_type == "seq":
            self.lang_preprocess = nn.Linear(lang_emb_dim, self.im_channels * 2)

        # proprioception
        if self.low_dim_size > 0:
            self.proprio_preprocess = DenseBlock(
                self.low_dim_size,
                self.im_channels,
                norm=None,
                activation=activation,
            )

        # pooling functions
        self.local_maxp = nn.MaxPool3d(3, 2, padding=1)
        self.global_maxp = nn.AdaptiveMaxPool3d(1)

        # 1st 3D softmax
        self.ss0 = SpatialSoftmax3D(
            self.voxel_size, self.voxel_size, self.voxel_size, self.im_channels
        )
        flat_size = self.im_channels * 4

        # latent vectors (that are randomly initialized)
        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))

        # encoder cross attention
        self.cross_attend_blocks = nn.ModuleList(
            [
                PreNorm(
                    latent_dim,
                    Attention(
                        latent_dim,
                        self.input_dim_before_seq,
                        heads=cross_heads,
                        dim_head=cross_dim_head,
                        dropout=input_dropout,
                    ),
                    context_dim=self.input_dim_before_seq,
                ),
                PreNorm(latent_dim, FeedForward(latent_dim)),
                PreNorm(latent_dim, FeedForward(latent_dim)),
            ]
        )

        get_latent_attn = lambda: PreNorm(
            latent_dim,
            Attention(
                latent_dim,
                heads=latent_heads,
                dim_head=latent_dim_head,
                dropout=attn_dropout,
            ),
        )
        get_latent_ff = lambda: PreNorm(latent_dim, FeedForward(latent_dim))
        get_latent_attn, get_latent_ff = map(cache_fn, (get_latent_attn, get_latent_ff))

        # self attention layers
        self.layers = nn.ModuleList([])
        cache_args = {"_cache": weight_tie_layers}

        for i in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [get_latent_attn(**cache_args), get_latent_ff(**cache_args), 
                     get_latent_attn(**cache_args), get_latent_ff(**cache_args)]
                )
            )


        self.combined_latent_attn = get_latent_attn(**cache_args)
        self.combined_latent_ff = get_latent_ff(**cache_args)


        # decoder cross attention
        self.decoder_cross_attn_right = PreNorm(
            self.input_dim_before_seq,
            Attention(
                self.input_dim_before_seq,
                latent_dim,
                heads=cross_heads,
                dim_head=cross_dim_head,
                dropout=decoder_dropout,
            ),
            context_dim=latent_dim,
        )

        self.decoder_cross_attn_left = PreNorm(
            self.input_dim_before_seq,
            Attention(
                self.input_dim_before_seq,
                latent_dim,
                heads=cross_heads,
                dim_head=cross_dim_head,
                dropout=decoder_dropout,
            ),
            context_dim=latent_dim,
        )

        # upsample conv
        self.up0 = Conv3DUpsampleBlock(
            self.input_dim_before_seq,
            self.final_dim,
            kernel_sizes=self.voxel_patch_size,
            strides=self.voxel_patch_stride,
            norm=None,
            activation=activation,
        )

        # 2nd 3D softmax
        self.ss1 = SpatialSoftmax3D(
            spatial_size, spatial_size, spatial_size, self.input_dim_before_seq
        )

        flat_size += self.input_dim_before_seq * 4

        # final 3D softmax
        self.final = Conv3DBlock(
            self.im_channels
            if (self.no_perceiver or self.no_skip_connection)
            else self.im_channels * 2,
            self.im_channels,
            kernel_sizes=3,
            strides=1,
            norm=None,
            activation=activation,
        )

        self.right_trans_decoder = Conv3DBlock(
            self.final_dim,
            1,
            kernel_sizes=3,
            strides=1,
            norm=None,
            activation=None,
        )

        self.left_trans_decoder = Conv3DBlock(
            self.final_dim,
            1,
            kernel_sizes=3,
            strides=1,
            norm=None,
            activation=None,
        )

        # rotation, gripper, and collision MLP layers
        if self.num_rotation_classes > 0:
            self.ss_final = SpatialSoftmax3D(
                self.voxel_size, self.voxel_size, self.voxel_size, self.im_channels
            )

            flat_size += self.im_channels * 4

            self.right_dense0 = DenseBlock(flat_size, 256, None, activation)
            self.right_dense1 = DenseBlock(256, self.final_dim, None, activation)

            self.left_dense0 = DenseBlock(flat_size, 256, None, activation)
            self.left_dense1 = DenseBlock(256, self.final_dim, None, activation)

            self.right_rot_grip_collision_ff = DenseBlock(
                self.final_dim,
                self.num_rotation_classes * 3
                + self.num_grip_classes
                + self.num_collision_classes,
                None,
                None,
            )

            self.left_rot_grip_collision_ff = DenseBlock(
                self.final_dim,
                self.num_rotation_classes * 3
                + self.num_grip_classes
                + self.num_collision_classes,
                None,
                None,
            )

    def encode_text(self, x):
        with torch.no_grad():
            text_feat, text_emb = self._clip_rn50.encode_text_with_embeddings(x)

        text_feat = text_feat.detach()
        text_emb = text_emb.detach()
        text_mask = torch.where(x == 0, x, 1)  # [1, max_token_len]
        return text_feat, text_emb

    def forward(
        self,
        ins,
        proprio,
        lang_goal_emb,
        lang_token_embs,
        prev_layer_voxel_grid,
        bounds,
        prev_layer_bounds,
        mask=None,
    ):
        # preprocess input
        d0 = self.input_preprocess(ins)  # [B,10,100,100,100] -> [B,64,100,100,100]

        # aggregated features from 1st softmax and maxpool for MLP decoders
        feats = [self.ss0(d0.contiguous()), self.global_maxp(d0).view(ins.shape[0], -1)]

        # patchify input (5x5x5 patches)
        ins = self.patchify(d0)  # [B,64,100,100,100] -> [B,64,20,20,20]

        b, c, d, h, w, device = *ins.shape, ins.device
        axis = [d, h, w]
        assert (
            len(axis) == self.input_axis
        ), "input must have the same number of axis as input_axis"

        # concat proprio
        if self.low_dim_size > 0:
            p = self.proprio_preprocess(proprio)  # [B,4] -> [B,64]
            p = p.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, d, h, w)
            ins = torch.cat([ins, p], dim=1)  # [B,128,20,20,20]

        # language ablation
        if self.no_language:
            lang_goal_emb = torch.zeros_like(lang_goal_emb)
            lang_token_embs = torch.zeros_like(lang_token_embs)

        # option 1: tile and concat lang goal to input
        if self.lang_fusion_type == "concat":
            lang_emb = lang_goal_emb
            lang_emb = lang_emb.to(dtype=ins.dtype)
            l = self.lang_preprocess(lang_emb)
            l = l.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, d, h, w)
            ins = torch.cat([ins, l], dim=1)

        # channel last
        ins = rearrange(ins, "b d ... -> b ... d")  # [B,20,20,20,128]

        # add pos encoding to grid
        if not self.pos_encoding_with_lang:
            ins = ins + self.pos_encoding

        ######################## NOTE #############################
        # NOTE: If you add positional encodings ^here the lang embs
        # won't have positional encodings. I accidently forgot
        # to turn this off for all the experiments in the paper.
        # So I guess those models were using language embs
        # as a bag of words :( But it doesn't matter much for
        # RLBench tasks since we don't test for novel instructions
        # at test time anyway. The recommend way is to add
        # positional encodings to the final input sequence
        # fed into the Perceiver Transformer, as done below
        # (and also in the Colab tutorial).
        ###########################################################

        # concat to channels of and flatten axis
        queries_orig_shape = ins.shape

        # rearrange input to be channel last
        ins = rearrange(ins, "b ... d -> b (...) d")  # [B,8000,128]
        ins_wo_prev_layers = ins

        # option 2: add lang token embs as a sequence
        if self.lang_fusion_type == "seq":
            l = self.lang_preprocess(lang_token_embs)  # [B,77,1024] -> [B,77,128]
            ins = torch.cat((l, ins), dim=1)  # [B,8077,128]

        # add pos encoding to language + flattened grid (the recommended way)
        if self.pos_encoding_with_lang:
            ins = ins + self.pos_encoding

        # batchify latents
        x = repeat(self.latents, "n d -> b n d", b=b)

        cross_attn, cross_ff_right, cross_ff_left = self.cross_attend_blocks

        for it in range(self.iterations):

            # encoder cross attention
            x = cross_attn(x, context=ins, mask=mask) + x

            # x.size() = [1, num_latents, latent_dim]
            x_right, x_left = x.chunk(2, dim=1)

            x_right = cross_ff_right(x_right) + x_right
            x_left = cross_ff_left(x_left) + x_left

            # self-attention layers
            for self_attn_right, self_ff_right, self_attn_left, self_ff_left in self.layers:

                x_right = self_attn_right(x_right) + x_right
                x_right = self_ff_right(x_right) + x_right

                x_left = self_attn_left(x_left) + x_left
                x_left = self_ff_left(x_left) + x_left

            
            x = torch.concat([x_right, x_left], dim=1)
            x = self.combined_latent_attn(x) + x
            x = self.combined_latent_ff(x) + x

        x_right, x_left = x.chunk(2, dim=1)

        # decoder cross attention
        latents_right = self.decoder_cross_attn_right(ins, context=x_right)
        latents_left = self.decoder_cross_attn_left(ins, context=x_left)

        # crop out the language part of the output sequence
        if self.lang_fusion_type == "seq":
            latents_right = latents_right[:, l.shape[1] :]
            latents_left = latents_left[:, l.shape[1] :]

        # reshape back to voxel grid
        latents_right = latents_right.view(
            b, *queries_orig_shape[1:-1], latents_right.shape[-1]
        )  # [B,20,20,20,64]
        latents_right = rearrange(latents_right, "b ... d -> b d ...")  # [B,64,20,20,20]

        # reshape back to voxel grid
        latents_left = latents_left.view(
            b, *queries_orig_shape[1:-1], latents_left.shape[-1]
        )  # [B,20,20,20,64]
        latents_left = rearrange(latents_left, "b ... d -> b d ...")  # [B,64,20,20,20]

        # aggregated features from 2nd softmax and maxpool for MLP decoders

        feats_right = feats.copy()
        feats_left = feats


        feats_right.extend(
            [self.ss1(latents_right.contiguous()), self.global_maxp(latents_right).view(b, -1)]
        )
        feats_left.extend(
            [self.ss1(latents_left.contiguous()), self.global_maxp(latents_left).view(b, -1)]
        )

        # upsample
        u0_right = self.up0(latents_right)
        u0_left = self.up0(latents_left)

        # ablations
        if self.no_skip_connection:
            u_right = self.final(u0_right)
            u_left = self.final(u0_left)
        elif self.no_perceiver:
            u_right = self.final(d0)
            u_left = self.final(d0)
        else:
            u_right = self.final(torch.cat([d0, u0_right], dim=1))
            u_left = self.final(torch.cat([d0, u0_left], dim=1))

        # translation decoder
        right_trans = self.right_trans_decoder(u_right)
        left_trans = self.left_trans_decoder(u_left)

        # rotation, gripper, and collision MLPs
        rot_and_grip_out = None
        if self.num_rotation_classes > 0:
            feats_right.extend(
                [self.ss_final(u_right.contiguous()), self.global_maxp(u_right).view(b, -1)]
            )

            right_dense0 = self.right_dense0(torch.cat(feats_right, dim=1))
            right_dense1 = self.right_dense1(right_dense0)  # [B,72*3+2+2]

            right_rot_and_grip_collision_out = self.right_rot_grip_collision_ff(
                right_dense1
            )
            right_rot_and_grip_out = right_rot_and_grip_collision_out[
                :, : -self.num_collision_classes
            ]
            right_collision_out = right_rot_and_grip_collision_out[
                :, -self.num_collision_classes :
            ]

            feats_left.extend(
                [self.ss_final(u_left.contiguous()), self.global_maxp(u_left).view(b, -1)]
            )

            left_dense0 = self.left_dense0(torch.cat(feats_left, dim=1))
            left_dense1 = self.left_dense1(left_dense0)  # [B,72*3+2+2]

            left_rot_and_grip_collision_out = self.left_rot_grip_collision_ff(
                left_dense1
            )
            left_rot_and_grip_out = left_rot_and_grip_collision_out[
                :, : -self.num_collision_classes
            ]
            left_collision_out = left_rot_and_grip_collision_out[
                :, -self.num_collision_classes :
            ]

        return (
            right_trans,
            right_rot_and_grip_out,
            right_collision_out,
            left_trans,
            left_rot_and_grip_out,
            left_collision_out,
        )
