# ------------------------------------------------------------------------------
# Copyright (c) TJU
# Licensed under the MIT License.
# Written by Yinbo Liu (liushuo618@gmail.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os

import torch
import torch.nn as nn

from PrincipalPoint_version_1.model.HeatmapNets.backbone.layers.contourlet_fusion_module import SGM
from PrincipalPoint_version_1.model.HeatmapNets.backbone.layers.contourlet_lane_mask import Contourlet_edge_exact
from PrincipalPoint_version_1.model.HeatmapNets.backbone.layers.layer import BasicBlock, Bottleneck
from PrincipalPoint_version_1.model.HeatmapNets.backbone.layers.mutilscale_fusion_module import LightHamHead, \
    channel_shuffle
from PrincipalPoint_version_1.model.HeatmapNets.backbone.layers.upsample_module import DySample

BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)
__all__ = ['hrnet_yolo']


class HighResolutionModule(nn.Module):
    def __init__(self, num_branches, blocks, num_blocks, num_inchannels,
                 num_channels, fuse_method, multi_scale_output=True):
        super(HighResolutionModule, self).__init__()
        self._check_branches(
            num_branches, blocks, num_blocks, num_inchannels, num_channels)

        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(
            num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(True)

    def _check_branches(self, num_branches, blocks, num_blocks,
                        num_inchannels, num_channels):
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(
                num_branches, len(num_blocks))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(
                num_branches, len(num_channels))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_inchannels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(
                num_branches, len(num_inchannels))
            logger.error(error_msg)
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels,
                         stride=1):
        downsample = None
        if stride != 1 or \
                self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.num_inchannels[branch_index],
                    num_channels[branch_index] * block.expansion,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(
                    num_channels[branch_index] * block.expansion,
                    momentum=BN_MOMENTUM
                ),
            )

        layers = []
        layers.append(
            block(
                self.num_inchannels[branch_index],
                num_channels[branch_index],
                stride,
                downsample
            )
        )
        self.num_inchannels[branch_index] = \
            num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(
                block(
                    self.num_inchannels[branch_index],
                    num_channels[branch_index]
                )
            )

        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []

        for i in range(num_branches):
            branches.append(
                self._make_one_branch(i, block, num_blocks, num_channels)
            )

        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(
                        nn.Sequential(
                            nn.Conv2d(
                                num_inchannels[j],
                                num_inchannels[i],
                                1, 1, 0, bias=False
                            ),
                            nn.BatchNorm2d(num_inchannels[i]),
                            nn.Upsample(scale_factor=2 ** (j - i), mode='nearest')
                        )
                    )
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.Conv2d(
                                        num_inchannels[j],
                                        num_outchannels_conv3x3,
                                        3, 2, 1, bias=False
                                    ),
                                    nn.BatchNorm2d(num_outchannels_conv3x3)
                                )
                            )
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.Conv2d(
                                        num_inchannels[j],
                                        num_outchannels_conv3x3,
                                        3, 2, 1, bias=False
                                    ),
                                    nn.BatchNorm2d(num_outchannels_conv3x3),
                                    nn.ReLU(True)
                                )
                            )
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []

        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))

        return x_fuse


blocks_dict = {
    'BASIC': BasicBlock,
    'BOTTLENECK': Bottleneck
}


class PoseHighResolutionNet(nn.Module):

    def __init__(self, **kwargs):
        self.inplanes = 64
        # extra = cfg.MODEL.EXTRA
        super(PoseHighResolutionNet, self).__init__()

        # stem net
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1,
                               bias=False, groups=64)  # , groups=64
        self.bn2 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(Bottleneck, 64, 4)

        self.stage2_cfg = {'NUM_MODULES': 1,
                           'NUM_BRANCHES': 2,
                           'NUM_BLOCKS': [4, 4],
                           'NUM_CHANNELS': [48, 96],
                           'BLOCK': 'BASIC',
                           'FUSE_METHOD': 'SUM', }
        num_channels = self.stage2_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage2_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition1 = self._make_transition_layer([256], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(
            self.stage2_cfg, num_channels)
        self.stage3_cfg = {'NUM_MODULES': 4,
                           'NUM_BRANCHES': 3,
                           'NUM_BLOCKS': [4, 4, 4],
                           'NUM_CHANNELS': [48, 96, 192],
                           'BLOCK': 'BASIC',
                           'FUSE_METHOD': 'SUM', }
        num_channels = self.stage3_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage3_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition2 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(
            self.stage3_cfg, num_channels)
        self.stage4_cfg = {'NUM_MODULES': 3,
                           'NUM_BRANCHES': 4,
                           'NUM_BLOCKS': [4, 4, 4, 4],
                           'NUM_CHANNELS': [48, 96, 192, 384],
                           'BLOCK': 'BASIC',
                           'FUSE_METHOD': 'SUM', }
        num_channels = self.stage4_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage4_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition3 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage4, pre_stage_channels = self._make_stage(
            self.stage4_cfg, num_channels, multi_scale_output=True)

        self.final_layer = nn.Conv2d(
            in_channels=pre_stage_channels[0],
            out_channels=1,  # output num
            kernel_size=1,
            stride=1,
            padding=1 if 1 == 3 else 0
        )
        self.final_layerh = nn.Conv2d(
            in_channels=pre_stage_channels[0] + 1,  # *2 for hlh out others *1
            out_channels=1,  # output num
            kernel_size=1,
            stride=1,
            padding=1 if 1 == 3 else 0
        )
        self.contour_exact = SGM(48)
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(49, 48, 3, 1, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )
        self.pretrained_layers = ''
        self.fusion = LightHamHead([0, 1, 2, 3])
        self.up1 = DySample(in_channels=49)
        # self.up1 = UpBlock1(pre_stage_channels[0], 4, 2, 1)
        # self.down1 = DownBlock(pre_stage_channels[0], 4, 2, 1)
        # self.up2 = UpBlock(pre_stage_channels[0], 4, 2, 1)

    def _make_transition_layer(
            self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(
                        nn.Sequential(
                            nn.Conv2d(
                                num_channels_pre_layer[i],
                                num_channels_cur_layer[i],
                                3, 1, 1, bias=False
                            ),
                            nn.BatchNorm2d(num_channels_cur_layer[i]),
                            nn.ReLU(inplace=True)
                        )
                    )
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i + 1 - num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] \
                        if j == i - num_branches_pre else inchannels
                    conv3x3s.append(
                        nn.Sequential(
                            nn.Conv2d(
                                inchannels, outchannels, 3, 2, 1, bias=False
                            ),
                            nn.BatchNorm2d(outchannels),
                            nn.ReLU(inplace=True)
                        )
                    )
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_deconv_layers(self, input_channels):
        dim_tag = 17
        deconv_layers = []
        for i in range(1):
            final_output_channels = 17 + dim_tag
            input_channels = 49
            output_channels = 48
            deconv_kernel, padding, output_padding = (4, 1, 0)
            layers = []
            layers.append(nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels=input_channels,
                    out_channels=output_channels,
                    kernel_size=deconv_kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=False),
                nn.BatchNorm2d(output_channels, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True)
            ))
            deconv_layers.append(nn.Sequential(*layers))
            input_channels = output_channels

        return nn.ModuleList(deconv_layers)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes, planes * block.expansion,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_stage(self, layer_config, num_inchannels,
                    multi_scale_output=True):
        num_modules = layer_config['NUM_MODULES']
        num_branches = layer_config['NUM_BRANCHES']
        num_blocks = layer_config['NUM_BLOCKS']
        num_channels = layer_config['NUM_CHANNELS']
        block = blocks_dict[layer_config['BLOCK']]
        fuse_method = layer_config['FUSE_METHOD']

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True

            modules.append(
                HighResolutionModule(
                    num_branches,
                    block,
                    num_blocks,
                    num_inchannels,
                    num_channels,
                    fuse_method,
                    reset_multi_scale_output
                )
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels


    def forward(self, x, inl):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)
        x_list1 = []
        for i in range(self.stage2_cfg['NUM_BRANCHES']):
            if self.transition1[i] is not None:
                x_list1.append(self.transition1[i](x))
            else:
                x_list1.append(x)

        y_list1 = self.stage2(x_list1)

        x_list2 = []
        for i in range(self.stage3_cfg['NUM_BRANCHES']):
            if self.transition2[i] is not None:
                x_list2.append(self.transition2[i](y_list1[-1]))
            else:
                x_list2.append(y_list1[i])
        y_list2 = self.stage3(x_list2)

        x_list3 = []
        for i in range(self.stage4_cfg['NUM_BRANCHES']):
            if self.transition3[i] is not None:
                x_list3.append(self.transition3[i](y_list2[-1]))
            else:
                x_list3.append(y_list2[i])
        y_list3 = self.stage4(x_list3)

        cef_list = self.contour_exact(x_list1[0], y_list1[0], y_list2[0], y_list3[0])
        edge_fusion = self.fusion(cef_list)
        # feature_fusion = self.fusion_conv(channel_shuffle(torch.cat((edge_fusion, y_list3[0]), 1), 7))
        outl = self.final_layer(y_list3[0])
        h1, offset = self.up1(torch.cat((inl, y_list3[0]), 1), torch.cat((edge_fusion, cef_list[0]), 1))
        outh = self.final_layerh(h1)  # for only h1

        coord = torch.cat((outh, h1), 1)  # for high res out only h1

        return outl, outh, coord, offset


def init_weights(self, pretrained=''):
    logger.info('=> init weights from normal distribution')
    for m in self.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, std=0.001)
            for name, _ in m.named_parameters():
                if name in ['bias']:
                    nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.ConvTranspose2d):
            nn.init.normal_(m.weight, std=0.001)
            for name, _ in m.named_parameters():
                if name in ['bias']:
                    nn.init.constant_(m.bias, 0)

    if os.path.isfile(pretrained):
        pretrained_state_dict = torch.load(pretrained)
        logger.info('=> loading pretrained model {}'.format(pretrained))

        need_init_state_dict = {}
        for name, m in pretrained_state_dict.items():
            if name.split('.')[0] in self.pretrained_layers \
                    or self.pretrained_layers[0] is '*':
                need_init_state_dict[name] = m
        self.load_state_dict(need_init_state_dict, strict=False)
    elif pretrained:
        logger.error('=> please download pre-trained models first!')
        raise ValueError('{} is not exist!'.format(pretrained))


def hrnet_yolo(**kwargs):
    model = PoseHighResolutionNet(**kwargs)
    # if pretrained:
    #     if isinstance(pretrained, str):
    #         dict_trained = torch.load(pretrained)
    #         dict_new = model.state_dict().copy()
    #         new_list = list(model.state_dict().keys())
    #         trained_list = list(dict_trained.keys())
    #         for i in range(1752):
    #             dict_new[new_list[i]] = dict_trained[trained_list[i]]
    #         print("mobilenetv2 pretrained loaded !!!")
    #         # print(dict_new)
    #
    #         model.load_state_dict(dict_new)
    #     else:
    #         raise Exception("mobilenetv1 request a pretrained path. got [{}]".format(pretrained))
    return model


if __name__ == "__main__":
    backbone = hrnet_yolo()
    x = torch.rand(4, 3, 320, 320)
    hmh, out0, offset = backbone(x, x)
