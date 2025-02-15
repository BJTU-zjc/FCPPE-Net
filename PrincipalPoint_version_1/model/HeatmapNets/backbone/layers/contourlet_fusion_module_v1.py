import torch
import torch.nn as nn
import torch.nn.functional as F

from PrincipalPoint_version_1.model.HeatmapNets.backbone.layers.layer import Bottleneck1, BasicBlock1
from PrincipalPoint_version_1.model.HeatmapNets.backbone.layers.mutilscale_fusion_module import channel_shuffle
from PrincipalPoint_version_1.model.HeatmapNets.backbone.layers.contourlet_lane_mask import Contourlet_edge_exact_v2


class SGM(nn.Module):
    def __init__(self, in_c):
        super(SGM, self).__init__()
        # downsample
        # self.conv1 = nn.Sequential(
        #     nn.Conv2d(in_c, in_c * 2, 3, 2, 1, bias=False),
        #     nn.BatchNorm2d(in_c * 2),
        #     nn.ReLU(inplace=True))
        # self.conv2 = nn.Sequential(
        #     nn.Conv2d(in_c * 2, in_c * 4, 3, 2, 1, bias=False),
        #     nn.BatchNorm2d(in_c * 4),
        #     nn.ReLU(inplace=True))
        # self.conv3 = nn.Sequential(
        #     nn.Conv2d(in_c * 4, in_c * 8, 3, 2, 1, bias=False),
        #     nn.BatchNorm2d(in_c * 8),
        #     nn.ReLU(inplace=True))
        self.contourlet_down1 = nn.Sequential(
            nn.Conv2d(in_c, in_c * 2, 3, 2, 1, bias=False),
            nn.BatchNorm2d(in_c * 2),
            nn.ReLU(inplace=True))
        self.contourlet_down2 = nn.Sequential(
            nn.Conv2d(in_c * 2, in_c * 4, 3, 2, 1, bias=False),
            nn.BatchNorm2d(in_c * 4),
            nn.ReLU(inplace=True))
        self.contourlet_down3 = nn.Sequential(
            nn.Conv2d(in_c * 4, in_c * 8, 3, 2, 1, bias=False),
            nn.BatchNorm2d(in_c * 8),
            nn.ReLU(inplace=True))
        self.contourlet_conv1 = Bottleneck1(inplanes=in_c * 2, planes=in_c * 2)
        self.contourlet_conv2 = Bottleneck1(inplanes=in_c * 4, planes=in_c * 4)
        self.contourlet_conv3 = Bottleneck1(inplanes=in_c * 8, planes=in_c * 8)
        # out
        self.out_contourlet_conv1 = BasicBlock1(inplanes=in_c * 4, planes=in_c * 2)
        self.out_contourlet_conv2 = BasicBlock1(inplanes=in_c * 8, planes=in_c * 4)
        self.out_contourlet_conv3 = BasicBlock1(inplanes=in_c * 16, planes=in_c * 8)

        # self.out_conv1 = nn.Sequential(
        #     nn.Conv2d(in_c*2, in_c * 2, 3, 2, 1, bias=False),
        #     nn.BatchNorm2d(in_c * 2),
        #     nn.ReLU(inplace=True))
        # self.out_conv2 = nn.Sequential(
        #     nn.Conv2d(in_c * 4, in_c * 4, 3, 2, 1, bias=False),
        #     nn.BatchNorm2d(in_c * 4),
        #     nn.ReLU(inplace=True))
        # self.out_conv3 = nn.Sequential(
        #     nn.Conv2d(in_c * 8, in_c * 8, 3, 2, 1, bias=False),
        #     nn.BatchNorm2d(in_c * 8),
        #     nn.ReLU(inplace=True))

        self.CEE1 = Contourlet_edge_exact_v2(in_c, in_c * 1)
        self.CEE2 = Contourlet_edge_exact_v2(in_c, in_c * 2)
        self.CEE3 = Contourlet_edge_exact_v2(in_c, in_c * 4)
        self.CEE4 = Contourlet_edge_exact_v2(in_c, in_c * 8)

    def forward(self, ce_x1, ce_x2, ce_x3, ce_x4):
        # x_list = []
        cee_list = []

        edge1, x_l1 = self.CEE1(ce_x1)
        # x_list.append(x_l1)
        cee_list.append(edge1)

        edge2, x_l2 = self.CEE2(ce_x2)
        edge_l1 = self.contourlet_conv1(self.contourlet_down1(edge1))
        # xll_1 = self.conv1(x_l1)
        edge_out_2 = self.out_contourlet_conv1(channel_shuffle(torch.cat((F.interpolate(edge2, scale_factor=0.5), edge_l1), 1), 12))
        # x_out_2 = self.out_conv1(torch.cat((x_l2, xll_1), 1))
        # x_list.append(x_out_2)
        cee_list.append(edge_out_2)

        edge3, x_l3 = self.CEE3(ce_x3)
        edge_l2 = self.contourlet_conv2(self.contourlet_down2(edge_out_2))
        # xll_2 = self.conv1(xll_1)
        edge_out_3 = self.out_contourlet_conv2(channel_shuffle(torch.cat((F.interpolate(edge3, scale_factor=0.25), edge_l2), 1), 12))
        # x_out_3 = self.out_conv2(torch.cat((x_l3, xll_2), 1))
        # x_list.append(x_out_3)
        cee_list.append(edge_out_3)

        edge4, x_l4 = self.CEE4(ce_x4)
        edge_l3 = self.contourlet_conv3(self.contourlet_down3(edge_out_3))
        # xll_3 = self.conv1(xll_2)
        edge_out_4 = self.out_contourlet_conv3(channel_shuffle(torch.cat((F.interpolate(edge4, scale_factor=0.125), edge_l3), 1), 12))
        # x_out_4 = self.out_conv2(torch.cat((x_l4, xll_3), 1))
        # x_list.append(edge_out_4)
        cee_list.append(edge_out_4)

        return cee_list
