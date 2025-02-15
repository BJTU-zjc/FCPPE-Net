import cv2
import numpy as np
import torch.nn as nn
from collections import OrderedDict

from PrincipalPoint_version_1.model.HeatmapNets.backbone import hrnet2_yolo_edge_contourlet_v1_4


class ModelMain_PL(nn.Module):
    def __init__(self, ):
        super(ModelMain_PL, self).__init__()
        self.backbone = hrnet2_yolo_edge_contourlet_v1_4.hrnet_yolo()
        final_out_filter0 = 3
        self.embedding0 = self._make_embedding([64, 128], 48 + 2,
                                               final_out_filter0)  # for hrnet2yolo DCD or h1 or hrnethyolo DCD

    def _make_cbl(self, _in, _out, ks):
        ''' cbl = conv + batch_norm + leaky_relu
        '''
        pad = (ks - 1) // 2 if ks else 0
        return nn.Sequential(OrderedDict([
            ("conv", nn.Conv2d(_in, _out, kernel_size=ks, stride=1, padding=pad, bias=False)),
            ("bn", nn.BatchNorm2d(_out)),
            ("relu", nn.LeakyReLU(0.1)),
        ]))

    def _make_embedding(self, filters_list, in_filters, out_filter):
        m = nn.ModuleList([
            self._make_cbl(in_filters, filters_list[0], 1),
            self._make_cbl(filters_list[0], filters_list[1], 3),
        ])
        m.add_module("conv_out", nn.Conv2d(filters_list[1], out_filter, kernel_size=1,
                                           stride=1, padding=0, bias=True))
        return m

    def forward(self, x, inl):
        def _branch(_embedding, _in):
            for i, e in enumerate(_embedding):
                _in = e(_in)
            return _in

        hm, hmh, out0, offset = self.backbone(x, inl)
        out = _branch(self.embedding0, out0)  # (8,3,160,160)

        return hm, hmh, out, offset  # for hrnet2 yolo
