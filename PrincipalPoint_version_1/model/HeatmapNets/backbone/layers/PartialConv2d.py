import torch
import torch.nn.functional as F
from torch import nn


class PartialConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        # whether the mask is multi-channel or not
        if 'multi_channel' in kwargs:
            self.multi_channel = kwargs['multi_channel']
            kwargs.pop('multi_channel')
        else:
            self.multi_channel = False
        # 设置为True表示同时返回卷积后的特征和更新后的掩模M
        self.return_mask = True
        # 这里的继承包括继承父类nn.Conv2d中，__init__函数里定义的所有属性，包括前向传播用到的偏置项
        super(PartialConv2d, self).__init__(*args, **kwargs)
        if self.multi_channel:
            self.weight_maskUpdater = torch.ones(self.out_channels, self.in_channels, self.kernel_size[0],
                                                 self.kernel_size[1])
        else:
            self.weight_maskUpdater = torch.ones(1, 1, self.kernel_size[0], self.kernel_size[1])
        # 滑动窗口的尺寸，即长乘宽，对应论文中的sum(1)
        self.slide_winsize = self.weight_maskUpdater.shape[1] * self.weight_maskUpdater.shape[2] * \
                             self.weight_maskUpdater.shape[3]

        self.last_size = (None, None, None, None)
        self.update_mask = None
        self.mask_ratio = None

    def forward(self, input, mask_in=None):
        assert len(input.shape) == 4
        if mask_in is not None or self.last_size != tuple(input.shape):
            self.last_size = tuple(input.shape)
            # 以下操作不产生梯度
            with torch.no_grad():
                # 保证mask卷积参数与输入图像的数据类型一致
                if self.weight_maskUpdater.type() != input.type():
                    self.weight_maskUpdater = self.weight_maskUpdater.to(input)
                # 如果不存在掩模，则创建一个全一数组当做掩模图
                if mask_in is None:
                    # if mask is not provided, create a mask
                    if self.multi_channel:
                        mask = torch.ones(input.data.shape[0], input.data.shape[1], input.data.shape[2],
                                          input.data.shape[3]).to(input)
                    else:
                        mask = torch.ones(1, 1, input.data.shape[2], input.data.shape[3]).to(input)
                else:
                    mask = mask_in
                # 先对掩模做卷积，得到论文中的sum(M)
                self.update_mask = F.conv2d(mask, self.weight_maskUpdater, bias=None, stride=self.stride,
                                            padding=self.padding, dilation=self.dilation, groups=1)
                # 这里得到掩模比率，用于调整有效输入的特征大小，相当于论文中的sum(1)/sum(M)
                self.mask_ratio = self.slide_winsize / (self.update_mask + 1e-8)
                # 通过限制最大为1，最小为0，从而得到论文中的m'
                self.update_mask = torch.clamp(self.update_mask, 0, 1)
                # 更新后的掩模图与之前得到的比率相乘，使sum(M)<0时的比率变为0，用于后续和卷积后的图像做点乘，得到x'
                # 即对应公式(1)中，sum(M)<0的像素点经过局部卷积后像素点依然是0
                self.mask_ratio = torch.mul(self.mask_ratio, self.update_mask)
        # 这里执行PartialConv2d父类的前向传播操作，而父类为nn.Conv2d，因此相当于直接做一个卷积运算
        # 如果存在掩模M，则先让输入图像与掩模M做点乘，之后做卷积运算，否则直接让输入图像做卷积运算
        raw_out = super(PartialConv2d, self).forward(torch.mul(input, mask) if mask_in is not None else input)
        # 因为这个类继承nn.Conv2d，因此这里的self.bias是nn.Conv2d中自带的偏置项
        if self.bias is not None:
            # 如果该卷积操作存在偏置项的话，则先让卷积操作减去偏置项，之后再与掩模比率相乘，最后再加上偏置项
            # 对应论文中公式1，先让权重乘以输入数据，之后乘以掩模比率，最后加上偏置项
            bias_view = self.bias.view(1, self.out_channels, 1, 1)
            output = torch.mul(raw_out - bias_view, self.mask_ratio) + bias_view
            # 再让输出的数据与更新后的掩模图相乘(相当于再减去sum(M)<0处的偏置数据)
            output = torch.mul(output, self.update_mask)
        else:
            # 如果没有偏置项的话，则直接做点乘即可
            output = torch.mul(raw_out, self.mask_ratio)
        # 返回输出数据和更新后的掩模图
        if self.return_mask:
            return output, self.update_mask
        else:
            return output
