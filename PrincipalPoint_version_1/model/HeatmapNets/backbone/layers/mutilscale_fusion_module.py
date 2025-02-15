import torch
from torch import nn
import torch.nn.functional as F


def channel_shuffle(x, groups: int):  # 输入是tensor 高维度数据，和一个整型，输出还是tensor
    # 在原作者的代码中将通道分离和重排操作集成在一起，参考上一章节的内容。其实我觉得那个更好一点，不过目的一致就行

    batch_size, num_channels, height, width = x.size()  # 查看收到的数据，对其形状进行分析，可以得到批次数量，通道个数，图片的大小
    channels_per_group = num_channels // groups  # 将通道数量进行分组，就知道多少个通道一组了

    # reshape
    # [batch_size, num_channels, height, width] -> [batch_size, groups, channels_per_group, height, width]
    x = x.view(batch_size, groups, channels_per_group, height, width)
    # 将数据利用分组进行操作了 简单理解groups * channels_per_group = num_channels。
    # 在代码层面上就是一种reshape呗
    x = torch.transpose(x, 1, 2).contiguous()

    # flatten成功完成了通道重组
    x = x.view(batch_size, -1, height, width)

    return x


class ConvModule(nn.Module):
    """Replacement for mmcv.cnn.ConvModule to avoid mmcv dependency."""

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            padding: int = 0,
            use_norm: bool = False,
            bias: bool = True,
    ):
        """Simple convolution block.

        Args:
            in_channels (int): Input channels.
            out_channels (int): Output channels.
            kernel_size (int): Kernel size.
            padding (int, optional): Padding. Defaults to 0.
            use_norm (bool, optional): Whether to use normalization. Defaults to False.
            bias (bool, optional): Whether to use bias. Defaults to True.
        """
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels) if use_norm else nn.Identity()
        self.activate = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.conv(x)
        x = self.bn(x)
        return self.activate(x)


class ResidualConvUnit(nn.Module):
    """Residual convolution module."""

    def __init__(self, features):
        """Simple residual convolution block.

        Args:
            features (int): number of features
        """
        super().__init__()

        self.conv1 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True)

        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        out = self.relu(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)
        return out + x


class FeatureFusionBlock(nn.Module):
    """Feature fusion block."""

    def __init__(self, features: int, unit2only=False, upsample=True):
        """Feature fusion block.

        Args:
            features (int): Number of features.
            unit2only (bool, optional): Whether to use only the second unit. Defaults to False.
            upsample (bool, optional): Whether to upsample. Defaults to True.
        """
        super().__init__()
        self.upsample = upsample

        if not unit2only:
            self.resConfUnit1 = ResidualConvUnit(features)
        self.resConfUnit2 = ResidualConvUnit(features)

    def forward(self, *xs: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        output = xs[0]

        if len(xs) == 2:
            output = output + self.resConfUnit1(xs[1])

        output = self.resConfUnit2(output)

        if self.upsample:
            output = F.interpolate(output, scale_factor=2, mode="bilinear", align_corners=False)

        return output


###################################################
########### Light Hamburger Decoder ###############
###################################################


class NMF2D(nn.Module):
    """Non-negative Matrix Factorization (NMF) for 2D data."""

    def __init__(self):
        """Non-negative Matrix Factorization (NMF) for 2D data."""
        super().__init__()
        self.S, self.D, self.R = 1, 512, 64
        self.train_steps = 6
        self.eval_steps = 7
        self.inv_t = 1

    def _build_bases(self, B: int, S: int, D: int, R: int, device: str = "cpu") -> torch.Tensor:
        bases = torch.rand((B * S, D, R)).to(device)
        return F.normalize(bases, dim=1)

    def local_step(self, x, bases, coef):
        """Update bases and coefficient."""
        # (B * S, D, N)^T @ (B * S, D, R) -> (B * S, N, R)
        numerator = torch.bmm(x.transpose(1, 2), bases)
        # (B * S, N, R) @ [(B * S, D, R)^T @ (B * S, D, R)] -> (B * S, N, R)
        denominator = coef.bmm(bases.transpose(1, 2).bmm(bases))
        # Multiplicative Update
        coef = coef * numerator / (denominator + 1e-6)
        # (B * S, D, N) @ (B * S, N, R) -> (B * S, D, R)
        numerator = torch.bmm(x, coef)
        # (B * S, D, R) @ [(B * S, N, R)^T @ (B * S, N, R)] -> (B * S, D, R)
        denominator = bases.bmm(coef.transpose(1, 2).bmm(coef))
        # Multiplicative Update
        bases = bases * numerator / (denominator + 1e-6)
        return bases, coef

    def compute_coef(
            self, x: torch.Tensor, bases: torch.Tensor, coef: torch.Tensor
    ) -> torch.Tensor:
        """Compute coefficient."""
        # (B * S, D, N)^T @ (B * S, D, R) -> (B * S, N, R)
        numerator = torch.bmm(x.transpose(1, 2), bases)
        # (B * S, N, R) @ (B * S, D, R)^T @ (B * S, D, R) -> (B * S, N, R)
        denominator = coef.bmm(bases.transpose(1, 2).bmm(bases))
        # multiplication update
        return coef * numerator / (denominator + 1e-6)

    def local_inference(
            self, x: torch.Tensor, bases: torch.Tensor
    ):
        """Local inference."""
        # (B * S, D, N)^T @ (B * S, D, R) -> (B * S, N, R)
        coef = torch.bmm(x.transpose(1, 2), bases)
        coef = F.softmax(self.inv_t * coef, dim=-1)

        steps = self.train_steps if self.training else self.eval_steps
        for _ in range(steps):
            bases, coef = self.local_step(x, bases, coef)

        return bases, coef

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        B, C, H, W = x.shape

        # (B, C, H, W) -> (B * S, D, N)
        D = C // self.S
        N = H * W
        x = x.contiguous().view(B * self.S, D, N)

        # (S, D, R) -> (B * S, D, R)
        bases = self._build_bases(B, self.S, D, self.R, device=x.device)
        bases, coef = self.local_inference(x, bases)
        # (B * S, N, R)
        coef = self.compute_coef(x, bases, coef)
        # (B * S, D, R) @ (B * S, N, R)^T -> (B * S, D, N)
        x = torch.bmm(bases, coef.transpose(1, 2))
        # (B * S, D, N) -> (B, C, H, W)
        x = x.contiguous().view(B, C, H, W)
        # (B * H, D, R) -> (B, H, N, D)
        # bases = bases.view(B, self.S, D, self.R)

        return x


class Hamburger(nn.Module):
    """Hamburger Module."""

    def __init__(self, ham_channels: int = 512):
        """Hambuger Module.

        Args:
            ham_channels (int, optional): Number of channels in the hamburger module. Defaults to
            512.
        """
        super().__init__()
        self.ham_in = ConvModule(ham_channels, ham_channels, 1)
        self.ham = NMF2D()
        self.ham_out = ConvModule(ham_channels, ham_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        enjoy = self.ham_in(x)
        enjoy = F.relu(enjoy, inplace=False)
        enjoy = self.ham(enjoy)
        enjoy = self.ham_out(enjoy)
        ham = F.relu(x + enjoy, inplace=False)
        return ham


class LightHamHead(nn.Module):
    """Is Attention Better Than Matrix Decomposition?

    This head is the implementation of `HamNet <https://arxiv.org/abs/2109.04553>`.
    """

    def __init__(self, in_index_list=None):
        """Light hamburger decoder head."""
        super().__init__()
        if in_index_list is None:
            in_index_list = [3, 2, 1, 0]
        self.in_index = in_index_list
        self.in_channels = [48, 96, 192, 384]
        self.out_channels = 48
        self.ham_channels = 384
        self.align_corners = False

        self.squeeze = ConvModule(sum(self.in_channels), self.ham_channels, 1)

        self.hamburger = Hamburger(self.ham_channels)

        self.align = ConvModule(self.ham_channels, self.out_channels, 1)

        # self.linear_pred_uncertainty = nn.Sequential(
        #     ConvModule(
        #         in_channels=self.out_channels,
        #         out_channels=self.out_channels,
        #         kernel_size=3,
        #         padding=1,
        #         bias=False,
        #     ),
        #     nn.Conv2d(in_channels=self.out_channels, out_channels=1, kernel_size=1),
        # )

        self.out_conv = nn.Sequential(
            ConvModule(
                in_channels=self.out_channels,
                out_channels=self.out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.Conv2d(in_channels=self.out_channels, out_channels=1, kernel_size=1),
        )
        # ConvModule(self.out_channels, self.out_channels, 3, padding=1, bias=False)

        # self.ll_out_conv = ConvModule(256, self.out_channels, 3, padding=1, bias=False)
        # self.ll_fusion = FeatureFusionBlock(self.out_channels, upsample=False)

    def forward(self, features):
        """Forward pass."""
        inputs = [features[i] for i in self.in_index]  # list[0, 1, 2, 3] [64, 128, 320, 512]

        inputs = [
            F.interpolate(
                level, size=inputs[0].shape[2:], mode="bilinear", align_corners=self.align_corners
            )
            for level in inputs
        ]  # [1, (64, 128, 320, 512),80, 104]

        inputs = torch.cat(inputs, dim=1)  # [1, 1024, 80, 104]
        x = self.squeeze(inputs)  # [1, 512, 80, 104]

        x = self.hamburger(x)  # [1, 512, 80, 104]

        feats = self.align(x)  # [1, 64, 80, 104]

        # assert "ll" in features, "Low-level features are required for this model"
        # feats = F.interpolate(feats, scale_factor=2, mode="bilinear", align_corners=False)
        feats = self.out_conv(feats)
        # ll_feats = F.interpolate(ll_feature, scale_factor=2, mode="bilinear", align_corners=False)
        # ll_feats = self.ll_out_conv(ll_feats)
        # feats = self.ll_fusion(feats)  # [1, 64, 320, 416]

        # uncertainty = self.linear_pred_uncertainty(feats).squeeze(1)

        return feats  # [1, 320, 416]


