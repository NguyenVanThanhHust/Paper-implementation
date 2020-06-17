import fvcore.nn.weight_init as weight_init
import torch
import torch.nn.functional as F

from detectron2.layers import Conv2d, FrozenBatchNorm2d, get_norm
from detectron2.modeling import BACKBONE_REGISTRY, ResNet, ResNetBlockBase, make_stage
from detectron2.modeling.backbone.resnet import BasicStem, BottleneckBlock, DeformBottleneckBlock

from .trident_conv import TridentConv

__all__ = ["TridentBottleneckBlock", "make_trident_stage", "build_trident_resnet_backbone"]

class TridentBottleneckBlock(ResNetBlockBase):
    def __init__(
        self,
        in_channels,
        out_channels,
        *,
        bottleneck_channels,
        stride=1,
        num_groups=1,
        norm="BN",
        stride_in_1x1=False,
        num_branch=3,
        dilations=(1, 2, 3),
        concat_output=False,
        test_branch_idx=-1,
    ):
        """
        Args:
            num_branch (int): the number of branches in TridentNet.
            dilations (tuple): the dilations of multiple branches in TridentNet.
            concat_output (bool): if concatenate outputs of multiple branches in TridentNet.
                Use 'True' for the last trident block.
        """
        super().__init__(in_channels, out_channels, stride)

        assert num_branch == len(dilations)

        self.num_branch = num_branch
        self.concat_output = concat_output
        self.test_branch_idx = test_branch_idx

        if in_channels != out_channels:
            self.shortcut = Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                bias=False,
                norm=get_norm(norm, out_channels),
            )
        else:
            self.shortcut = None

        stride_1x1, stride_3x3 = (stride, 1) if stride_in_1x1 else (1, stride)

        self.conv1 = Conv2d(
            in_channels,
            bottleneck_channels,
            kernel_size=1,
            stride=stride_1x1,
            bias=False,
            norm=get_norm(norm, bottleneck_channels),
        )

        self.conv2 = TridentConv(
            bottleneck_channels,
            bottleneck_channels,
            kernel_size=3,
            stride=stride_3x3,
            paddings=dilations,
            bias=False,
            groups=num_groups,
            dilations=dilations,
            num_branch=num_branch,
            test_branch_idx=test_branch_idx,
            norm=get_norm(norm, bottleneck_channels),
        )

        self.conv3 = Conv2d(
            bottleneck_channels,
            out_channels,
            kernel_size=1,
            bias=False,
            norm=get_norm(norm, out_channels),
        )

        for layer in [self.conv1, self.conv2, self.conv3, self.shortcut]:
            if layer is not None:  # shortcut can be None
                weight_init.c2_msra_fill(layer)

    def forward(self, x):
        num_branch = self.num_branch if self.training or self.test_branch_idx == -1 else 1
        if not isinstance(x, list):
            x = [x] * num_branch
        out = [self.conv1(b) for b in x]
        out = [F.relu_(b) for b in out]

        out = self.conv2(out)
        out = [F.relu_(b) for b in out]

        out = [self.conv3(b) for b in out]

        if self.shortcut is not None:
            shortcut = [self.shortcut(b) for b in x]
        else:
            shortcut = x

        out = [out_b + shortcut_b for out_b, shortcut_b in zip(out, shortcut)]
        out = [F.relu_(b) for b in out]
        if self.concat_output:
            out = torch.cat(out)
        return out

def make_trident_stage(block_class, num_block, first_stride, **kwargs):
    """
    Create a resnet stage by creating many block for TridentNet
    """
    blocks = []
    for i in range(num_blocks - 1):
        blocks.append(block_class())

@BACKBONE_REGISTRY.register()
def build_trident_resnet_backbone(cfg, input_shape):
    """
    """
    #  need registration of new blocks
    norm = cfg.MODEL.RESNETS.NORM
    stem = BasicStem(
        in_channels=input_shape.channels,
        out_channels = cfg.MODEL.RESNET.STEM_OUT_CHANNELS, 
        norm = norm
    )
    freeze_at = cfg.MODEL.BACKBONE.FREEZE_AT
    if freeze_at >= 1:
        for p in stem.parameters():
            p.requires_grad = False
        stem = FrozenBatchNorm2d.convert_frozen_batchnorm(stem) 
