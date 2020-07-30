import torch.nn as nn

from ..builder import BACKBONES
from .base_backbone import BaseBackbone
import numbers
import collections
import logging
import functools
import torch
from torch import nn
from torch.nn import functional as F


def get_active_fn(name):
    """Select activation function."""
    active_fn = {
        'nn.ReLU6': functools.partial(nn.ReLU6, inplace=True),
        'nn.ReLU': functools.partial(nn.ReLU, inplace=True),
    }[name]
    return active_fn


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class ConvBNReLU(nn.Sequential):
    """Convolution-BatchNormalization-ActivateFn."""

    def __init__(self,
                 in_planes,
                 out_planes,
                 kernel_size=3,
                 stride=1,
                 groups=1,
                 active_fn=None,
                 batch_norm_kwargs=None,
                 expand_ratio=None,
                 kernel_sizes=None):
        if batch_norm_kwargs is None:
            batch_norm_kwargs = {}
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes,
                      out_planes,
                      kernel_size,
                      stride,
                      padding,
                      groups=groups,
                      bias=False),
            nn.BatchNorm2d(out_planes, **batch_norm_kwargs), active_fn())


class InvertedResidualChannels(nn.Module):
    """MobiletNetV2 building block."""

    def __init__(self,
                 inp,
                 oup,
                 stride,
                 kernel_sizes=[3, 5, 7],
                 expand=True,
                 expand_ratio=2,
                 active_fn=nn.ReLU,
                 batch_norm_kwargs={'momentum': 0.1, 'eps': 1e-5}):
        super(InvertedResidualChannels, self).__init__()
        print(kernel_sizes,expand_ratio)
        assert stride in [1, 2]
#         assert len(channels) == len(kernel_sizes)

        channels = [expand_ratio * inp for item in kernel_sizes]
        self.input_dim = inp
        self.output_dim = oup
        self.expand = expand
        self.stride = stride
        self.kernel_sizes = kernel_sizes
        self.channels = channels
        self.use_res_connect = self.stride == 1 and inp == oup
        self.batch_norm_kwargs = batch_norm_kwargs
        self.active_fn = active_fn

        self.ops, self.pw_bn = self._build(channels, kernel_sizes, expand)

        if not self.use_res_connect:  # TODO(Mingyu): Add this residual
            # assert (self.input_dim % min(self.input_dim, self.output_dim) == 0
            #         and self.output_dim % min(self.input_dim, self.output_dim) == 0)
            group = [x for x in range(1, self.input_dim + 1)
                     if self.input_dim % x == 0 and self.output_dim % x == 0][-1]
            self.residual = nn.Conv2d(self.input_dim,
                                      self.output_dim,
                                      kernel_size=1,
                                      stride=self.stride,
                                      padding=0,
                                      groups=group,
                                      bias=False)

    def _build(self, hidden_dims, kernel_sizes, expand):
        _batch_norm_kwargs = self.batch_norm_kwargs \
            if self.batch_norm_kwargs is not None else {}

        narrow_start = 0
        ops = nn.ModuleList()
        for k, hidden_dim in zip(kernel_sizes, hidden_dims):
            layers = []
            if expand:
                # pw
                layers.append(
                    ConvBNReLU(self.input_dim,
                               hidden_dim,
                               kernel_size=1,
                               batch_norm_kwargs=_batch_norm_kwargs,
                               active_fn=self.active_fn))
            else:
                if hidden_dim != self.input_dim:
                    raise RuntimeError('uncomment this for search_first model')
                # layers.append(Narrow(1, narrow_start, hidden_dim))
                narrow_start += hidden_dim
            layers.extend([
                # dw
                ConvBNReLU(hidden_dim,
                           hidden_dim,
                           kernel_size=k,
                           stride=self.stride,
                           groups=hidden_dim,
                           batch_norm_kwargs=_batch_norm_kwargs,
                           active_fn=self.active_fn),
                # pw-linear
                nn.Conv2d(hidden_dim, self.output_dim, 1, 1, 0, bias=False),
                # nn.BatchNorm2d(oup, **batch_norm_kwargs),
            ])
            ops.append(nn.Sequential(*layers))
        pw_bn = nn.BatchNorm2d(self.output_dim, **_batch_norm_kwargs)

        if not expand and narrow_start != self.input_dim:
            raise ValueError('Part of input are not used')

        return ops, pw_bn


    def forward(self, x):
        if len(self.ops) == 0:
            if not self.use_res_connect:
                return self.residual(x)
            else:
                return x
        tmp = sum([op(x) for op in self.ops])
        tmp = self.pw_bn(tmp)
        if self.use_res_connect:
            return x + tmp
        else:
            return self.residual(x) + tmp
        return tmp


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 kernel_size=3,
                 active_fn=None,
                 batch_norm_kwargs=None,
                 expand_ratio=None,
                 kernel_sizes=None
                 ):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=0.1)
        self.downsample = None
        self.stride = stride
        if self.stride != 1 or inplanes != planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes,
                          planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes, momentum=0.1),
            )

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


def get_block_wrapper(block_str):
    """Wrapper for MobileNetV2 block.
    Use `expand_ratio` instead of manually specified channels number."""

    # return ConvBNReLU
    # return BasicBlock
    return InvertedResidualChannels

class ParallelModule(nn.Module):
    def __init__(self,
                 num_branches=2,
                 block=get_block_wrapper('InvertedResidualChannels'),
                 num_blocks=[2, 2],
                 num_channels=[32, 32],
                 expand_ratio=6,
                 kernel_sizes=[3, 5, 7],
                 batch_norm_kwargs=None,
                 active_fn=get_active_fn('nn.ReLU6')):
        super(ParallelModule, self).__init__()

        self.num_branches = num_branches
        self.active_fn = active_fn
        self.batch_norm_kwargs = batch_norm_kwargs
        self.expand_ratio = expand_ratio
        self.kernel_sizes = kernel_sizes

        self._check_branches(
            num_branches, num_blocks, num_channels)
        self.branches = self._make_branches(
            num_branches, block, num_blocks, num_channels)

    def _check_branches(self, num_branches, num_blocks, num_channels):
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(
                num_branches, len(num_blocks))
            logging.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(
                num_branches, len(num_channels))
            logging.error(error_msg)
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels):
        layers = []
        for i in range(0, num_blocks[branch_index]):
            layers.append(
                block(
                    num_channels[branch_index],
                    num_channels[branch_index],
                    expand_ratio=self.expand_ratio,
                    kernel_sizes=self.kernel_sizes,
                    stride=1,
                    batch_norm_kwargs=self.batch_norm_kwargs,
                    active_fn=self.active_fn))
        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []
        for i in range(num_branches):
            branches.append(
                self._make_one_branch(i, block, num_blocks, num_channels))
        return nn.ModuleList(branches)

    def forward(self, x):
        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])
        return x


class FuseModule(nn.Module):
    def __init__(self,
                 in_branches=1,
                 out_branches=2,
                 block=get_block_wrapper('InvertedResidualChannels'),
                 in_channels=[16],
                 out_channels=[16, 32],
                 expand_ratio=6,
                 kernel_sizes=[3, 5, 7],
                 batch_norm_kwargs=None,
                 active_fn=get_active_fn('nn.ReLU6')):
        super(FuseModule, self).__init__()

        self.out_branches = out_branches
        self.in_branches = in_branches
        self.active_fn = active_fn
        self.batch_norm_kwargs = batch_norm_kwargs
        self.expand_ratio = expand_ratio
        self.kernel_sizes = kernel_sizes

        self.relu = nn.ReLU(False)

        fuse_layers = []
        for i in range(out_branches):
            fuse_layer = []
            for j in range(in_branches):
                if j > i:
                    fuse_layer.append(nn.Sequential(
                        block(
                            in_channels[j],
                            out_channels[i],
                            expand_ratio=self.expand_ratio,
                            kernel_sizes=self.kernel_sizes,
                            stride=1,
                            batch_norm_kwargs=self.batch_norm_kwargs,
                            active_fn=self.active_fn),
                        nn.Upsample(scale_factor=2 ** (j - i), mode='nearest')))
                elif j == i:
                    fuse_layer.append(
                        block(
                            in_channels[j],
                            out_channels[i],
                            expand_ratio=self.expand_ratio,
                            kernel_sizes=self.kernel_sizes,
                            stride=1,
                            batch_norm_kwargs=self.batch_norm_kwargs,
                            active_fn=self.active_fn))
                else:
                    downsamples = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            downsamples.append(
                                block(
                                    out_channels[j + k],
                                    out_channels[i],
                                    expand_ratio=self.expand_ratio,
                                    kernel_sizes=self.kernel_sizes,
                                    stride=2,
                                    batch_norm_kwargs=self.batch_norm_kwargs,
                                    active_fn=self.active_fn))
                        elif k == 0:
                            downsamples.append(
                                block(
                                    in_channels[j],
                                    out_channels[j + 1],
                                    expand_ratio=self.expand_ratio,
                                    kernel_sizes=self.kernel_sizes,
                                    stride=2,
                                    batch_norm_kwargs=self.batch_norm_kwargs,
                                    active_fn=self.active_fn))
                        else:
                            downsamples.append(
                                block(
                                    out_channels[j + k],
                                    out_channels[j + k + 1],
                                    expand_ratio=self.expand_ratio,
                                    kernel_sizes=self.kernel_sizes,
                                    stride=2,
                                    batch_norm_kwargs=self.batch_norm_kwargs,
                                    active_fn=self.active_fn))
                    fuse_layer.append(nn.Sequential(*downsamples))
            fuse_layers.append(nn.ModuleList(fuse_layer))
        self.fuse_layers = nn.ModuleList(fuse_layers)

    def forward(self, x):
        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = self.fuse_layers[i][0](x[0])
            for j in range(1, self.in_branches):
                y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))  # TODO(Mingyu): Use ReLU?
        return x_fuse


class HeadModule(nn.Module):
    def __init__(self,
                 pre_stage_channels=[16, 32, 64, 128],
                 head_channels=None,  # [32, 64, 128, 256],
                 last_channel=1024,
                 avg_pool_size=7,
                 block=get_block_wrapper('InvertedResidualChannels'),
                 expand_ratio=6,
                 kernel_sizes=[3, 5, 7],
                 batch_norm_kwargs=None,
                 active_fn=get_active_fn('nn.ReLU6')):
        super(HeadModule, self).__init__()

        self.active_fn = active_fn
        self.batch_norm_kwargs = batch_norm_kwargs
        self.expand_ratio = expand_ratio
        self.kernel_sizes = kernel_sizes
        self.avg_pool_size = avg_pool_size

        # Increasing the #channels on each resolution
        if head_channels:
            incre_modules = []
            for i, channels in enumerate(pre_stage_channels):
                incre_module = block(
                    pre_stage_channels[i],
                    head_channels[i],
                    expand_ratio=self.expand_ratio,
                    kernel_sizes=self.kernel_sizes,
                    stride=1,
                    batch_norm_kwargs=self.batch_norm_kwargs,
                    active_fn=self.active_fn)
                incre_modules.append(incre_module)
            self.incre_modules = nn.ModuleList(incre_modules)
        else:
            head_channels = pre_stage_channels
            self.incre_modules = []

        # downsampling modules
        downsamp_modules = []
        for i in range(len(pre_stage_channels) - 1):
            downsamp_module = block(
                head_channels[i],
                head_channels[i + 1],
                expand_ratio=self.expand_ratio,
                kernel_sizes=self.kernel_sizes,
                stride=2,
                batch_norm_kwargs=self.batch_norm_kwargs,
                active_fn=self.active_fn)
            downsamp_modules.append(downsamp_module)
        self.downsamp_modules = nn.ModuleList(downsamp_modules)

        self.final_layer = ConvBNReLU(
            head_channels[-1],
            last_channel,
            kernel_size=1,
            batch_norm_kwargs=batch_norm_kwargs,
            active_fn=active_fn)

    def forward(self, x_list):
        if self.incre_modules:
            x = self.incre_modules[0](x_list[0])
            for i in range(len(self.downsamp_modules)):
                x = self.incre_modules[i + 1](x_list[i + 1]) \
                    + self.downsamp_modules[i](x)
        else:
            x = x_list[0]
            for i in range(len(self.downsamp_modules)):
                x = x_list[i + 1] \
                    + self.downsamp_modules[i](x)

        x = self.final_layer(x)

        # assert x.size()[2] == self.avg_pool_size

        # if torch._C._get_tracing_state():
        #     x = x.flatten(start_dim=2).mean(dim=2)
        # else:
        #     x = F.avg_pool2d(x, kernel_size=x.size()
        #                      [2:]).view(x.size(0), -1)
        return x


@BACKBONES.register_module()
class HighResolutionNetAtom(nn.Module):

    def __init__(self,
                 num_classes=1000,
                 input_size=224,
                 input_stride=4,
                 input_channel=16,
                 last_channel=1024,
                 head_channels=None,
                 bn_momentum=0.1,
                 bn_epsilon=1e-5,
                 dropout_ratio=0.2,
                 active_fn='nn.ReLU6',
                 block='InvertedResidualChannels',
                 width_mult=1.0,
                 round_nearest=8,
                 expand_ratio=4,
                 kernel_sizes=[3, 5, 7],
                 inverted_residual_setting=None,
                 ** kwargs):
        super(HighResolutionNetAtom, self).__init__()


        batch_norm_kwargs = {
            'momentum': bn_momentum,
            'eps': bn_epsilon
        }

        self.avg_pool_size = input_size // 32
        self.input_stride = input_stride
        self.input_channel = input_channel
        self.last_channel = last_channel
        self.batch_norm_kwargs = batch_norm_kwargs
        self.active_fn = get_active_fn(active_fn)
        self.kernel_sizes = kernel_sizes
        self.expand_ratio = expand_ratio

        self.block = get_block_wrapper(block)
        self.inverted_residual_setting = inverted_residual_setting

        downsamples = []
        if self.input_stride > 1:
            downsamples.append(ConvBNReLU(
                3,
                self.input_channel,
                kernel_size=3,
                stride=2,
                batch_norm_kwargs=self.batch_norm_kwargs,
                active_fn=self.active_fn))
        if self.input_stride > 2:
            downsamples.append(ConvBNReLU(
                self.input_channel,
                self.input_channel,
                kernel_size=3,
                stride=2,
                batch_norm_kwargs=self.batch_norm_kwargs,
                active_fn=self.active_fn))
        self.downsamples = nn.Sequential(*downsamples)

        features = []
        for index in range(len(inverted_residual_setting) - 1):
            features.append(
                ParallelModule(
                    num_branches=inverted_residual_setting[index][0],
                    num_blocks=inverted_residual_setting[index][1],
                    num_channels=inverted_residual_setting[index][2],
                    block=self.block,
                    expand_ratio=self.expand_ratio,
                    kernel_sizes=self.kernel_sizes,
                    batch_norm_kwargs=self.batch_norm_kwargs,
                    active_fn=self.active_fn)
            )
            features.append(
                FuseModule(
                    in_branches=inverted_residual_setting[index][0],
                    out_branches=inverted_residual_setting[index + 1][0],
                    in_channels=inverted_residual_setting[index][-1],
                    out_channels=inverted_residual_setting[index + 1][-1],
                    block=self.block,
                    expand_ratio=self.expand_ratio,
                    kernel_sizes=self.kernel_sizes,
                    batch_norm_kwargs=self.batch_norm_kwargs,
                    active_fn=self.active_fn)
            )

        for index in range(len(inverted_residual_setting) - 1, len(inverted_residual_setting)):
            features.append(
                ParallelModule(
                    num_branches=inverted_residual_setting[index][0],
                    num_blocks=inverted_residual_setting[index][1],
                    num_channels=inverted_residual_setting[index][2],
                    block=self.block,
                    expand_ratio=self.expand_ratio,
                    kernel_sizes=self.kernel_sizes,
                    batch_norm_kwargs=self.batch_norm_kwargs,
                    active_fn=self.active_fn)
            )

        features.append(HeadModule(
            pre_stage_channels=inverted_residual_setting[-1][2],
            head_channels=head_channels,
            last_channel=self.last_channel,
            avg_pool_size=self.avg_pool_size,
            block=self.block,
            expand_ratio=self.expand_ratio,
            kernel_sizes=self.kernel_sizes,
            batch_norm_kwargs=self.batch_norm_kwargs,
            active_fn=self.active_fn))

        self.features = nn.Sequential(*features)

        # self.classifier = nn.Sequential(
        #     nn.Dropout(dropout_ratio),
        #     nn.Linear(last_channel, num_classes),
        # )

        self.init_weights()

    def init_weights(self, pretrained=None):
        logging.info('=> init weights from normal distribution')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.downsamples(x)
        x = self.features([x])
        # x = self.classifier(x)
        return x


# Model = HighResolutionNet