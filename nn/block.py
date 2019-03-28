import mxnet as mx
from mxnet import gluon
from mxnet.gluon import nn as gnn

__all__ = ['conv_bn', 'CReLU', 'ResBlock', 'Bottleneck', 'DWiseBlock', 'InvBottleneck']

class ChannelShuffle(gluon.HybridBlock):
    """Used in shuffle block. ref https://arxiv.org/pdf/1707.01083"""
    def __init__(self, groups, **kwargs):
        super(ChannelShuffle, self).__init__(**kwargs)
        self._groups = groups

    def hybrid_forward(self, F, x):
        x = F.reshape(x, shape=(0, -4, self._groups, -1, -2))
        x = F.swapaxes(x, 1, 2)
        return F.reshape(x, shape=(0, -3, -2))

def _to_list(param):
    if isinstance(param, int):
        return [param, param]
    else:
        return param

def conv_bn(channels, kernel, stride=1, pad=0, group=1, act=None, spatial=False):
    layers = []
    if spatial:
        kernel, stride, pad = [_to_list(x) for x in [kernel, stride, pad]]
        layers.append(gnn.Conv2D(channels, (kernel[0],1), (stride[0],1), (pad[0],0), groups=group, use_bias=False))
        layers.append(gnn.Conv2D(channels, (1,kernel[1]), (1,stride[1]), (0,pad[1]), groups=group, use_bias=False))
    else:
        layers.append(gnn.Conv2D(channels, kernel, stride, pad, groups=group, use_bias=False))
    layers.append(gnn.BatchNorm())
    if act:
        layers.append(gnn.Activation(act))
    return layers

class CReLU(gluon.HybridBlock):
    """Concatenated ReLU: https://arxiv.org/pdf/1603.05201"""
    def __init__(self, channels, kernel, stride=1, pad=0, **kwargs):
        super(CReLU, self).__init__(**kwargs)
        assert channels % 2 == 0
        self.stem = gnn.HybridSequential()
        self.stem.add(*conv_bn(channels//2, kernel, stride, pad))
    
    def hybrid_forward(self, F, x):
        x = self.stem(x)
        return F.relu(F.concat(x, -x, dim=1))

class ResBlock(gluon.HybridBlock):
    def __init__(self, channels, stride=1, **kwargs):
        super(ResBlock, self).__init__(**kwargs)
        self.stem = gnn.HybridSequential()
        self.stem.add(
            *conv_bn(channels, 3, pad=1, act='relu'),
            *conv_bn(channels, 3, stride=stride, pad=1))
        if stride != 1:
            self.downsample = gnn.Conv2D(channels, 1, strides=stride)
        else:
            self.downsample = None

    def hybrid_forward(self, F, x):
        res = self.downsample(x) if self.downsample else x
        return F.relu(res + self.stem(x))

class Bottleneck(gluon.HybridBlock):
    def __init__(self, channels, stride=1, **kwargs):
        super(Bottleneck, self).__init__(**kwargs)
        self.stem = gnn.HybridSequential()
        self.stem.add(
            *conv_bn(channels//4, 1, act='relu'),
            *conv_bn(channels//4, 3, stride=stride, pad=1, act='relu'),
            *conv_bn(channels, 1))
        if stride != 1:
            self.downsample = gnn.Conv2D(channels, 1, strides=stride)
        else:
            self.downsample = None

    def hybrid_forward(self, F, x):
        res = self.downsample(x) if self.downsample else x
        return F.relu(res + self.stem(x))

class DWiseBlock(gluon.HybridBlock):
    def __init__(self, channels, in_channels, stride=1, spatial=False, shuffle=False, **kwargs):
        super(DWiseBlock, self).__init__(**kwargs)
        self.stem = gnn.HybridSequential()
        self.stem.add(*conv_bn(in_channels, 3, stride, group=in_channels, act='relu', spatial=spatial))
        if shuffle and in_channels >= 8:
            groups = in_channels // 4
            self.stem.add(
                ChannelShuffle(groups),
                *conv_bn(channels, 1, group=groups, act='relu'))
        else:
            self.stem.add(*conv_bn(channels, 1, act='relu'))

    def hybrid_forward(self, F, x):
        return self.stem(x)

class InvBottleneck(gluon.HybridBlock):
    '''param t: expansion factor'''
    def __init__(self, channels, t=2, stride=1, spatial=False, **kwargs):
        super(Bottleneck, self).__init__(**kwargs)
        self.stem = gnn.HybridSequential()
        self.stem.add(
            *conv_bn(channels*t, 1, act='relu'),
            *conv_bn(channels*t, 3, stride, 1, group=channels, act='relu', spatial=spatial),
            *conv_bn(channels, 1)
        )
        if stride != 1:
            self.downsample = gnn.Conv2D(channels, 1, strides=stride)
        else:
            self.downsample = None

    def hybrid_forward(self, F, x):
        res = self.downsample(x) if self.downsample else x
        return F.relu(res + self.stem(x))
