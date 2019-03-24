import mxnet as mx
from mxnet import gluon
from mxnet.gluon import nn as gnn

class ResBlock(gluon.HybridBlock):
    def __init__(self, channels, stride=1, **kwargs):
        super(ResBlock, self).__init__(**kwargs)
        self.stem = gnn.HybridSequential()
        self.stem.add(gnn.Conv2D(channels, 3, padding=1, use_bias=False))
        self.stem.add(gnn.BatchNorm())
        self.stem.add(gnn.Activation('relu'))
        self.stem.add(gnn.Conv2D(channels, 3, strides=stride, padding=1, use_bias=False))
        self.stem.add(gnn.BatchNorm())
        self.relu = gnn.Activation('relu')
        if stride != 1:
            self.downsample = gnn.Conv2D(channels, 1, strides=stride)
        else:
            self.downsample = None

    def hybrid_forward(self, F, x):
        res = self.downsample(x) if self.downsample else x
        return self.relu(res + self.stem(x))

class Bottleneck(gluon.HybridBlock):
    def __init__(self, channels, stride=1, **kwargs):
        super(Bottleneck, self).__init__(**kwargs)
        self.stem = gnn.HybridSequential()
        self.stem.add(gnn.Conv2D(channels//4, 1, use_bias=False))
        self.stem.add(gnn.BatchNorm())
        self.stem.add(gnn.Activation('relu'))
        self.stem.add(gnn.Conv2D(channels//4, 3, strides=stride, padding=1, use_bias=False))
        self.stem.add(gnn.BatchNorm())
        self.stem.add(gnn.Activation('relu'))
        self.stem.add(gnn.Conv2D(channels, 1, use_bias=False))
        self.stem.add(gnn.BatchNorm())
        self.relu = gnn.Activation('relu')
        if stride != 1:
            self.downsample = gnn.Conv2D(channels, 1, strides=stride)
        else:
            self.downsample = None

    def hybrid_forward(self, F, x):
        res = self.downsample(x) if self.downsample else x
        return self.relu(res + self.stem(x))

class DWiseBlock(gluon.HybridBlock):
    def __init__(self, channels, stride=1, spatial=False, **kwargs):
        super(DWiseBlock, self).__init__(**kwargs)
        self.stem = gnn.HybridSequential()
        if spatial:
            self.stem.add(gnn.Conv2D(channels, 3, strides=(stride, 1),
                                     padding=1, groups=channels, use_bias=False))
            self.stem.add(gnn.Conv2D(channels, 3, strides=(1, stride),
                                     padding=1, groups=channels, use_bias=False))
        else:
            self.stem.add(gnn.Conv2D(channels, 3, strides=stride,
                                     padding=1, groups=channels, use_bias=False))
        self.stem.add(gnn.BatchNorm())
        self.stem.add(gnn.Activation('relu'))
        self.stem.add(gnn.Conv2D(channels, 1, use_bias=False))
        self.stem.add(gnn.BatchNorm())
        self.stem.add(gnn.Activation('relu'))

    def hybrid_forward(self, F, x):
        return self.stem(x)

class InvBottleneck(gluon.HybridBlock):
    '''param t: expansion factor'''
    def __init__(self, channels, t=2, stride=1, spatial=False, **kwargs):
        super(Bottleneck, self).__init__(**kwargs)
        self.stem = gnn.HybridSequential()
        self.stem.add(gnn.Conv2D(channels*t, 1, use_bias=False))
        self.stem.add(gnn.BatchNorm())
        self.stem.add(gnn.Activation('relu'))
        if spatial:
            self.stem.add(gnn.Conv2D(channels*t, 3, strides=(stride, 1),
                                     padding=1, groups=channels, use_bias=False))
            self.stem.add(gnn.Conv2D(channels*t, 3, strides=(1, stride),
                                     padding=1, groups=channels, use_bias=False))
        else:
            self.stem.add(gnn.Conv2D(channels, 3, strides=stride,
                                     padding=1, groups=channels, use_bias=False))
        self.stem.add(gnn.BatchNorm())
        self.stem.add(gnn.Activation('relu'))
        self.stem.add(gnn.Conv2D(channels, 1, use_bias=False))
        self.stem.add(gnn.BatchNorm())
        self.relu = gnn.Activation('relu')
        if stride != 1:
            self.downsample = gnn.Conv2D(channels, 1, strides=stride)
        else:
            self.downsample = None

    def hybrid_forward(self, F, x):
        res = self.downsample(x) if self.downsample else x
        return self.relu(res + self.stem(x))

