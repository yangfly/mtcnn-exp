import mxnet as mx
from mxnet import gluon
from mxnet.gluon import nn as gnn
import nn

__all__ = ['get_net', 'XPnet']

class XPnet(gluon.HybridBlock):
    def __init__(self, size=12, pool=True, crelu=False, dwconv=False, spatial=False, shuffle=False, **kwargs):
        super(DPnet, self).__init__(**kwargs)
        self.size = size
        self.base = gnn.HybridSequential()
        if pool:
            if crelu:
                self.base.add(nn.CReLU(10, 3), gnn.MaxPool2D(ceil_mode=True))
            else:
                self.base.add(gnn.Conv2D(10, 3), gnn.PReLU(), gnn.MaxPool2D(ceil_mode=True))
        else:
            if crelu:
                self.base.add(nn.CReLU(10, 3, 2))
            else:
                self.base.add(gnn.Conv2D(10, 3, 2), gnn.PReLU())
        if dwconv or shuffle:
            self.base.add(
                nn.DWiseBlock(16, 10, 1, spatial, shuffle),
                nn.DWiseBlock(32, 16, 1, spatial, shuffle))
        else:
            self.base.add(
                gnn.Conv2D(16, 3), gnn.PReLU(),
                gnn.Conv2D(32, 3), gnn.PReLU())
        self.conv4_1 = gnn.Conv2D(2, 1)
        self.conv4_2 = gnn.Conv2D(4, 1)

    def hybrid_forward(self, F, x):
        x = self.base(x)
        cls_pred = self.conv4_1(x)
        bbx_pred = self.conv4_2(x)
        if not mx.autograd.is_training():
            cls_pred = F.softmax(cls_pred, axis=1)
        return cls_pred, bbx_pred

def get_net(network):
    if network == 'pnet':
        net = nn.Pnet()
    elif network == 'rnet':
        net = nn.Rnet()
    elif network == 'onet':
        net = nn.Onet()
    elif network == 'cpnet':
        net = nn.XPnet(pool=False)
    elif network == 'ppnet':
        net = nn.XPnet(crelu=True)
    elif network == 'dpnet':
        net = nn.XPnet(dwconv=True)
    elif network == 'dcpnet':
        net = nn.XPnet(12, pool=True, dwconv=True)
    elif network == 'dspnet':
        net = nn.XPnet(dwconv=True, spatial=True)
    elif network == 'spnet':
        net = nn.XPnet(shuffle=True)
    elif network == 'scpnet':
        net = nn.DPnet(pool=True, spatial=True)
    elif network == 'sspnet':
        net = nn.XPnet(shuffle=True, spatial=True)
    else:
        raise ValueError('Unsupported network in mtcnn: {}'.format(network))
    return net

if __name__ == '__main__':
    net = get_net('sspnet')
    net.hybridize()
    x = mx.sym.var('data')
    sym = net(x)
    mx.viz.print_summary(sym[0], {'data': (1, 3, 12, 12)}, line_length=100)