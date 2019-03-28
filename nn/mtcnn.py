"""Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks"""

import mxnet as mx
from mxnet import gluon
from mxnet.gluon import nn as gnn

__all__ = ['Pnet', 'Rnet', 'Onet', 'Mtcnn']

class Pnet(gluon.HybridBlock):
    """Proposal Network"""
    def __init__(self, size=12, **kwargs):
        super(Pnet, self).__init__(**kwargs)
        self.size = size
        self.base = gnn.HybridSequential()
        self.base.add(
            gnn.Conv2D(10, 3), gnn.PReLU(),
            gnn.MaxPool2D(ceil_mode=True), # caffe default
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


class Rnet(gluon.HybridBlock):
    """Refine Network"""
    def __init__(self, size=24, **kwargs):
        super(Rnet, self).__init__(**kwargs)
        self.size = size
        self.base = gnn.HybridSequential()
        self.base.add(
            gnn.Conv2D(28, 3), gnn.PReLU(),
            gnn.MaxPool2D(3, 2, ceil_mode=True),
            gnn.Conv2D(48, 3), gnn.PReLU(),
            gnn.MaxPool2D(3, 2, ceil_mode=True),
            gnn.Conv2D(64, 2), gnn.PReLU(),
            gnn.Dense(128), gnn.PReLU())
        self.fc5_1 = gnn.Dense(2)
        self.fc5_2 = gnn.Dense(4)

    def hybrid_forward(self, F, x):
        x = self.base(x)
        cls_pred = self.fc5_1(x)
        bbx_pred = self.fc5_2(x)
        if not mx.autograd.is_training():
            cls_pred = F.softmax(cls_pred, axis=1)
        return cls_pred, bbx_pred


class Onet(gluon.HybridBlock):
    """Output Network"""
    def __init__(self, size=48, **kwargs):
        super(Onet, self).__init__(**kwargs)
        self.size = size
        self.base = gnn.HybridSequential()
        self.base.add(
            gnn.Conv2D(32, 3), gnn.PReLU(),
            gnn.MaxPool2D(3, 2, ceil_mode=True),
            gnn.Conv2D(64, 3), gnn.PReLU(),
            gnn.MaxPool2D(3, 2, ceil_mode=True),
            gnn.Conv2D(64, 3), gnn.PReLU(),
            gnn.MaxPool2D(2, 2, ceil_mode=True),
            gnn.Conv2D(128, 2), gnn.PReLU(),
            gnn.Dense(256), gnn.PReLU())
        self.fc6_1 = gnn.Dense(2)
        self.fc6_2 = gnn.Dense(4)

    def hybrid_forward(self, F, x):
        x = self.base(x)
        cls_pred = self.fc6_1(x)
        bbx_pred = self.fc6_2(x)
        if not mx.autograd.is_training():
            cls_pred = F.softmax(cls_pred, axis=1)
        return cls_pred, bbx_pred


import cv2 as cv
import numpy as np
import os.path as osp
from math import ceil
import utils

class Mtcnn(object):
    """MTCNN face detector.
    
    Parameters
    ----------
    model_dir : str, required
        Path to mxnet model
    min_face : int, default is 20
        Minimum size of face to be detected.
    scale_factor : float, default is 0.709
        Scale factor for image pyramid in Pnet.
    thresholds : list of float, default is [0.6, 0.7, 0.7]
        Thresholds for Pnet, Rnet and Onet.
    ctx : mx.context.Context, default is mx.cpu()
        Device context to run detector.
    """
    def __init__(self, model_dir, min_face=20, scale_factor=0.709,
      thresholds=[0.6, 0.7, 0.7], ctx=mx.cpu()):
        self._min_face = min_face
        self._scale_factor = scale_factor
        self._thresholds = thresholds
        # load models
        self._nets = []
        for name, size, _ in zip(['pnet', 'rnet', 'onet'], [12, 24, 48], thresholds):
            self._nets.append(utils.load_model(osp.join(model_dir, name), 0, size, ctx))
    
    def detect(self, image):
        """Detect faces with Pnet, Rnet and Onet.
        
        Parameters
        ----------
        image : numpy.ndarray, h x w x c [RGB]
        
        Example:
        ```
        mtcnn = Mtcnn("models/caffe", min_face=40)
        image = cv.imread('data/test.jpg', cv.IMREAD_COLOR)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        bbox = mtcnn.detect(image)
        utils.plot_bbox(image, bbox)
        ```

        Returns:
        -------
            numpy.ndarray, n x 5 [score, x1, y1, x2, y2]
        """
        im = (image.astype(np.float32) - 127.5) * 0.0078125 # normalize

        ## Pnet
        min_len = min(im.shape[:2])
        scales = utils.scale_pyramid(min_len, self._min_face, self._scale_factor)
        total_bbox = []
        for scale in scales:
            width = int(ceil(im.shape[1] * scale))
            height = int(ceil(im.shape[0] * scale))
            img = cv.resize(im, (width, height))
            img = img.transpose((2, 0, 1))[None, ...] # HWC -> 1CHW
            self._nets[0].forward(mx.io.DataBatch([mx.nd.array(img)]))
            bbox_pred, cls_pred = self._nets[0].get_outputs()
            bbox = utils.generate_bbox(cls_pred, bbox_pred, scale, self._thresholds[0])
            if bbox.size > 0:
                bbox = bbox[utils.bbox_nms(bbox, 0.5, 'union')]
                total_bbox.append(bbox)
        if len(total_bbox) == 0:
            return np.array([])
        total_bbox = np.vstack(total_bbox)
        total_bbox = total_bbox[utils.bbox_nms(total_bbox, 0.7, 'union')]
        total_bbox = utils.bbox_refine(total_bbox)
        if len(self._nets) == 1:
            return total_bbox

        ## Rnet
        data = utils.crop_pad(im, total_bbox, 24)
        data_iter = mx.io.NDArrayIter(data, batch_size=min(len(data), 100))
        cls_pred = []
        bbox_pred = []
        for batch in data_iter:
            self._nets[1].forward(batch)
            cls_pred.append(self._nets[1].get_outputs()[1].asnumpy())
            bbox_pred.append(self._nets[1].get_outputs()[0].asnumpy())
        num = len(total_bbox)
        cls_pred = np.vstack(cls_pred)[:num]
        bbox_pred = np.vstack(bbox_pred)[:num]
        total_bbox[:, 0] = cls_pred[:, 1]
        total_bbox = np.hstack((total_bbox, bbox_pred))
        keep = np.where(total_bbox[:,0] >= self._thresholds[1])[0]
        if len(keep) == 0:
            return np.array([])
        total_bbox = total_bbox[keep]
        total_bbox = total_bbox[utils.bbox_nms(total_bbox, 0.7, 'union')]
        total_bbox = utils.bbox_refine(total_bbox)
        if len(self._nets) == 2:
            return total_bbox
        
        ## onet
        data = utils.crop_pad(im, total_bbox, 48)
        data_iter = mx.io.NDArrayIter(data, batch_size=min(len(data), 20))
        cls_pred = []
        bbox_pred = []
        for batch in data_iter:
            self._nets[2].forward(batch)
            cls_pred.append(self._nets[2].get_outputs()[1].asnumpy())
            bbox_pred.append(self._nets[2].get_outputs()[0].asnumpy())
        num = len(total_bbox)
        cls_pred = np.vstack(cls_pred)[:num]
        bbox_pred = np.vstack(bbox_pred)[:num]
        total_bbox[:, 0] = cls_pred[:, 1]
        total_bbox = np.hstack((total_bbox, bbox_pred))
        keep = np.where(total_bbox[:,0] >= self._thresholds[1])[0]
        if len(keep) == 0:
            return np.array([])
        total_bbox = total_bbox[keep]
        total_bbox = utils.bbox_refine(total_bbox)
        total_bbox = total_bbox[utils.bbox_nms(total_bbox, 0.7, 'min')]
        return total_bbox
        
if __name__ == '__main__':
    for net, size in zip([Pnet(), Rnet(), Onet()], [12, 24, 48]):
        # print(net)
        net.hybridize()
        x = mx.sym.var('data')
        sym = net(x)
        print('')
        mx.viz.print_summary(sym[0], {'data': (1, 3, size, size)}, line_length=100)
    