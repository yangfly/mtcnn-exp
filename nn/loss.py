import mxnet as mx
from mxnet import nd
from mxnet import gluon

__all__ = ['MtcnnLoss']

class MtcnnLoss(gluon.Block):
    """Detect Loss (softmax + smoothl1/euclid) with online hard example mining.

    Parameters
    ----------
    pos_thresh : float, default 0.65
        IoU above pos_thresh assigned as positive face.
    part_thresh : float, default 0.4
        Iou between part_thresh and pos_thresh assigned part face.
    neg_thresh : float, default 0.3
        IoU less than neg_thresh assigned negative face.
    ohem_ratio : float, default 0.7
        The ratio cls_loss to been kept after online hard example mining.
    cls_weight : float, default 1.0
        Scalar weight for ohem softmax loss.
    loc_weight : float, default 1.0
        Scalar weight for L2 loss.
    loc_loss : str, default 'smoothl1'
        Loss used for bounding box localization, optional ['smoothl1', 'euclid']
    """
    def __init__(self, pos_thresh=0.65, part_thresh=0.4, neg_thresh=0.3,
                 ohem_ratio=0.7, cls_weight=1.0, loc_weight=1.0, loc_loss='smoothl1', **kwargs):
        super(MtcnnLoss, self).__init__(**kwargs)
        self._pos_thresh = pos_thresh
        self._part_thresh = part_thresh
        self._neg_thresh = neg_thresh
        self._ohem_ratio = ohem_ratio
        self._cls_weight = cls_weight
        self._loc_weight = loc_weight
        if loc_loss == 'smoothl1':
            self._loc_loss = self._smoothl1_loss
        elif loc_loss == 'euclid':
            self._loc_loss = self._euclid_loss
        else:
            raise ValueError('Unsupported loc loss: {}'.format(loc_loss))
     
    def forward(self, cls_preds, loc_preds, labels):
        # transport to cpu
        cls_preds = [p.as_in_context(mx.cpu()) for p in cls_preds]
        loc_preds = [p.as_in_context(mx.cpu()) for p in loc_preds]
        labels = [l.as_in_context(mx.cpu()) for l in labels]
        cls_pred = nd.concat(*cls_preds, dim=0)
        loc_pred = nd.concat(*loc_preds, dim=0)
        label = nd.concat(*labels, dim=0)
        cls_loss = self._softmax_loss(cls_pred, label)
        loc_loss = self._loc_loss(loc_pred, label)
        sum_loss = self._cls_weight * cls_loss + self._loc_weight * loc_loss
        cls_losses = self._split_like(cls_loss, labels)
        loc_losses = self._split_like(loc_loss, labels)
        sum_losses = self._split_like(sum_loss, labels)
        return sum_losses, cls_losses, loc_losses

    def _softmax_loss(self, pred, label):
        if pred.ndim > 2:
            pred = pred.reshape(0, -1)
        pos_mask = label[:,0] >= self._pos_thresh
        neg_mask = label[:,0] < self._neg_thresh
        mask = nd.logical_or(pos_mask, neg_mask)
        pred = nd.log_softmax(pred)
        label = nd.where(pos_mask, nd.ones_like(mask), nd.zeros_like(mask))
        loss = -nd.pick(pred, label, axis=1, keepdims=False)
        loss = nd.where(mask, loss, nd.zeros_like(loss))
        keep_num = round(self._ohem_ratio * nd.sum(mask).asscalar())
        keep_mask = loss.argsort(axis=0, is_ascend=False).argsort(axis=0) < keep_num
        loss = nd.where(keep_mask, loss, nd.zeros_like(loss))
        return loss / max(keep_num, 1)
        
    def _smoothl1_loss(self, pred, label, ratio=1.0):
        if pred.ndim > 2:
            pred = pred.reshape(0, -1)
        mask = label[:,0] >= self._part_thresh
        loss = nd.abs(pred - label[:,1:])
        loss = nd.where(loss > ratio, loss - 0.5 * ratio, (0.5 / ratio) * nd.square(loss))
        loss = nd.mean(loss, axis=1)
        loss = nd.where(mask, loss, nd.zeros_like(loss))
        return loss / max(nd.sum(mask).asscalar(), 1)
    
    def _euclid_loss(self, pred, label):
        if pred.ndim > 2:
            pred = pred.reshape(0, -1)
        mask = label[:,0] >= self._part_thresh
        loss = nd.square(pred - label[:,1:]) / 2
        loss = nd.mean(loss, axis=1)
        loss = nd.where(mask, loss, nd.zeros_like(loss))
        return loss / max(nd.sum(mask).asscalar(), 1)
    
    def _split_like(self, loss, labels):
        losses = []
        start_idx = 0
        for label in labels:
            num = label.shape[0]
            losses.append(loss[start_idx:start_idx+num])
            start_idx += num
        return losses
    
    def test(self, batch_size=87, num_gpu=4):
        '''Test for multi-gpu training.'''
        from math import ceil
        gpu_batch_size = ceil(float(batch_size) / num_gpu)
        gpu_bs = [gpu_batch_size] * num_gpu
        gpu_bs[-1] = batch_size - gpu_batch_size * (num_gpu - 1)
        cls_preds = []
        loc_preds = []
        labels = []
        for bs in gpu_bs:
            cls_preds.append(nd.random.randn(bs, 2, 1, 1))
            loc_preds.append(nd.random.randn(bs, 4, 1, 1))
            labels.append(nd.random.uniform(0, 1, (bs, 5)))
        sum_losses, cls_losses, loc_losses = self.forward(cls_preds, loc_preds, labels)
        print('sum ratio: {}'.format(sum([nd.sum(loss > 0).asscalar() for loss in sum_losses]) / batch_size))
        for i in range(num_gpu):
            print('------------- on gpu {} ---------------'.format(i))
            print(sum_losses[i])
            print(cls_losses[i])
            print(loc_losses[i])

if __name__ == '__main__':
    mloss = MtcnnLoss(loc_loss='euclid')
    mloss.test()