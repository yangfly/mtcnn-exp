import mxnet as mx
from mxnet import nd

__all__ = ['Accuracy', 'LocMSE', 'metric_msg']

class Accuracy(mx.metric.EvalMetric):
    def __init__(self, pos_thresh = 0.65, neg_thresh = 0.3):
        super(Accuracy, self).__init__('Accuracy')
        self._pos_thresh = pos_thresh
        self._neg_thresh = neg_thresh

    def update(self, labels, preds):
        labels = [l.as_in_context(mx.cpu()) for l in labels]
        preds = [p.as_in_context(mx.cpu()) for p in preds]
        label = nd.concat(*labels, dim=0)
        pred = nd.concat(*preds, dim=0)
        if pred.ndim > 2:
            pred = pred.reshape(0, -1)
        pred_label = nd.argmax_channel(pred)
        pos_mask = label[:,0] >= self._pos_thresh
        neg_mask = label[:,0] < self._neg_thresh
        mask = nd.logical_or(pos_mask, neg_mask)
        right = nd.logical_and(pos_mask == pred_label, mask)
        self.sum_metric += nd.sum(right).asscalar()
        self.num_inst += nd.sum(mask).asscalar()

class LocMSE(mx.metric.EvalMetric):
    def __init__(self, part_thresh = 0.4):
        super(LocMSE, self).__init__('LocMSE')
        self._part_thresh = part_thresh

    def update(self, labels, preds):
        labels = [l.as_in_context(mx.cpu()) for l in labels]
        preds = [p.as_in_context(mx.cpu()) for p in preds]
        label = nd.concat(*labels, dim=0)
        pred = nd.concat(*preds, dim=0)
        if pred.ndim > 2:
            pred = pred.reshape(0, -1)
        mask = label[:,0] >= self._part_thresh
        loss = nd.square(nd.abs(pred, label[:,1:]))
        loss = nd.where(mask, loss, nd.zeros_like(loss))
        self.sum_metric += nd.sum(loss).asscalar()
        self.num_inst += nd.sum(mask).asscalar()

def metric_msg(*args):
    msg = ''
    for metric in args:
        msg += ', {}={:.3f}'.format(*metric.get())
    return msg
