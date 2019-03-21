import cv2 as cv
import numpy as np

__all__ = ['scale_pyramid', 'generate_bbox', 'bbox_nms', 'bbox_refine', 'crop_pad', 'bbox_iou']

def scale_pyramid(min_len, min_face, scale_factor):
    """Get scales for image pyramid."""
    min_scale = 12.0 / min_len
    scale = 12.0 / min_face
    scales = []
    while scale > min_scale:
        scales.append(scale)
        scale *= scale_factor
    return scales

def generate_bbox(cls_pred, bbox_pred, scale, threshold):
    """Generate bbox from predicted cls and bbox map.
    
    Parameters
    ----------
    cls_pred : mxnet.nd.ndarray, 1 x 2 x h x w
    bbox_pred : mxnet.nd.ndarray, 1 x 4 x h x w
    scale: float, scale of this forward
    threshold : float, thresh to filter bbox

    Returns:
    -------
        numpy.ndarray, n x 9 [score, x1, y1, x2, y2, dx1, dy1, dx2, dy2]
    """
    cls_pred = cls_pred.asnumpy()[0, 1] # h x w
    bbox_pred = bbox_pred.asnumpy()[0]  # 4 x h x w
    stride = 2
    cell_size = 12
    inv_scale = 1.0 / scale

    keep = np.where(cls_pred >= threshold)
    score = cls_pred[keep[0], keep[1]]
    # find nothing
    if score.size == 0:
        return np.array([], dtype=np.float32)
    reg = bbox_pred[:, keep[0], keep[1]]
    # print('scale = {}, cls_shape = {}, bbox_shape = {}'.format(scale, score.shape, reg.shape))
    bbox = np.vstack([score,
                      np.round((stride * keep[1] + 1) * inv_scale - 1),
                      np.round((stride * keep[0] + 1) * inv_scale - 1),
                      np.round((stride * keep[1] + cell_size) * inv_scale),
                      np.round((stride * keep[0] + cell_size) * inv_scale),
                      reg]).T
    return bbox

def bbox_nms(bbox, threshold, mode='union'):
    """Non Maximum Suppression with different mode.
    
    Parameters
    ----------
    bbox : numpy.ndarray, m x n, m >= 5
    threshold : float, threshold to suppress overlapped face
    mode : str, optional ['union', 'min']

    Returns:
    -------
        list of int, indexes to keep
    """
    scores = bbox[:, 0]
    x1 = bbox[:, 1]
    y1 = bbox[:, 2]
    x2 = bbox[:, 3]
    y2 = bbox[:, 4]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        if mode == 'union':
            overlap = inter / (areas[i] + areas[order[1:]] - inter)
        elif mode == 'min':
            overlap = inter / np.minimum(areas[i], areas[order[1:]])

        inds = np.where(overlap <= threshold)[0]
        order = order[inds + 1]
    return keep

def bbox_refine(bbox):
    """Refine bbox.
    
    Parameters
    ----------
    bbox : numpy.ndarray, n x 9
  
    Returns:
    -------
        numpy.ndarray, n x 5 [score, x1, y1, x2, y2]
    """
    bbsize = np.tile(bbox[:, 3:5] - bbox[:, 1:3], 2)
    return np.hstack([bbox[:, 0][..., None],
                      bbox[:, 1:5] + bbox[:, 5:] * bbsize])

def crop_pad(image, bbox, size):
    """Crop square patch with zero padding.
    
    Parameters
    ----------
    image : numpy.ndarray, h x w x 3
    bbox : numpy.ndarray, n x 5 [score, x1, y1, x2, y2]
    size : int, output size of resized square patch
  
    Returns:
    -------
        numpy.ndarray, n x 3 x size x size
    """
    # square bbox
    wh = bbox[:, 3:5] - bbox[:, 1:3]
    maxl = np.max(wh, axis=1)
    x = np.round(bbox[:,1] + (wh[:,0] - maxl) * 0.5).astype(np.int)
    y = np.round(bbox[:,2] + (wh[:,1] - maxl) * 0.5).astype(np.int)
    maxl = maxl.astype(np.int)
    bbox[:, 1] = x
    bbox[:, 2] = y
    bbox[:, 3] = x + maxl
    bbox[:, 4] = y + maxl

    patchs = []
    h, w = image.shape[:2]
    for x1, y1, ml in zip(x, y, maxl):
        x2, y2 = x1 + ml, y1 + ml 
        roi_on_image = [max(0, x1), max(0, y1), min(w, x2), min(h, y2)]
        roi_on_patch = [roi_on_image[0] - x1, roi_on_image[1] - y1,
                        roi_on_image[2] - x1, roi_on_image[3] - y1]
        patch = np.zeros((ml, ml, 3), dtype=np.float32)
        patch[roi_on_patch[1]:roi_on_patch[3], roi_on_patch[0]:roi_on_patch[2], :] \
            = image[roi_on_image[1]:roi_on_image[3], roi_on_image[0]:roi_on_image[2], :]
        patchs.append(cv.resize(patch, (size, size)).transpose((2,0,1)))
    return np.stack(patchs, axis=0)


def bbox_iou(bbox, bboxes):
    """Calculate Intersection-Over-Union(IOU) of bbox and bboxes.

    Parameters
    ----------
    bbox : numpy.ndarray, (4,)
    bboxes : numpy.ndarray, n x 4

    Returns
    -------
        numpy.ndarray, (n,)
    """
    if bbox.shape[0] < 4 or bboxes.shape[1] < 4:
        raise IndexError("Bounding boxes axis 1 must have at least length 4")
    
    tl = np.maximum(bbox[:2], bboxes[:, :2])
    br = np.minimum(bbox[2:4], bboxes[:, 2:4])

    inter = np.prod(br - tl, axis=1) * (tl < br).all(axis=1)
    area = np.prod(bbox[2:4] - bbox[:2], axis=0)
    areas = np.prod(bboxes[:, 2:4] - bboxes[:, :2], axis=1)
    iou = inter / (area + areas - inter)
    return iou