import os
import argparse
import cv2 as cv
import mxnet as mx
import numpy as np
from tqdm import tqdm
import numpy.random as npr
from easydict import EasyDict
from utils import bbox_iou
npr.seed(233)

# sample settings
_C = EasyDict(dict(
    pos_thresh     = 0.65,
    part_thresh    = 0.4,
    neg_thresh     = 0.3,
    global_trails  = 50, # max trails to generate global negative samples
    overlap_trails = 5,  # max trails to generate overlap negative samples
    nearby_trails  = 20, # max trails to generate nearby positive and part samples 
    ignore_face    = 40, # ignore small faces
    filter_neg     = 600000,
    filter_part    = 300000
))
# todo ignore max/min face

def random_sample(id, path, gt_bbox):
    image = cv.imread(path, cv.IMREAD_COLOR)
    height, width, _ = image.shape
    samples = []

    # sample global negative samples
    for _ in range(_C.global_trails):
        size = npr.randint(12, min(width, height) / 2)
        nx = npr.randint(0, width - size)
        ny = npr.randint(0, height - size)
        box = np.array([nx, ny, nx + size, ny + size])
        max_iou = np.max(bbox_iou(box, gt_bbox))
        if max_iou < _C.neg_thresh: #  or max_iou >= _C.part_thresh:
            samples.append((id, max_iou, nx, ny, nx+size, ny+size))

    # generate local negative / positive / part samples
    for gt_box in gt_bbox:
        x1, y1, x2, y2 = gt_box
        w, h = x2 - x1, y2 - y1
        # ignore small faces in case of imprecise groudtruth
        if max(w, h) < _C.ignore_face:
            continue
        # generate negative samples that overlap with gt.
        for i in range(_C.overlap_trails):
            size = npr.randint(12, min(width, height) / 2)
            nx = npr.randint(-size, w) + x1
            ny = npr.randint(-size, h) + y1
            if nx < 0 or ny < 0 or nx + size > width or ny + size > height:
                continue
            box = np.array([nx, ny, nx + size, ny + size])
            max_iou = np.max(bbox_iou(box, gt_bbox))
            if max_iou < _C.neg_thresh: #  or max_iou >= _C.part_thresh:
                samples.append((id, max_iou, nx, ny, nx+size, ny+size))
        # generate nearby positive and part samples
        for i in range(_C.nearby_trails):
            size = npr.randint(min(w,h) * 0.8, np.ceil(max(w,h) * 1.25))
            dx = npr.randint(w * -0.2, w * 0.2)
            dy = npr.randint(h * -0.2, h * 0.2)
            nx = x1 + dx + (w - size) // 2
            ny = y1 + dy + (h - size) // 2
            if nx < 0 or ny < 0 or nx + size > width or ny + size > height:
                continue
            box = np.array([nx, ny, nx + size, ny + size])
            max_iou = np.max(bbox_iou(box, gt_bbox))
            if max_iou < _C.neg_thresh or max_iou >= _C.part_thresh:
                samples.append((id, max_iou, nx, ny, nx+size, ny+size))
    return samples

def filter_samples(samples, subset):
    samples = np.array(samples)
    ious = samples[:, 1]
    pos_ids = np.where(ious >= _C.pos_thresh)[0]
    part_ids = np.where(np.logical_and(ious >= _C.part_thresh, ious < _C.pos_thresh))[0]
    neg_ids = np.where(ious < _C.neg_thresh)[0]
    print('[{}] sample: pos = {}, part = {}, neg = {}, total={}'.format(subset, len(pos_ids),
        len(part_ids), len(neg_ids), len(pos_ids) + len(part_ids) + len(neg_ids)))
    part_keep = npr.choice(part_ids, size=min(_C.filter_part, len(part_ids)))
    neg_keep = npr.choice(neg_ids, size=min(_C.filter_neg, len(neg_ids)))
    print('[{}] filter: pos = {}, part = {}, neg = {}, total={}'.format(subset, len(pos_ids),
        len(part_keep), len(neg_keep), len(pos_ids) + len(part_keep) + len(neg_keep)))
    keep = np.hstack([pos_ids, part_keep, neg_keep])
    samples = samples[keep]
    return samples

def sample_dataset(subset, annotxt, root, prefix):
    # load annotations
    annos = []
    with open(annotxt.format(subset), 'r') as f:
        while True:
            path = f.readline().strip()
            if path == '':
                break
            num = int(f.readline().strip())
            gt_bbox = []
            for i in range(num):
                box = [int(a) for a in f.readline().strip().split()[:4]]
                if all(x > 0 for x in box):
                    # matlab to c style
                    box[2] += box[0]
                    box[3] += box[1]
                    box[0] -= 1
                    box[1] -= 1
                    gt_bbox.append(box)
            if gt_bbox:
                annos.append((path, np.array(gt_bbox)))

    # sample images
    samples = []
    root = root.format(subset)
    with tqdm(enumerate(annos), total=len(annos), unit=' annos', ncols=0) as t:
        for id, anno in t:
            path, gt_bbox = anno
            path = os.path.join(root, path)
            samples.extend(random_sample(id, path, gt_bbox))
    
    # filter samples and save rec
    prefix = prefix.format(subset)
    samples = filter_samples(samples, subset)
    rec = mx.recordio.MXIndexedRecordIO(prefix + '.idx', prefix + '.rec', 'w')
    index = samples[:, 0]
    unique_id = 0
    with tqdm(enumerate(annos), total=len(annos), unit=' annos', ncols=0) as t:
        for id, anno in t:
            path, gt_bbox = anno
            image = cv.imread(os.path.join(root, path), cv.IMREAD_COLOR)
            for i in np.where(index==id)[0]:
                box = samples[i, 2:]
                ious = bbox_iou(box, gt_bbox)
                max_id = np.argmax(ious)
                gbox = gt_bbox[max_id].astype(np.float32)
                x1, y1, x2, y2 = box.astype(np.int)
                label = np.empty(5, dtype=np.float32)
                label[0] = ious[max_id] # iou
                label[1:] = (gbox - box) / (x2 - x1) # delta
                crop = image[y1:y2, x1:x2, :] 
                data = cv.resize(crop, (12, 12))
                header = mx.recordio.IRHeader(0, label, unique_id, 0)
                try:
                    packed = mx.recordio.pack_img(header, data)
                    rec.write_idx(unique_id, packed)
                    unique_id += 1
                except Exception as e:
                    print('pack_img error on sample: %d' % unique_id, e)
                    raise e
    rec.close()

def main():
    annotxt = 'data/widerface/wider_face_split/wider_face_{}_bbx_gt.txt'
    prefix = 'data/widerface/mtcnn/pnet_{}'
    root = 'data/widerface/WIDER_{}/images'
    for subset in ['train', 'val']:
        sample_dataset(subset, annotxt, root, prefix)
    
def test():
    subset = 'train'
    annotxt = 'data/widerface/wider_face_split/wider_face_{}_bbx_gt.txt'
    prefix = 'data/widerface/mtcnn/pnet_{}'
    root = 'data/widerface/WIDER_{}/images'
    # load annotations
    annos = []
    with open(annotxt.format(subset), 'r') as f:
        while True:
            path = f.readline().strip()
            if path == '':
                break
            num = int(f.readline().strip())
            gt_bbox = []
            for i in range(num):
                box = [int(a) for a in f.readline().strip().split()[:4]]
                if all(x > 0 for x in box):
                    # matlab to c style
                    box[2] += box[0]
                    box[3] += box[1]
                    box[0] -= 1
                    box[1] -= 1
                    gt_bbox.append(box)
            if gt_bbox:
                annos.append((path, np.array(gt_bbox)))
    path, gt_bbox = annos[231]
    samples = random_sample(231, os.path.join(root.format(subset), path), gt_bbox)
    samples = np.array(samples)
    ious = samples[:, 1]
    pos_ids = np.where(ious >= _C.pos_thresh)[0]
    part_ids = np.where(np.logical_and(ious >= _C.part_thresh, ious < _C.pos_thresh))[0]
    neg_ids = np.where(ious < _C.neg_thresh)[0]
    print('[sample] pos = {}, part = {}, neg = {}, total={}'.format(len(pos_ids), len(part_ids), len(neg_ids), len(pos_ids) + len(part_ids) + len(neg_ids)))


if __name__ == '__main__':
    main()
    
    