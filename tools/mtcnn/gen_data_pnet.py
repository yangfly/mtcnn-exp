import os
import time
import argparse
import cv2 as cv
import mxnet as mx
import numpy as np
import numpy.random as npr
import multiprocessing as mp
from utils import bbox_iou

def data_encode(image, samples, gt_bbox, q_out, counts, unique_id):
    neg = np.where(samples[:,4] == 0)[0]
    pos = np.where(samples[:,4] == 1)[0]
    part = np.where(samples[:,4] == 2)[0]
    # neg : pos : part = 3 : 1 : 1
    num = len(pos)
    neg_keep = npr.choice(neg, size=num*3, replace=True)
    if len(part) > 0:
        part_keep = npr.choice(part, size=num, replace=True)
    else:
        part_keep = part
    keep = np.hstack((neg_keep, pos, part_keep))
    counts[0] += len(neg_keep)
    counts[1] += len(pos)
    counts[2] += len(part_keep)
    # print(samples[keep, 4])
    for box in samples[keep]:
        x1, y1, x2, y2, _ = box
        crop = image[y1:y2, x1:x2, :]
        data = cv.resize(crop, (12, 12))
        iou = bbox_iou(box, gt_bbox)
        max_id = np.argmax(iou)
        gbox = gt_bbox[max_id].astype(np.float32)
        label = np.empty(5, dtype=np.float32)
        # print(box.shape, iou.shape)
        label[0] = iou[max_id] # iou
        label[1:] = (gbox - box[:4]) / (x2 - x1) # delta
        # print(_, label)
        header = mx.recordio.IRHeader(0, label, unique_id.value, 0) # unique id
        try:
            packed = mx.recordio.pack_img(header, data)
            q_out.put((unique_id.value, packed))
            unique_id.value += 1
        except Exception as e:
            traceback.print_exc()
            print('pack_img error on sample: %d' % unique_id.value, e)
            q_out.put((unique_id.valiue, None))

def add_samples(samples, box, max_iou):
    if max_iou < 0.3:
        box[4] = 0  # neg
    elif max_iou >= 0.65:
        box[4] = 1  # pos
    elif max_iou >= 0.4:
        box[4] = 2  # part
    if box[4] >= 0:
        samples.append(box)

def random_sample(path, gt_bbox, q_out, counts, unique_id):
    image = cv.imread(path, cv.IMREAD_COLOR)
    height, width, channel = image.shape
    samples = []
    # sample global negative samples
    for i in range(50):
        size = npr.randint(12, min(width, height) / 2)
        nx = npr.randint(0, width - size)
        ny = npr.randint(0, height - size)
        box = np.array([nx, ny, nx + size, ny + size, -1])
        max_iou = np.max(bbox_iou(box, gt_bbox))
        add_samples(samples, box, max_iou)
    # generate local negative / positive / part samples
    for gt_box in gt_bbox:
        x1, y1, x2, y2 = gt_box
        w, h = x2 - x1, y2 - y1
        # ignore small faces in case of imprecise groudtruth
        if max(w, h) < 40 or x1 < 0 or y1 < 0:
            continue
        # generate local hard negative samples
        for i in range(5):
            size = npr.randint(12, min(width, height) / 2)
            nx = npr.randint(max(x1-size+1, 0), min(x2-1, width-size))
            ny = npr.randint(max(y1-size+1, 0), min(y2-1, height-size))
            box = np.array([nx, ny, nx + size, ny + size, -1])
            max_iou = np.max(bbox_iou(box, gt_bbox))
            add_samples(samples, box, max_iou)
        # generate local positive and part samples
        for i in range(20):
            size = npr.randint(min(w,h) * 0.8, np.ceil(max(w,h) * 1.25))
            dx = npr.randint(w * -0.2, w * 0.2)
            dy = npr.randint(h * -0.2, h * 0.2)
            nx = max(x1 + dx + (w - size) // 2, 0)
            ny = max(y1 + dy + (h - size) // 2, 0)
            if nx + size > width or ny + size > height:
                continue
            box = np.array([nx, ny, nx + size, ny + size, -1])
            max_iou = np.max(bbox_iou(box, gt_bbox))
            add_samples(samples, box, max_iou)
    samples = np.array(samples)
    # encode data and label to dataset
    data_encode(image, samples, gt_bbox, q_out, counts, unique_id)

def sample_worker(root, pid, q_in, q_out, counts, unique_id):
    while True:
        data = q_in.get()
        if data is None:
            break
        path, gt_bbox = data
        if len(gt_bbox) == 0:
            continue
        path = os.path.join(root, path)
        # [x, y, w, h] to [x1, y1, x2, y2]
        gt_bbox[:, 2:] += gt_bbox[:, :2]
        random_sample(path, gt_bbox, q_out, counts, unique_id)

def write_worker(prefix, q_out, counts, unique_id):
    print('[write] %s.[rec|idx]' % prefix)
    if not os.path.exists(os.path.dirname(prefix)):
        os.makedirs(os.path.dirname(prefix))
    record = mx.recordio.MXIndexedRecordIO(prefix + '.idx',
                                           prefix + '.rec', 'w')
    pre_time = time.time()
    while True:
        data = q_out.get()
        if data is None:
            break
        if data[1] is None:
            continue  # pass failure sample
        record.write_idx(*data)
        if unique_id.value % 1000 == 0:
            cur_time = time.time()
            print('[write] {}  ({}  {}  {})  time: {:.3f}'.format(unique_id.value, *counts, cur_time-pre_time))
            pre_time = cur_time

    if unique_id.value % 1000 != 0:
        cur_time = time.time()
        print('[write] {}  ({}  {}  {})  time: {:.3f}'.format(unique_id.value, *counts, cur_time-pre_time))
    record.close()

def parse_annos(subset, annotxt):
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
                if box[2] > 0 and box[3] > 0:
                    box[2] += 1 # inter-w to real-w
                    box[3] += 1 # inter-h to real-h
                    gt_bbox.append(box)
            annos.append((path, np.array(gt_bbox)))
    return annos

def gen_data(subsets, annotxt, root, prefix, num_worker):
    for subset in subsets:
        annos = parse_annos(subset, annotxt)
        print('[{}] {} images.'.format(subset, len(annos)))
        q_in = [mp.Queue() for i in range(num_worker)]
        q_out = mp.Queue()
        counts = mp.Array('i', [0, 0, 0])
        unique_id = mp.Value('i', 0)
        sample_procs = [mp.Process(target=sample_worker, args=(root.format(subset), pid, q_in[pid], q_out, counts, unique_id)) for pid in range(num_worker)]
        write_proc = mp.Process(target=write_worker, args=(prefix.format(subset), q_out, counts, unique_id))
        for p in sample_procs:
            p.start()
        write_proc.start()
        for i, anno in enumerate(annos):
            q_in[i % num_worker].put(anno)
        for q in q_in: # input None as stop signal
            q.put(None)
        for p in sample_procs:
            p.join()
        q_out.put(None) # input None as stop signal
        write_proc.join()
        print('[{}] down.'.format(subset))

def parse_args():
    parser = argparse.ArgumentParser(description='Generate train & val data for Pnet.')
    parser.add_argument('-j', dest='num_worker', type=int, default=4,  help='Multi cores.')
    parser.add_argument('--subsets', type=lambda s: tuple(s.split(',')), default=['train', 'val'],  help='Data subsets to generate.')
    return parser.parse_args()

if __name__ == '__main__':
    from pprint import pprint
    args = parse_args()
    pprint(args)
    annotxt = 'data/widerface/wider_face_split/wider_face_{}_bbx_gt.txt'
    prefix = 'data/widerface/mtcnn/pnet_{}'
    root = 'data/widerface/WIDER_{}/images'
    gen_data(args.subsets, annotxt, root, prefix, args.num_worker)
    