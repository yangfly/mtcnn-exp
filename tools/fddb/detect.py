from __future__ import division

import os
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
import argparse
import cv2 as cv
import mxnet as mx
from tqdm import tqdm
from nn import MtcnnV2

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Script to detect fddb image')
    parser.add_argument('--gpu', dest='gpu_id', type=int, default=2,
                        help='Id of gpu device')
    parser.add_argument('--pnet', type=str, default=None, help='Path to pnet')
    parser.add_argument('--rnet', type=str, default=None, help='Path to rnet')
    parser.add_argument('--onet', type=str, default=None, help='Path to onet')
    parser.add_argument('--dets', type=str, default='pnet.txt', help='Path to onet')
    args = parser.parse_args()
    return args



if __name__ == '__main__':
    args = parse_args()
    ctx = mx.gpu(args.gpu_id) if args.gpu_id >= 0 else mx.cpu()
    mtcnn = MtcnnV2(pnet=args.pnet, rnet=args.rnet, onet=args.onet, ctx=ctx)

    image_root = 'data/fddb/images'
    names = [name.strip() for name in open('data/fddb/eval/imList.txt')]
    with open(args.dets, 'w') as f:
        with tqdm(names, total=len(names), unit=' imgs', ncols=0) as t:
            for name in t:
                im = cv.imread(os.path.join(image_root, name + '.jpg'), cv.IMREAD_COLOR)
                im = cv.cvtColor(im, cv.COLOR_BGR2RGB)
                bbox = mtcnn.detect(im)
                f.write(name + '\n')
                f.write('{}\n'.format(len(bbox)))
                for score, x1, y1, x2, y2 in bbox:
                    f.write('{:.3f} {:.3f} {:.3f} {:.3f} {:.10f}\n'
                            .format(x1, y1, x2-x1, y2-y1, score))
