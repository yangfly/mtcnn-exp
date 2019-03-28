import os
import sys
import cv2 as cv
import mxnet as mx
import argparse
import subprocess
from multiprocessing import Process
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
import nn

def evaluate(network, epoch, save_prefix, maxfp, ctx):
    fddb_prefix = os.path.join(save_prefix, 'fddb')
    image_root = 'data/fddb/images'
    names = [name.strip() for name in open('data/fddb/eval/imList.txt')]
    if not os.path.exists(fddb_prefix):
        os.mkdir(fddb_prefix)
    mtcnn = nn.MtcnnV2(pnet=nn.load_model(network, save_prefix + '/{:03d}.params'.format(epoch), ctx))
    # detect fddb images and write to detect file
    det_name = '{:s}/{:03d}'.format(fddb_prefix, epoch)
    with open(det_name + '.txt', 'w') as f:
        for name in names:
            im = cv.imread(os.path.join(image_root, name + '.jpg'), cv.IMREAD_COLOR)
            im = cv.cvtColor(im, cv.COLOR_BGR2RGB)
            bbox = mtcnn.detect(im)
            f.write(name + '\n')
            f.write('{}\n'.format(len(bbox)))
            for score, x1, y1, x2, y2 in bbox:
                f.write('{:.3f} {:.3f} {:.3f} {:.3f} {:.10f}\n'
                        .format(x1, y1, x2-x1, y2-y1, score))
    # run evaluate
    subprocess.call(['data/fddb/evaluation/evaluate', '-f', '0',
                     '-l', 'data/fddb/eval/imList.txt',
                     '-a', 'data/fddb/eval/ellipseList.txt',
                     '-i', 'data/fddb/images/',
                     '-d', det_name + '.txt',
                     '-r', det_name + '-'], stdout=subprocess.PIPE)
                    
class MPDet(object):
    def __init__(self, network, save_prefix, maxfp, ctx):
        self._network = network
        self._save_prefix = save_prefix
        self._maxfp = maxfp
        self._ctx = ctx
        self._workers = []

    def add(self, epoch):
        self._workers.append(Process(target=evaluate,
            args = (self._network, epoch, self._save_prefix, self._maxfp, self._ctx)))
        self._workers[-1].start()
    
    def join(self, show=False):
        for p in self._workers:
            p.join()
        # write scores into map file
        args = ['python', 'tools/fddb/plot.py', os.path.join(self._save_prefix, 'fddb'), str(self._maxfp)]
        if show:
            args.append('-s')
        subprocess.call(args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Eval model snapshots.')
    parser.add_argument('network', type=str, help='network name.')
    parser.add_argument('path', type=str, help='model path.')
    parser.add_argument('epochs', type=str, help='model epochs.')
    parser.add_argument('gpu', type=int, help='device id.')
    parser.add_argument('-s', '--show', action='store_true', help='show ploted image.')
    parser.add_argument('maxfp', type=int, default=300000, help='max false positive.')
    args = parser.parse_args()
    args.epochs = map(int, args.epochs.split(','))
    args.ctx = mx.gpu(args.gpu)
    mpd = MPDet(args.network, args.path, args.maxfp, args.ctx)
    for epoch in args.epochs:
        mpd.add(epoch)
    mpd.join(args.show)
