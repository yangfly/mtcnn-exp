import os
import cv2 as cv
import subprocess
from multiprocessing import Process
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
import nn

__all__ = ['fddb_eval', 'MPDet']

def evaluate(epoch, save_prefix, maxfp, ctx):
    fddb_prefix = os.path.join(save_prefix, 'fddb')
    image_root = 'data/fddb/images'
    names = [name.strip() for name in open('data/fddb/eval/imList.txt')]
    if not os.path.exists(fddb_prefix):
        os.mkdir(fddb_prefix)
    mtcnn = nn.MtcnnV2(pnet=save_prefix + '/{:03d}.params'.format(epoch), ctx=ctx)
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
    def __init__(self, save_prefix, maxfp, ctx):
        self._save_prefix = save_prefix
        self._maxfp = maxfp
        self._ctx = ctx
        self._workers = []

    def add(self, epoch):
        self._workers.append(Process(target=evaluate,
            args = (epoch, self._save_prefix, self._maxfp, self._ctx)))
        self._workers[-1].start()
    
    def join(self, show=False):
        for p in self._workers:
            p.join()
        # write scores into map file
        args = ['python', 'tools/fddb/plot.py', os.path.join(self._save_prefix, 'fddb'), str(self._maxfp)]
        if show:
            args.append('-s')
        subprocess.call(args)

def fddb_eval(epochs, save_prefix, ctx, maxfp=300000, show=True):
    mpd = MPDet(save_prefix, maxfp, ctx)
    for epoch in epochs:
        mpd.add(epoch)
    mpd.join(show)

if __name__ == '__main__':
    import mxnet as mx
    import sys
    # mpd = MPDet('models/mtcnn/pnet2', 300000, mx.gpu(7))
    mpd = MPDet('models/v1/'+sys.argv[1], 300000, mx.gpu(int(sys.argv[2])))
    for epoch in range(28, 33):
        mpd.add(epoch)
    mpd.join(True)

