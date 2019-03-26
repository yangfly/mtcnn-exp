#!/usr/bin/env python

import os
import sys
import pylab
import argparse
import numpy as np
import matplotlib
from utils import FileLogger

def load_roc(method, color, root, mmax, logger):
    rocs = []
    for subfix in ['-DiscROC.txt', '-ContROC.txt']:
        path = os.path.join(root, method + subfix)
        data = np.loadtxt(path, delimiter=' ')
        data.sort(axis=0)
        data = data[::-1]
        pos = np.where(data[:, 1] < mmax)[0][0]
        pr = data[pos, 0]
        num = int(data[pos, 1])
        label = '{}-{}: {:.3f}'.format(subfix[1], method, pr)
        pid = pylab.plot(data[:, 1], data[:, 0], color=color, linewidth=2)[0]
        rocs.append((pr, pid, label))
    logger.info('{:>10}  {}  {:.3f}  {:.3f}'.format(method, num, rocs[0][0], rocs[1][0]))
    return rocs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot FDDB AP curves.')
    parser.add_argument('det_path', type=str, help='Path to fddb format detect files.')
    parser.add_argument('max', type=int, default=300000, help='max false postives.')
    parser.add_argument('-s', '--show', action='store_true', help='show ploted image.')
    args = parser.parse_args()
    assert os.path.exists(args.det_path)

    methods = []
    for txt in os.listdir(args.det_path):
        if txt.endswith('DiscROC.txt'):
            txt = txt[:-11]
            if txt[-1] in ('-' or '_'):
                txt = txt[:-1]
            methods.append(txt)

    logger = FileLogger(os.path.join(args.det_path, 'maps.txt'), False)
    colors = pylab.cm.Set1(np.linspace(0, 1, max(5, len(methods))))
    pylab.figure(figsize=(12, 8))
    algos = []
    for i, method in enumerate(methods):
        rocs = load_roc(method, colors[i], args.det_path, args.max, logger)
        algos.extend(rocs)
    algos = sorted(algos, key=lambda x: x[0], reverse=True)

    labels = []
    pids = []
    for i, algo in enumerate(algos):
        pids.append(algo[1])
        labels.append(algo[2])
        pylab.setp(algo[1], zorder=len(algos) - i)

    pylab.legend(pids, labels, loc='lower right', ncol=2)
    pylab.ylabel("True positive rate")
    pylab.xlabel("False positives")
    pylab.grid()
    pylab.gca().set_xlim((0, args.max))
    pylab.gca().set_ylim((0, 1))
    pylab.yticks(np.linspace(0, 1, 11))
    pylab.savefig(os.path.join(args.det_path, 'fddb.jpg'))
    if args.show:
        pylab.show()
        pylab.draw()
