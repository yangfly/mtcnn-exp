import os
import argparse
import mxnet.ndarray.random as ndr
from mxnet import profiler
from nn import get_net
import time

profiler.set_config(profile_all     =True,
                    filename        = 'chrome_tracing_profile.json',  # File used for chrome://tracing visualization
                    continuous_dump = True,
                    aggregate_stats = True)  # Stats printed by dumps() call

def clean(network, param, size, root):
    net = get_net(network)
    net.load_parameters(param)
    net.hybridize()
    x = ndr.randn(1, 3, size, size)
    profiler.set_config('run')
    _ = net(x)
    t = time.time()
    for i in range(1000):
        net(x)
    print((time.time() - t))
    profiler.set_state('stop')
    print(profiler.dumps())
    # if not os.path.exists(root):
    #     os.mkdir(root)
    # net.export(os.path.join(root, network))

def parse_args():
    parser = argparse.ArgumentParser(description='Convert MXNet model to Caffe model')
    parser.add_argument('network', type=str)
    parser.add_argument('param', type=str)
    parser.add_argument('--size', type=int, default=1024)
    parser.add_argument('--root', type=str, default='models/2014')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    clean(args.network, args.param, args.size, args.root)