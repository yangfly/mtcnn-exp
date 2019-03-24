import os
import random
import logging
import mxnet as mx
import numpy as np

__all__ = ['random_seed', 'init_net', 'load_model', 'save_model', 'net_export', 'FileLogger']

logging.basicConfig(
    level   = logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt = '%Y-%m-%d %H:%M:%S')

def random_seed(a=233):
    '''Fix random seed for 100 percent reproducibility.'''
    random.seed(a)
    mx.random.seed(a)
    np.random.seed(a)

def init_net(net, resume, init, ctx):
    if resume:
        net.load_parameters(resume, ctx=ctx)
    else:
        if init.lower() =='xavier':
            initializer = mx.init.Xavier(factor_type='in', rnd_type='gaussian', magnitude=2)
        else:
            initializer = mx.init.MSRAPrelu()
        net.initialize(init=initializer, ctx=ctx)

def load_model(prefix, epoch=0, input_size=12, ctx=mx.cpu()):
    sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
    mod = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
    mod.bind(data_shapes=[('data', (1, 3, input_size, input_size))], for_training=False)           
    mod.set_params(arg_params, aux_params, allow_missing=False)
    return mod

def save_model(net, prefix, epoch, maps, end_epoch):
    def update_symlink(src, dest):
        try:
            os.remove(dest)
        except:
            pass
        os.symlink(src, dest)

    model_path = '{:s}{:03d}.params'.format(prefix, epoch)
    best_path = '{:s}best.params'.format(prefix)
    if epoch == 0:
        fmap = open('{}maps.log'.format(prefix), 'w')
        fmap.write('epoch\tval_acc  val_mse\ttrain_acc\tcls_loss  loc_loss  sum_loss\n')
    else:
        fmap = open('{}maps.log'.format(prefix), 'a')
    msg = '{:03d}:\t{:.6f}  {:.6f}\t{:.6f}\t{:.6f}  {:.6f}  {:.6f}'.format(epoch, *maps[-1])
    if maps[-1][0] == max(maps)[0]:
        net.save_parameters(model_path)
        update_symlink(model_path, best_path)
        msg += '    *'
    elif end_epoch - epoch <= 5:
        net.save_parameters(model_path)
    fmap.write(msg + '\n')
    fmap.close()

def net_export(net, json):
    x = mx.sym.var('data', shape=(1, 3, net.size, net.size))
    try:
        y = net(x)
    except Exception:
        net.initialize()
        net.hybridize()
        y = net(x)
    if isinstance(y, tuple):
        y = mx.sym.Group(y)
    y.save(json)

def FileLogger(path):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    logger = logging.getLogger()
    formatter = logging.Formatter('[%(asctime)s] %(message)s')
    logger.setLevel(logging.INFO)
    log_dir = os.path.dirname(path)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if os.path.exists(path):
        os.remove(path)
    fh = logging.FileHandler(path)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger