"""Adapted from https://github.com/cypw/MXNet2Caffe"""
import os
import sys
import json
import argparse
import mxnet as mx
from easydict import EasyDict as edict
os.environ['GLOG_minloglevel'] = '1'
sys.path.insert(0, '/home/yf/test/caffe/python')
import caffe
import nn

def add_kernel_stride_pad(kwargs, attrs):
    for key in ['kernel', 'stride', 'pad']:
        attr = eval(attrs[key])
        if attr[0] == attr[1]:
            name = key + '_size' if key == 'kernel' else key
            kwargs[name] = attr[0]
        else:
            kwargs[key + '_h'] = attr[0]
            kwargs[key + '_w'] = attr[1]

def convert_net(symbol, prototxt, size):
    with open(symbol, 'r') as sym:
        nodes = json.load(sym)['nodes']
    layers = []
    bottoms = []
    for node in nodes:
        node = edict(node)
        attrs = edict(getattr(node, 'attrs', {}))
        if node.op == 'null':
            if node['name'] == 'data':
                layers.append(caffe.layers.Input(name='data', shape={'dim': [1, 3, size, size]}))
            else:
                layers.append(None)
                continue
        elif node.op == 'Convolution':
            kwargs = {}
            kwargs['name'] = node.name.replace('_fwd', '')
            kwargs['num_output'] = eval(attrs.num_filter)
            kwargs['group'] = eval(getattr(attrs, 'num_group', '1'))
            add_kernel_stride_pad(kwargs, attrs)
            bottom = node['inputs'][0][0]
            if eval(getattr(attrs, 'no_bias', 'False')):
                kwargs['bias_term'] = False
            if kwargs['num_output'] == kwargs['group']:
                op = 'Convolution' #'ConvolutionDepthwise'
            else:
                op = 'Convolution'
            layers.append(getattr(caffe.layers, op)(layers[bottom], **kwargs))
            bottoms.append(bottom)
        elif node.op == 'FullyConnected':
            bottom = node['inputs'][0][0]
            layers.append(caffe.layers.InnerProduct(layers[bottom], name=node.name.replace('_fwd', ''),
                          num_output=eval(attrs.num_hidden)))
            bottoms.append(bottom)
        elif node.op == 'Pooling':
            kwargs = {}
            kwargs['name'] = node.name.replace('_fwd', '')
            add_kernel_stride_pad(kwargs, attrs)
            if attrs.pool_type == 'max':
                kwargs['pool'] = caffe.params.Pooling.MAX
            elif attrs.pool_type == 'avg':
                kwargs['pool'] = caffe.params.Pooling.MAX
            else:
                raise ValueError('unsupported pool type: {}'.format(attrs.pool_type))
            if attrs.pooling_convention == 'full':
                pass
            elif attrs.pooling_convention == 'valid':
                kwargs['round_mode'] = caffe.params.Pooling.FLOOR
            else:
                raise ValueError('unsupported pool convention: {}'.format(attrs.pooling_convention))    
            if eval(getattr(attrs, 'global_pool', 'False')):
                kwargs['global_pooling'] = True
            bottom = node['inputs'][0][0]
            layers.append(caffe.layers.Pooling(layers[bottom], **kwargs))
            bottoms.append(bottom)
        elif node.op == 'Activation':
            bottom = node['inputs'][0][0]
            mdict = {'relu': 'ReLU', 'sigmoid': 'Sigmoid', 'tanh': 'TanH', 'softsign': None, 'softrelu': None}
            if not mdict[attrs.act_type]:
                raise ValueError('unsupported act type: {}'.format(attrs.act_type)) 
            layers.append(getattr(caffe.layers, mdict[attrs.act_type])(layers[bottom], name=node.name.replace('_fwd', '')))
            bottoms.append(bottom)
        elif node.op == 'LeakyReLU':
            act_type = attrs.act_type
            bottom = node['inputs'][0][0]
            if act_type == 'elu':
                layers.append(caffe.layers.ELU(layers[bottom], name=node.name.replace('_fwd', ''),
                              alpha=eval(getattr(attrs, 'slope', '0.25'))))
            elif act_type == 'prelu':
                layers.append(caffe.layers.PReLU(layers[bottom], name=node.name.replace('_fwd', '')))
            else:
                raise ValueError('unsupported act type: {}'.format(attrs.act_type))
            bottoms.append(bottom)
        elif node.op == 'BatchNorm':
            bottom = node['inputs'][0][0]
            bn = caffe.layers.BatchNorm(layers[bottom], name=node.name.replace('_fwd', '')+'/bn',
                    moving_average_fraction=eval(getattr(attrs, 'momentum', '0.9')),
                    eps=eval(getattr(attrs, 'eps', '0.001')))
            layers.append(caffe.layers.Scale(bn, name=node.name.replace('_fwd', '')+'/scale', bias_term=True))
            bottoms.append(bottom)
        elif node.op == 'elemwise_add':
            bottom = [b[0] for b in node['inputs'][0]]
            layers.append(caffe.layers.Eltwise(*[layers[b] for b in bottom], name=node.name.replace('_fwd', ''),
                          operation='SUM'))
            bottoms.extend(bottom)
        elif node.op == 'Concat':
            bottom = [b[0] for b in node['inputs'][0]]
            layers.append(caffe.layers.Concat(*[layers[b] for b in bottom], name=node.name.replace('_fwd', '')))
            bottoms.extend(bottom)
        elif node.op == 'softmax':
            bottom = node['inputs'][0][0]
            layers.append(caffe.layers.Softmax(layers[bottom], name=node.name.replace('_fwd', ''),
                          axis=eval(getattr(attrs, 'axis', '-1'))))
            bottoms.append(bottom)
        elif node.op in ['Dropout','Flatten', 'Reshape', 'SwapAxis']:
            bottom = node['inputs'][0][0]
            layers.append(layers[bottom])
            bottoms.append(bottom)
        else:
            raise ValueError('unsupported operator: {}'.format(node.op))
    outputs = []
    for i, layer in enumerate(layers):
        if layer and i not in bottoms:
            outputs.append(layer)
    # print(caffe.to_proto(*outputs))
    
    # Manually add ChannelShuffle if needed.
    # Layer {
    #     name: "shuffle"
    #     type: "ShuffleChannel"
    #     bottom: "conv"
    #     top: "shuffle"
    #     shuffle_channel_param = {
    #         group = 2
    #     }
    # }
    with open(prototxt, 'w') as f:
        print(caffe.to_proto(*outputs), file=f)

def convert_weight(mx_net, prototxt, model):
    cf_net = caffe.Net(prototxt, phase=caffe.TEST)
    cf_params = cf_net.params
    mx_params = mx_net.collect_params()

    for key, param in mx_params.items():
        if key.endswith('_weight'): # conv
            cf_params[key[:-7]][0].data.flat = mx_params[key].data().asnumpy().flat
        elif key.endswith('_bias'):
            cf_params[key[:-5]][1].data.flat = mx_params[key].data().asnumpy().flat
        elif key.endswith('_alpha'): # prelu
            cf_params[key[:-6]][0].data.flat = mx_params[key].data().asnumpy().flat
        elif key.endswith('_running_mean'): # bn
            cf_key = key.replace('_running_mean', '/bn')
            cf_params[cf_key][0].data.flat = mx_params[key].data().asnumpy().flat
        elif key.endswith('_running_var'):
            cf_key = key.replace('_running_var', '/bn')
            cf_params[cf_key][1].data.flat = mx_params[key].data().asnumpy().flat
            cf_params[cf_key][2].data[...] = 1
        elif key.endswith('_gamma'): # scale
            cf_key = key.replace('_gamma', '/scale')
            cf_params[cf_key][0].data.flat = mx_params[key].data().asnumpy().flat
        elif key.endswith('_beta'): # bn
            cf_key = key.replace('_beta', '/scale')
            cf_params[cf_key][1].data.flat = mx_params[key].data().asnumpy().flat
        else:
            raise KeyError('Unsupported param: {}'.format(key))
    cf_net.save(model)

def convert_model(network, param, size, root='caffe_/models'):
    if not os.path.exists(root):
        os.makedirs(root)
    from nn import get_net
    net = get_net(network)
    net.load_parameters(param)

    prefix = os.path.join(root, network)
    symbol = os.path.join(os.path.dirname(param), network + '.json')
    convert_net(symbol, prefix + '.prototxt', size)
    convert_weight(net, prefix+'.prototxt', prefix+'.caffemodel')
    # check forward
    net2 = caffe.Net(prefix+'.prototxt', weights=prefix+'.caffemodel', phase=caffe.TEST)
    data = mx.nd.random.randn(1,3,size,size)
    net2.blobs['Input1'].data[...] = data.asnumpy()
    print(net(data))
    print(net2.forward())

def parse_args():
    parser = argparse.ArgumentParser(description='Convert MXNet model to Caffe model')
    parser.add_argument('network', type=str)
    parser.add_argument('param', type=str)
    parser.add_argument('size', type=int)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    convert_model(args.network, args.param, args.size)
    
