"""Train MTCNN Detector"""
import os
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
import time
import argparse
import mxnet as mx
from mxnet.gluon.utils import split_and_load
import nn
import utils

def parse_args():
    parser = argparse.ArgumentParser(description='Train MTCNN Detector.')
    parser.add_argument('--network', type=str, default='pnet',
                        help='Network name in [pnet, rnet, onet]')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Training mini-batch size')
    parser.add_argument('--gpus', type=str, default='2',
                        help='Training with GPUs, you can specify 0,1 for example.')
    parser.add_argument('--num-workers', '-j', dest='num_workers', type=int,
                        default=16, help='Number of data workers, you can use larger '
                        'number to accelerate data loading, if you CPU and GPUs are powerful.')
    parser.add_argument('--init', type=str, default='xavier',
                        help='Network initializer optional [xavier, msra]')
    parser.add_argument('--seed', type=int, default=233,
                        help='Random seed to be fixed.')
    
    solver = parser.add_argument_group('Solver Settings')
    solver.add_argument('--resume', type=str, default='',
                        help='Resume from previously saved parameters if not None. '
                        'For example, you can resume from xxx.params')
    solver.add_argument('--start-epoch', type=int, default=0,
                        help='Starting epoch for resuming, default is 0 for new training.'
                        'You can specify it to 100 for example to start from 100 epoch.')
    solver.add_argument('--epochs', type=int, default=32,
                        help='Training epochs.')
    solver.add_argument('--log-interval', type=int, default=1000,
                        help='Logging mini-batch interval. Default is 100.')
    solver.add_argument('--val-interval', type=int, default=1,
                        help='Epoch interval for validation, increase the number will reduce the '
                             'training time if validation is slow.')
    solver.add_argument('--save-prefix', type=str, default='models/mtcnn/pnet',
                        help='Saving parameter prefix')
    solver.add_argument('--save-interval', type=int, default=1,
                        help='Saving parameters epoch interval, best model will always be saved.')

    optimizer = parser.add_argument_group('SGD Optimizer')
    optimizer.add_argument('--lr', type=float, default=0.01,
                           help='Learning rate, default is 0.01')
    optimizer.add_argument('--lr-decay', type=float, default=0.1,
                           help='decay rate of learning rate. default is 0.1.')
    optimizer.add_argument('--lr-decay-epoch', type=str, default='14,28',
                           help='epoches at which learning rate decays. default is 10,18,24,27.')
    optimizer.add_argument('--momentum', type=float, default=0.9,
                           help='SGD momentum, default is 0.9')
    optimizer.add_argument('--wd', type=float, default=5e-4,
                           help='Weight decay, default is 5e-4')

    loss = parser.add_argument_group('Mtcnn Loss')
    loss.add_argument('--pos-thresh', type=float, default=0.65,
                      help='IoU above pos_thresh assigned as positive face.')
    loss.add_argument('--part-thresh', type=float, default=0.4,
                      help='Iou between part_thresh and pos_thresh assigned part face.')
    loss.add_argument('--neg-thresh', type=float, default=0.3,
                      help='IoU less than neg_thresh assigned negative face.')
    loss.add_argument('--ohem-ratio', type=float, default=0.7,
                      help='The ratio of loss to been kept after online hard example mining.')
    loss.add_argument('--cls-weight', type=float, default=1.0,
                      help='Loss weight for cls_loss.')
    loss.add_argument('--loc-weight', type=float, default=5.0,
                      help='Loss weight for loc_loss.')
    loss.add_argument('--loc-loss', type=str, default='smoothl1',
                      help='Which loss as loc_loss, optional in [smoothl1, euclid].')

    args = parser.parse_args()
    if not args.save_prefix:
        args.save_prefix = os.path.join('models/mtcnn', args.network)
    if not args.save_prefix.endswith('/'):
        args.save_prefix = args.save_prefix + '/'
    return args

def validate(net, val_data, ctx, acc_metric, mse_metric):
    """Test on validation dataset."""
    acc_metric.reset()
    mse_metric.reset()
    val_data.reset()
    net.hybridize()
    for batch in val_data:
        data = split_and_load(batch.data[0], ctx_list=ctx, batch_axis=0, even_split=False)
        labels = split_and_load(batch.label[0], ctx_list=ctx, batch_axis=0, even_split=False)
        cls_preds = []
        loc_preds = []
        for x in data:
            cls_pred, loc_pred = net(x)
            cls_preds.append(cls_pred)
            loc_preds.append(loc_pred)
        acc_metric.update(labels, cls_preds)
        mse_metric.update(labels, loc_preds)
    return nn.metric_msg(acc_metric, mse_metric)

def train(args):
    logger = utils.FileLogger(args.save_prefix + 'train.log')
    logger.info(args)
    utils.random_seed(args.seed)
    ctx = [mx.gpu(int(i)) for i in args.gpus.split(',') if i.strip()]
    ctx = ctx if ctx else [mx.cpu()]
    net = nn.get_mtcnn(args.network)
    utils.init_net(net, args.resume, args.init, ctx)
    train_data = utils.MtcnnDataset(args.network, 'train', net.size, args.batch_size, args.num_workers)
    val_data = utils.MtcnnDataset(args.network, 'val', net.size, args.batch_size, args.num_workers)
    mtcnn_loss = nn.MtcnnLoss(args.pos_thresh, args.part_thresh, args.neg_thresh, args.ohem_ratio, args.cls_weight, args.loc_weight, args.loc_loss)
    trainer = mx.gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': args.lr, 'wd': args.wd, 'momentum': args.momentum})
    lr_steps = sorted([float(ls) for ls in args.lr_decay_epoch.split(',') if ls.strip()])
    ploter = utils.TrainingHistory(['val_acc', 'val_mse', 'train_acc', 'cls_loss', 'loc_loss', 'sum_loss', 'learning_rate'])
    
    # set up train and val metrics
    sum_metric = mx.metric.Loss('sum_loss')
    cls_metric = mx.metric.Loss('cls_loss')
    loc_metric = mx.metric.Loss('loc_loss')
    acc_metric = nn.Accuracy(args.pos_thresh, args.neg_thresh)
    val_acc_metric = nn.Accuracy(args.pos_thresh, args.neg_thresh)
    val_mse_metric = nn.LocMSE(args.part_thresh)
    
    maps = []
    logger.info('Start training from [Epoch {}]'.format(args.start_epoch))
    for epoch in range(args.start_epoch, args.epochs+1):
        while lr_steps and epoch >= lr_steps[0]:
            new_lr = trainer.learning_rate * args.lr_decay
            lr_steps.pop(0)
            trainer.set_learning_rate(new_lr)
            logger.info("[Epoch {}] Set learning rate to {}".format(epoch, new_lr))
        sum_metric.reset()
        cls_metric.reset()
        loc_metric.reset()
        acc_metric.reset()
        train_data.reset()
        tic = time.time()
        btic = time.time()
        net.hybridize()
        # logger.info('1: {:.3f}'.format(time.time()-tic)) #########
        for i, batch in enumerate(train_data):
            # logger.info('2 {:.3f}'.format(time.time()-tic)) #########
            batch_size = batch.label[0].shape[0]
            data = split_and_load(batch.data[0], ctx_list=ctx, batch_axis=0)
            labels = split_and_load(batch.label[0], ctx_list=ctx, batch_axis=0)
            # logger.info('3 {:.3f}'.format(time.time()-tic)) #########
            with mx.autograd.record():
                cls_preds = []
                loc_preds = []
                for x in data:
                    cls_pred, loc_pred = net(x)
                    cls_preds.append(cls_pred)
                    loc_preds.append(loc_pred)
                sum_losses, cls_losses, loc_losses = mtcnn_loss(cls_preds, loc_preds, labels)
                mx.autograd.backward(sum_losses)
            # since we have already normalized the loss, we don't want to normalize
            # by batch-size anymore
            trainer.step(1)
            sum_metric.update(0, [l * batch_size for l in sum_losses])
            cls_metric.update(0, [l * batch_size for l in cls_losses])
            loc_metric.update(0, [l * batch_size for l in loc_losses])
            acc_metric.update(labels, cls_preds)
            if args.log_interval and not (i + 1) % args.log_interval:
                msg = nn.metric_msg(sum_metric, cls_metric, loc_metric, acc_metric)
                logger.info('[Epoch {}][Batch {}], Speed: {:.3f} samples/sec{}'.format(
                    epoch, i, batch_size/(time.time()-btic), msg))
            btic = time.time()

        msg = nn.metric_msg(sum_metric, cls_metric, loc_metric, acc_metric)
        logger.info('[Epoch {}] Training cost: {:.3f}{}'.format(epoch, (time.time()-tic), msg))
        
        if epoch % args.val_interval == 0:
            vtic = time.time()
            val_msg = validate(net, val_data, ctx, val_acc_metric, val_mse_metric)
            logger.info('[Epoch {}] Validation: {:.3f}{}'.format(epoch, (time.time()-vtic), val_msg))
            val_acc = val_acc_metric.get()[1]
            val_mse = val_mse_metric.get()[1]
            train_acc = acc_metric.get()[1]
            cls_loss = cls_metric.get()[1]
            loc_loss = loc_metric.get()[1]
            sum_loss = sum_metric.get()[1]
            maps.append([val_acc, val_mse, train_acc, cls_loss, loc_loss, sum_loss])
            ploter.update([val_acc, val_mse, train_acc, cls_loss, loc_loss, sum_loss, trainer.learning_rate])
            ploter.plot(save_path=args.save_prefix + 'train.png')
            utils.save_model(net, args.save_prefix, epoch, maps, args.epochs)
    utils.net_export(net, args.save_prefix + args.network + '.json')
    utils.fddb_eval(range(args.epochs-5, args.epochs+1), args.save_prefix, ctx[-1])

if __name__ == '__main__':
    args = parse_args()
    train(args)
