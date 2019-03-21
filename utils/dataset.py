import mxnet as mx
import numpy as np

__all__ = ['CustomRecordIter', 'MtcnnDataset']

class CustomRecordIter(mx.io.DataIter):
    """A wrapper for mx.io.ImageRecordIter for custom augmentation.

    Parameters:
    -----------
    path_imgrec : str, path to the record file.
    path_imgidx : str, path to the index file.
    batch_size : int, batch size.
    data_shape : tuple, (3, height, width).
    is_train : bool, whether it's used for training.
    num_workers : int, the number of threads to do preprocessing.
    prefetch_buffer : int, maximum number of batches to prefetch.
    label_width : int, width of label.
    mean_pixels : float or list or tuple in order [RGB].
    scale : float, multiply the image with a scale value.
    kwargs : dict, more params for mx.io.ImageDetRecordIter
    """
    def __init__(self, path_imgrec, path_imgidx, batch_size, data_shape, is_train, num_workers,
                 prefetch_buffer=20, label_width=5, mean_pixels=127.5, scale=0.0078125, **kwargs):
        super(CustomRecordIter, self).__init__()
        if isinstance(mean_pixels, float):
            mean_pixels = [mean_pixels] * 3
        self.rec = mx.io.ImageRecordIter(
            path_imgrec        = path_imgrec,
            path_imgidx        = path_imgidx,
            batch_size         = batch_size,
            data_shape         = data_shape,
            shuffle            = is_train,
            label_width        = label_width,
            mean_r             = mean_pixels[0],
            mean_g             = mean_pixels[1],
            mean_b             = mean_pixels[2],
            scale              = scale,
            prefetch_buffer    = prefetch_buffer,
            preprocess_threads = num_workers,
            **kwargs)
        self._is_train = is_train
        if not self.iter_next():
            raise RuntimeError("Invalid CustomRecordIter: " + path_imgrec)
        self.reset()

    @property
    def provide_data(self):
        return self.rec.provide_data
    
    @property
    def provide_label(self):
        return self.rec.provide_label

    def reset(self):
        self.rec.reset()

    def iter_next(self):
        return self._get_batch()

    def next(self):
        if self.iter_next():
            return self._batch
        else:
            raise StopIteration

    def _get_batch(self):
        self._batch = self.rec.next()
        if self._batch:
            if self._is_train:
                # Perform custom data augmentation on data batch
                data = self._batch.data[0].asnumpy()
                label = self._batch.label[0].asnumpy()
                ctx = self._batch.data[0].context
                self._random_flip(data, label)
                self._batch.data[0] = mx.nd.array(data, ctx)
                self._batch.label[0] = mx.nd.array(label, ctx)
            return True
        else:
            return False

    def _random_flip(self, data, label):
        '''Inplace random flip data and label.'''
        for i in range(data.shape[0]):
            if np.random.choice([True, False]):
                data[i] = np.flip(data[i], axis=2)
                label[i, (1,3)] = label[i, (3,1)] * -1

def MtcnnDataset(network, subset, size, batch_size, num_workers, mean=127.5, std=0.0078125, seed=0):
    dataset = CustomRecordIter(
		path_imgrec        = 'data/widerface/mtcnn/{}_{}.rec'.format(network, subset),
        path_imgidx        = 'data/widerface/mtcnn/{}_{}.idx'.format(network, subset),
		batch_size         = batch_size,
        data_shape         = (3, size, size),
		is_train           = subset == 'train',
        num_workers        = num_workers,
        prefetch_buffer    = 20,
        label_width        = 5,
        mean_pixels        = mean,
        scale              = std,
        seed               = seed)
    return dataset
