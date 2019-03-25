import cv2 as cv
from nn import MtcnnV2
import mxnet as mx
from utils import plot_bbox

mtcnn = MtcnnV2('models/pnet')
image = cv.imread('data/fddb/images/2002/08/11/big/img_591.jpg', cv.IMREAD_COLOR)
image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
bbox = mtcnn.detect(image)
print(bbox.shape)
print(bbox[:5])
plot_bbox(image, bbox)
