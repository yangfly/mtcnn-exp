import cv2 as cv
from nn import Mtcnn
import mxnet as mx
from utils import plot_bbox

mtcnn = Mtcnn('models/pnet', ctx=mx.cpu(), thresholds=[0.6])
image = cv.imread('data/test.jpg', cv.IMREAD_COLOR)
image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
bbox = mtcnn.detect(image)
plot_bbox(image, bbox)
