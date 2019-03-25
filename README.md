### github

- [Seanlinx/mtcnn](https://github.com/Seanlinx/mtcnn) mxnet 4
  - [#61](https://github.com/Seanlinx/mtcnn/issues/61): train pnet slow: 1.ssd, 2.imdb, 3.delete all `out_grad=True` in core\symbol.py 4. 等几分钟就会快起来
  - [#57](https://github.com/Seanlinx/mtcnn/issues/1): roc compare 1~2% lower

  <img src="https://github.com/Seanlinx/mtcnn/raw/master/fddb_result.png" width="500" />

- [zuoqing1988/train-mtcnn](https://github.com/zuoqing1988/train-mtcnn) mxnet-win 借鉴自 Seanlinx/mtcnn 改进可以借鉴
  - [#5](https://github.com/zuoqing1988/train-mtcnn/issues/5): DiscROC 准确率。

- [AITTSMD/MTCNN-Tensorflow](https://github.com/AITTSMD/MTCNN-Tensorflow) tensorflow 3.5
  - [#6](https://github.com/AITTSMD/MTCNN-Tensorflow/issues/6): Pnet 准确率问题

<img src="https://camo.githubusercontent.com/52bd155eb111c221923d47daeb21886416bb6179/68747470733a2f2f692e6c6f6c692e6e65742f323031372f30382f33302f353961366238373566313739322e706e67" width="500" />

- [foreverYoungGitHub/MTCNN](https://github.com/foreverYoungGitHub/MTCNN) caffe 3
- [CongWeilin/mtcnn-caffe](https://github.com/CongWeilin/mtcnn-caffe) caffe 借鉴自 foreverYoungGitHub/MTCNN imdb 加速可以借鉴 

- [wujiyang/MTCNN_TRAIN](https://github.com/wujiyang/MTCNN_TRAIN) pytorch 2

### blog

- [mtcnn 训练日志](https://joshua19881228.github.io/2018-09-11-training-mtcnn/): 主要在 oreverYoungGitHub/MTCNN 上做的尝试

### paper

- [Anchor Cascade for Efficient Face Detection](https://arxiv.org/pdf/1805.03363.pdf)
  Anchor overlap

### Todo

- 生成更多数据： range(more)
- traininghistory: finalize() to dump plot data
- caffe pnet fddb vs mxnet pnet fddb

- default:
  - settings: max-40 [50, 5, 20]  3:1:1
  - train: 984792  (598251  199356  199464)
  - val: 254360  (154425  51511  51536) 
- v1: 4:1:1
- v2: 很低 (0.18, 0.9) [50, 10, 20]  min(w, h) < 25 or max(w, h) < 30
  - train: 2057290  (1102561  484812  510866)
  - val: 523244  (279235  123882  132353)

  celeba 数据集 生成负样本


  - mxnet-mtcnn: 
    - Sample: 12880 images done, pos: 199475 part: 548912 neg: 812070
    - Choose: total 1099474 (pos: 199475 part: 300000 neg: 600000)
# only neg
[sample] pos = 194346, part = 541767, neg = 767069, total=1503182
[filter] pos = 194346, part = 300000, neg = 600000, total=1094346


mxnet  299999  0.963  0.708
caffe  299999  0.970  0.730

# 2019.3.25
100% 12863/12863 [04:05<00:00, 52.41 annos/s]
[train] sample: pos = 284704, part = 749760, neg = 819079, total=1853543
[train] filter: pos = 284704, part = 300000, neg = 819079, total=1403783
100% 12863/12863 [07:38<00:00, 28.04 annos/s]
100% 3218/3218 [01:32<00:00, 34.76 annos/s]
[val] sample: pos = 73094, part = 191622, neg = 206715, total=471431
[val] filter: pos = 73094, part = 191622, neg = 206715, total=471431
100% 3218/3218 [02:00<00:00, 37.79 annos/s]