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

# dataset
- v1
[train] sample: pos = 205458, part = 533325, neg = 766085, total=1504868
[train] filter: pos = 205458, part = 300000, neg = 600000, total=1105458
[val] sample: pos = 52709, part = 137647, neg = 192566, total=382922
[val] filter: pos = 52709, part = 137647, neg = 192566, total=382922

- v2
[train] sample: pos = 195415, part = 540703, neg = 767363, total=1503481
[train] filter: pos = 195415, part = 300000, neg = 600000, total=1095415
[val] sample: pos = 50587, part = 139040, neg = 193038, total=382665
[val] filter: pos = 50587, part = 139040, neg = 193038, total=382665

python tools/fddb/eval.py -s cpnet models/mtcnn/cpnet 27,28,29,30,31,32 3 300000