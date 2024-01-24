# 补充说明

&emsp;&emsp;本文提出了ProbSparse self-attention mechanism的注意力方法，在耗时和内存上都进行了有效压缩；此外，每个注意力Block之间都添加了“蒸馏”操作，通过将序列的shape减半来突出主要注意力，原始的柱状Transformer变成金字塔形的Transformer，使得模型可以接受更长的序列输入，并且可以降低内存和时间损耗；最后，设计了一个较为简单但是可以一次性输出预测值的Decoder。


**相关论文解读1：**
* [Informer_CSDN](https://blog.csdn.net/fluentn/article/details/115392229)

**相关论文解读2：**
* [Informer_知乎](https://zhuanlan.zhihu.com/p/646853438)


