# 补充说明

&emsp;&emsp;本文设计的AGCN模型由多个连续的层组合构成，其核心层为SGC-LL层。层组合包括一个“SGC-LL层”，“一个批处理归一层”和一个“图最大池化层”。我们在每个SGC-LL层中训练一个残差图Laplacian(拉普拉斯矩阵)，并在接着一层最大图池化层。而由于SGC-LL层会变换特征，所以对于下一个SGC-LL层我们需要重新训练一个新的残差图Laplacian。

&emsp;&emsp;而在经过上述组合层（SGC-LL层，批处理归一层，图最大池化层）后，批处理的图结构将会被更新，但图的大小会保持不变。此外，如果对网络进行图预测任务，则图聚合层将是最后一层。

**相关论文解读1：**
* [AGCN_CSDN博客](https://blog.csdn.net/weixin_43450885/article/details/106554192)

**相关论文解读2：**
* [AGCN_知乎](https://zhuanlan.zhihu.com/p/359597251)

