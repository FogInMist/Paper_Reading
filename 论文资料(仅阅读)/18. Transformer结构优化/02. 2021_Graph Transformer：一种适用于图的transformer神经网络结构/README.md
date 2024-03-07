# 补充说明

&emsp;&emsp;本文提出了一种适用于任意图的transformer神经网络结构的推广方法。由于原始的transformer是建立在全连接的图上，这种结构不能很好地利用图的连通归纳偏置——arbitrary and sparsity，即把transformer推广到任意图结构，且表现较弱，因为图的拓扑结构也很重要。因此，本文提出的新graph transformer，带有以下四个新特征：1）在每个node的可连通临域做attention；2）用拉普拉斯特征向量表示positional encoding；3）用BN（batch normalization）代替LN（layer normalization），优点：训练更快，泛化性能更好；4）进一步将结构扩展到边特征表示。

**相关论文解读1：**
* [Graph Transfomer_CSDN](https://blog.csdn.net/chen_wangaa/article/details/113361075)

**相关论文解读2：**
* [Graph Transfomer_博客园](https://www.cnblogs.com/programmer-yuan/p/programmer_paper_graph-transformer1.html)

**相关代码实现仓库：**
* [Graph Transfomer_github](https://github.com/graphdeeplearning/graphtransformer)







