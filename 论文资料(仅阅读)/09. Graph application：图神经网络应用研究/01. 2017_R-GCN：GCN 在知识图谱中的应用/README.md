# 补充说明

本篇论文主要有两大贡献：
1. 证明了GCN可以应用于关系网络中，特别是链接预测和实体分类中；
2. 引入权值共享和系数约束的方法使得R-GCN可以应用于关系众多的网络中。

**结论：**

&emsp;&emsp;R-GCN 构建了一个编码器，并通过接入不同的层完成不同的建模问题，如接入Softmax层进行实体分类，接入解码器进行链接预测，并在相应数据集中取得了不错的成绩。总之，在R-GCN中显式建模邻域有助于恢复知识库中缺失的事实。

**相关论文解读1：**
* [R-GCN_CSDN博客](https://blog.csdn.net/afanti_1/article/details/125922855)

**相关论文解读2：**
* [R-GCN_bilibili](https://www.bilibili.com/video/BV12J411m75G/?spm_id_from=333.337.search-card.all.click&vd_source=36e948dc2acdb36d7055f879d377b529)

