# 补充说明

本文提出的General Attributed Multiplex Heterogeneous Network Embedding(GATNE)，希望每个节点在不同类型边中有不同的表示，
比如说用户A在点击商品的场景下学习一种向量表示，在购买商品的场景下学习另一种向量表示，
而不同场景之间并不完全独立，希望用base embedding来当作不同类型关系传递信息的桥梁，我们综合base embedding与每一类型边的edge embedding来进行建模。
在直推式学习(Transductive)背景下，提出GATNE-T模型，在归纳式学习(inductive)背景下，考虑节点特征，提出GATNE-I模型。

**( 本论文的目标是学习一种更加有效的节点嵌入表示方法。 )**



**相关论文解读1：**
* [GATNE_原作者说明](https://www.aminer.cn/research_report/5cf49ed300eea1f1d521d71f?download=false)

**相关论文解读2：**
* [GATNE_腾讯云](https://cloud.tencent.com/developer/inventory/531/article/1665814)
