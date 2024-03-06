# 补充说明

&emsp;&emsp;本文作者介绍了一种简化变压器模块的实现。由于标准的变压器模块很复杂，且存在可能导致架构不稳定的缺陷。因此，在这项工作中，作者主要研究了如何简化标准变压器模块。根据信号传播理论和经验观察，作者提出了在不牺牲训练速度或性能的情况下去除多个组件的修改。总之，本文所提出的简化版 Transformer块不仅实现了与标准 Transformer 相同的训练速度和性能，还提高了15%的训练吞吐量 ，以及减少了15%的参数使用量。

**相关论文解读1：**
* [Simplifying Transfomer_知乎](https://zhuanlan.zhihu.com/p/677511164)

**相关代码实现仓库：**
* [Simplifying Transfomer_github](https://github.com/bobby-he/simplified_transformers)

**torch版simplified transformers一键调用仓库：**
* [simplified_transformers一键调用_github](https://github.com/kyegomez/SimplifiedTransformers?tab=readme-ov-file)






