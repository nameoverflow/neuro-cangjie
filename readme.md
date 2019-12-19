# 神经仓颉机

TL,DR: 用于分析汉字字形结构的简单神经网络，预设提供仓颉五代模型。

分析仓颉码的任务与 image caption 有一定相似性，故本代码主要基于 show, atten and tell 方法 [1]，部分代码借用于 [2]。



[1] Xu K, Ba J, Kiros R, et al. Show, Attend and Tell: Neural Image Caption Generation with Visual Attention [arxiv](https://arxiv.org/abs/1502.03044)

[2] [https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning)