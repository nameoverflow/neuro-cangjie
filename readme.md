# 神经仓颉机

TL,DR: 用于分析汉字字形结构的简单神经网络，预设提供仓颉五代模型。

分析仓颉码的任务与 image caption 有一定相似性，故本代码主要基于 show, atten and tell 方法 [1]，部分代码借用于 [2]。

需要 Python >= 3.7，其它详细依赖见 `requirements.txt` 或 `environment.yaml`。

结果示例：

![](https://github.com/nameoverflow/neuro-cangjie/raw/master/img/example1.png)
![](https://github.com/nameoverflow/neuro-cangjie/raw/master/img/example2.png)

## 预测

执行 `inference.py` 使用预训练模型进行预测（预训练模型见 releases）。预训练模型使用花园明朝字形，需要将 `HanaMinA.ttf`、`HanaMinB.ttf` 放置在 `data/hanazono` 下。

```shell
python inference.py --model data/cangjie5.pth
```

程序将进入命令行交互界面：

```plain
>> 拉
qyt
```

并将可视化结果保存至 `result.png`。

如果需要使用 CPU 进行计算：

```shell
python inference.py --model data/cangjie5.pth --use_cpu
```

其它命令行参数详见 `--help`。


## 训练

见 `python train.py --help`。默认配置大约需要 5GB 显存，训练花费约 10 小时。

默认使用的仓五码表来自 [Jackchows/Cangjie5](https://github.com/Jackchows/Cangjie5)，去除了所有 X 与 Z 开头的编码。训练时随机 7:3 划分训练集与验证集。

训练进程：

![](https://github.com/nameoverflow/neuro-cangjie/raw/master/img/trainplot.png)

---
## 参考

[1] Xu K, Ba J, Kiros R, et al. Show, Attend and Tell: Neural Image Caption Generation with Visual Attention [arxiv](https://arxiv.org/abs/1502.03044)

[2] [https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning)