# 简介

- 时空序列预测
- 论文复现
- PyTorch

> 暂时没空看论文了，工作稳定我会继续复现工作

> 我不保证完全的正确性啊~仅供参考

# 快速开始

- models 文件夹

  在 models 目录中，每一个文件夹存储一个结构的完整模型代码，复现参照了论文中的公式、图示以及 GitHub 作者实现的代码（如果有的话）

  这些模型均假定输入的 Tensor 的 shape 为 `(batch, sequence, channel, height, width)`
  
  这里的目的是为了学习，尽可能内聚成一个个小的 Module 再组合的，应该效率很差

- util 文件夹

  - patch

    针对大尺寸数据进行 patch 分割的方法，不过这里要根据实际情况修改下，这里是针对五维数据的，如果针对四维，则参照逻辑修改下即可

  - TrainingTemplate 和 TestingTemplate

    我自己写的训练过程的模板类，一般继承重写一些方法即可

  - content_tree

    包含生成目录树的方法





