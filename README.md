# DCEIFlow_with_DSEC

## 摘要 Abstract

本项目为记录模式识别课程设计而创立。项目的建立依赖于以下参考论文与项目：

This project was created to document pattern recognition curriculum design. The establishment of the project relies on the following reference papers and projects:

[M. Gehrig, M. Millhäusler, D. Gehrig, D. Scaramuzza. E-RAFT: Dense Optical Flow from Event Cameras. International Conference on 3D Vision (3DV), 2021.](http://rpg.ifi.uzh.ch/ERAFT.html)

[Zhexiong Wan, Yuchao Dai, Yuxin Mao. Learning Dense and Continuous Optical Flow from an Event Camera. IEEE Transactions on Image Processing, 2022.](https://npucvr.github.io/DCEIFlow/)

在此感谢上述论文的开源与思路。

Thanks for the open source and ideas of the above paper.

## 项目介绍 Project introduction

本项目基于E-RAFT项目，进行了DCEIFlow与DSEC的适配，从而实现了DCEIFlow的预训练模型在DSEC数据集上的测试与可视化。

Based on the E-RAFT project, this project adapted DCEIFlow and DSEC, so as to realize the testing and visualization of the pre-training model of DCEIFlow on the DSEC data set.

至于训练还有待研究。

Training remains to be studied.

## 环境要求 Requirements

该代码已在PyTorch 1.12.1和Cuda 11.7上进行了测试。

The code has been tested with PyTorch 1.12.1 and Cuda 11.7.

## 预训练模型 Pretrained Weights

预训练的权重可以从[Google Drive]下载，请将它们放入 `checkpoint`文件夹中。

Pretrained weights can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1Dh7BqXozY59SZKOgVj7_yZ5d09R8qilw?usp=share_link). Please put them into the `checkpoints` folder.

```
checkpoints
├── DCEIFlow.pth
```

## 测试 Evaluation

要进行测试，你必须下载数据集。示例数据如下：
To evaluate our model, you need first download the files version of [DSEC](https://dsec.ifi.uzh.ch/dsec-datasets/download/) datasets. Example data is as follows:

```
data/DSEC/test
├── thun_00_a
│   ├── events_left
│   │	├── events.h5
│   │	├── rectify_map.h5
│   ├── flow
│   │   ├── forward
│   │   │   ├── 000002.png
│   │   │   ├── 000004.png
│   │   │   ├── ...
│   │   │   ├── 000082.png
│   ├── image_timestamps.txt
│   ├── test_forward_flow_timestamps.csv
```

配置好环境，下载预训练权值后，执行如下命令，得到测试结果。工作区文件夹中会生成 `saved`文件夹，里面存放着一些相关可视化数据。

After the environment is configured and the pretrained weights is downloaded, run the following command to get the consistent results as reported in the paper. A `saved` folder is generated in the workspace folder, which contains some relevant visual data.

```
python main.py --path ./data/DSEC --type standard --visualize True
```
