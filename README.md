# 版图热点检测算法实现
## 问题背景
在当前光刻条件下，版图（Layout）上的某些图案，即使进行后期处
理（如掩膜版优化等），光刻后仍然可能发生畸变。这些缺陷被称为
热点（Hotspot）。现有热点检测框架从架构到算法层面均存在不足。
本题目标是设计一种热点检测算法。该算法能够在合理的运行时间
内有效地检测出含有热点的版图，从而减少制造成本、缩短周转时
间。
## 数据集描述
本题中版图已被像素化，以图像形式出现。相关数据集分为训练集
合、测试集合。版图标签通过读取文件名首字母获得，“N”对应无
热点版图，“H”对应热点版图。
## 评估指标
1. 热点预测准确率（Accuracy）：正确预测的热点占总热点的比例，
数值为百分比，数值越高，算法性能越好。
2. 误报（False Alarm）：错误的将非热点版图预测为热点版图的版图
个数，数值越低，算法性能越好。
## 参考文献
[1] H. Yang, J. Su, Y. Zou, Y. Ma, B. Yu and E. F. Y. Young, "Layout Hotspot Detection With 
Feature Tensor Generation and Deep Biased Learning," in IEEE Transactions on Computer-Aided 
Design of Integrated Circuits and Systems, vol. 38, no. 6, pp. 1175-1187, June 2019, doi: 
10.1109/TCAD.2018.2837078. (有 Github Repo)
[2] Y. Jiang, F. Yang, H. Zhu, B. Yu, D. Zhou and X. Zeng, "Efficient Layout Hotspot Detection 
via Binarized Residual Neural Network," 2019 56th ACM/IEEE Design Automation Conference 
(DAC), Las Vegas, NV, USA, 2019, pp. 1-6.
[3] H. Geng, H. Yang, L. Zhang, F. Yang, X. Zeng and B. Yu, "Hotspot Detection via AttentionBased Deep Layout Metric Learning," in IEEE Transactions on Computer-Aided Design of 
Integrated Circuits and Systems, vol. 41, no. 8, pp. 2685-2698, Aug. 2022, doi: 
10.1109/TCAD.2021.3112637.
[4] S. Sun, Y. Jiang, F. Yang, B. Yu and X. Zeng, "Efficient Hotspot Detection via Graph Neural 
Network," 2022 Design, Automation & Test in Europe Conference & Exhibition (DATE), Antwerp, 
Belgium, 2022, pp. 1233-1238, doi: 10.23919/DATE54114.2022.9774579
