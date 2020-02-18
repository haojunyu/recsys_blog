---
title: "推荐算法深入浅出之逻辑回归"
date: 2019-12-06T22:00:47+08:00
lastmod: 2020-02-08T17:54:04+08:00
draft: false
tags: ["推荐算法","逻辑回归"]
categories: ["推荐系统"]
author: "禹过留声"

---

谈及机器学习或推荐算法，不得不提入门级的逻辑回归算法。但我们真的掌握逻辑回归了么？不妨回答一下以下问题：
1. 逻辑回归的代价函数以及公式推导？
2. 批量梯度下降和随机梯度下降的差异？
3. 样本太大，导致特征编码耗时太长怎么办？
4. 如何优化随机梯度下降使得算法又快又准？

如果你对上述问题心里没底不妨读下这篇文章。本文分为四个部分，第一部分介绍逻辑回归算法的推导过程，以便理解算法背后的理论基础；第二部分介绍逻辑回归的实现细节 ，包含特征散列的技巧以及学习率自适应，使得算法能够支撑更大的数据集。第三部分简单的安利一波逻辑回归的工业级实现工具 Vowpal Wabbit，最后一部分分享一些在学习逻辑回归过程中的思考。
<!--more-->


##  逻辑回归的理论推导
这部分的理论推导需要一些数学中的微积分和统计学的知识。我们分为三步，由浅入深的介绍。先从简单的一元线性回归入手（高中数学知识），再扩展到多元线性回归，最后延伸到逻辑回归。
### 一元线性回归
在高中时期，我们经常会碰到这样一道数学题：某小卖部为了了解热茶的销售量与气温之间的关系，随机统计了 6 天卖出热茶的杯数与当天气温的对照表(如下表)，如果某天的气温是 -5℃，你能根据这些数据预测这天小卖部卖出热茶的杯数吗？

|   序号    |   1   |   2   |   3   |   4   |   5   |   6   |
| :-------: | :---: | :---: | :---: | :---: | :---: | :---: |
| 气温（℃） |  26   |  18   |  13   |  10   |   4   |  -1   |
|   杯数    |  20   |  24   |  34   |  38   |  50   |  64   |

这是一道典型的求线性回归方程的问题。老师指导我们按照以下三步来解题：
1. 画出气温 $\mathbf{x}=[x_1;x_2;\dots;x_n]$ 和杯数 $\mathbf{y}=[y_1;y_2;\dots;y_n]$ 的散点图（ $\mathbf{x}$ 和 $\mathbf{y}$ 为 $n*1$ 的向量）。判定二者是否具有线性相关关系（下图中蓝点的散点图明显是线性相关关系，即能用 $\hat{y}=ax+b$ 近似表示，而且是负相关）。
![热茶杯数与当天气温的对照表](./1547266712307.png)
2. 按照如下公式求解线性回归方程中的两个参数：斜率 $a=-1.6477$ 和截距 $b=57.5568$

$$
\begin{eqnarray*}
a& =&\frac{n\sum_{i=1}^{n}x_iy_i - (\sum_{i=1}^{n}x_i)(\sum_{i=1}^{n}y_i)}{n\sum_{i=1}^{n}x_i^2-(\sum_{i=1}^{n}x_i)^2}  \\\\
b&=&\frac{\sum_{i=1}^ny_i}{n} - a\frac{\sum_{i=1}^nx_i}{n}
\end{eqnarray*} \tag{2.1}
$$

1. 根据线性回归方程 $\hat{y}=-1.6477x+57.5568$ 和 $x=-5$ 求解出小卖部能卖出66杯。

三步流程走完，基本这题已经答完了。但是对于斜率 $a$ 和截取 $b$ 的公式由来我们不一定知道。这里简要提一下，具体的推导过程见文末。本题的目的是构造线性回归方程 $\hat{y}=ax+b$ 来拟合观察到 $\mathbf{x},\mathbf{y}$ 数据，也就是在上图中找一条直线，使得所有样本到直线的欧氏距离和最小。
> **TIPS:**
> 欧式距离是指 m 维空间中两个点之间的真实距离。这里其实就是观察点 $(x_i,y_i)$ 和直线上的点 $(x_i,ax_i+b)$ 的距离，也就是 $|ax_i+b-y_i|=\sqrt{(ax_i+b-y_i)^2}$ 的值。注意这里并不是观察点到直线的垂直距离。

所以最终的问题就转换成求解 $a,b$ 使得代价函数 $Cost(a,b)=\sum_{i=1}^n (\hat{y}-y_i)^2=\sum_{i=1}^n(ax_i+b-y_i)^2$ 的取值最小。

### 多元线性回归
我们知道数学题目是对现实问题的一种简化，就像上述小卖部卖热水的问题。现实中卖出的热水杯数不仅和当天的气温温度有关，可能还有热水的温度，是否加茶叶等等。也就是影响小卖部卖出热水杯数 $y$ 由多个因素 $x_1,\dots,x_n$ 决定，其中 $x_1$ 表示当天气温，$x_2$ 表示热水温度，$x_3$ 表示是否加茶叶等。此时该问题就由一元线性回归问题变成了多元线性回归问题。其目的和之前一样：构造线性回归方程 $\hat{y}=\sum_{i=1}^n a_ix_i+b$ 来拟合观察到 $\mathbf{X},\mathbf{y}$ 数据，也就是在 $m+1$ 维空间中找一个判定界面，使得所有样本到判定界面的欧氏距离和最小。
这里为了方便，将参数 $a_1,\dots,a_m,b$ 合并成向量形式 $\mathbf{w}=[a_1;\dots;a_m;b]$，此时线性回归方程 $\hat{y}=\mathbf{w}^T\mathbf{x}$，观察样本 $\mathbf{X}$和$\mathbf{y}$ 如下公式，其中观察样本数目有 n 个，影响因素有 m 个。

$$
\begin{eqnarray*}
\mathbf{X} &=&\left[ 
\begin{matrix}
 x_{11}      &  x_{12}      & \cdots &  x_{1m}    &   1  \\\\
 x_{21}      &  x_{22}      & \cdots &  x_{2m}    &   1  \\\\
 \vdots & \vdots & \ddots & \vdots  & \vdots\\\\
 x_{n1}      &  x_{n2}      & \cdots &  x_{nm}    &   1  \\\\
\end{matrix}
\right] =\left[\begin{matrix}
\mathbf{x_1}^T       &   1  \\\\
\mathbf{x_2}^T       &   1  \\\\
 \vdots     & \vdots\\\\
\mathbf{x_n}^T    &   1  \\\\
\end{matrix}
\right]  \\\\
\mathbf{y}&=&\left[\begin{matrix}
y_1 \\\\
y_2 \\\\
\vdots \\\\
y_n \\\\
\end{matrix}\right]=\left[\begin{matrix}
\mathbf{w}^T*[\mathbf{x_1};1] \\\\
\mathbf{w}^T*[\mathbf{x_2};1] \\\\
\vdots \\\\
\mathbf{w}^T*[\mathbf{x_n};1] \\\\
\end{matrix}\right] = \mathbf{w}^T\mathbf{X}
\end{eqnarray*}\tag{2.2}
$$

对于多元线性回归问题的求解就变成了找到参数 $\mathbf{w}$ 使得代价函数 $Cost(\mathbf{w})=(\mathbf{y}-\mathbf{X}\mathbf{w})^T(\mathbf{y}-\mathbf{X}\mathbf{w})$ 取值最小。对代价函数的导函数置零得到 $\mathbf{w}=(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$ （具体推导见文末）。其中 $(\mathbf{X}^T\mathbf{X})^{-1}$ 表示矩阵 $(\mathbf{X}^T\mathbf{X})$ 的逆矩阵，而现实中该矩阵不一定可逆，比如样本数小于特征数（即 $n<m$ ），就像求解 n 个一阶变量的方程组，但是方程组的变量个数 m 不足 n，这样就会有无数 $\mathbf{w}$ 解，此时常引入正则化项L1($||\mathbf{w}||$）或L2（$||\mathbf{w}||^2$）使得参数 $\mathbf{w}$ 中的各个权重 $w_i$ 越小，使得多元线性回归方程越简单，更能有效的描绘观察数据。这就是正则化项能防止过拟合的原因。
> **过拟合vs欠拟合:**
> 过拟合是指模型把数据学习的太彻底，以至于把噪声数据的特征也学到了。体现在建模过程是训练时误差很小，测试的时候误差很大。通俗的讲就是泛化能力弱。一般有如下解决方案：
> 1. 重新清洗数据，清除噪声特征
> 2. 增大数据的训练量
> 3. 采用正则化方法
> 4. 采用 dropout 方法。该方法在神经网络中常用，就是在训练是让神经元以一定的概率不工作
>  
> 欠拟合是指模型没有很好的捕捉到数据的特征，不能很好的拟合数据。比如散点图显示数据是非线性的关系，如果此时还是用线性方程去拟合的话，就会造成欠拟合的情况。一般针对欠拟合有如下解决方案：
> 1. 添加其他特征项，比如特征交叉
> 2. 添加多项式特征，比如添加某个特征二次项，三次项等
> 3. 减少正则化参数。

### 逻辑回归
逻辑回归全称叫对数几率回归，英文为 `logistic regression` 或 `logit regression` 。之所以逻辑回归叫对数几率回归，是因为逻辑回归是对数几率函数和线性回归的强强组合。对数几率函数（logistic function） $logistic(z)=\frac{1}{1+e^{-z}}$ 是一种常用的“Sigmoid函数”，能够将 $z$ 值转换成一个接近0或1的值，其曲线见下图。所以逻辑回归并不是回归模型而是分类模型，常用来解决二分类的问题，预估某个事件发生的概率。
![对数几率函数](./logistic.jpg)

> **回归vs分类:**
> 回归模型和分类模型本质是一样的，差异在于模型的输出，回归模型输出的是连续数据，而分类模型输出的是离散数据。当然这样的理解：分类模型就是将回归模型的输出离散化也是合理的。比如线性回归模型与逻辑回归模型，支持向量回归（SVR）和支持向量机（SVM）等。

对于给定一组信号向量 $\mathbf{x_i}=[x_{i1},x_{i2},\cdots,x_{im}]$，也就是影响 $y $ 的多个自变量 $x_i $，逻辑回归模型会先对该组信号向量进行线性组合 $\mathbf{w}^{T}\mathbf{x_i} $,其中 $\mathbf{w}=[w_1;w_2;\cdots,w_m]$ 表示对应信号的权重，然后将线性组合的结果通过对数几率函数 logistic 映射到 $[0,1]$ 区间，也就是下面的表达式：
$$
y=h_{\mathbf{w}}(\mathbf{x}_i)=\operatorname{logistic}\left(\mathbf{w}^{T} \mathbf{x}_i\right)=\frac{1}{1+e^{-\mathbf{w}^{T}\mathbf{x_i}}} 
$$
这里 $h_{\mathbf{w}}(\mathbf{x}_i)$ 也可以解释为在输入信号量 $\mathbf{x}_i$ 的条件下 $y=1$ 的条件概率，也就是 $h_{\mathbf{w}}(\mathbf{x}_i)=p(y=1 | \mathbf{x}_i)$，如果想要用这个概率值来预测 $y=1$ 或者 $y=0$，则需要选择一个阈值，比如阈值取 0.5。
> **TIPS:**
> 我们说逻辑回归是一个线性分类器，是因为它的决策边界是以输入信号的线性组合。举例来说，当阈值取 0.5 时，模型判断一组信号 $ \mathbf{x}\_i $ 输入为正例$y=1$，等价于 $h_{\mathbf{w}}(\mathbf{x}_i)>=0.5$，等价于 $\mathbf{w}^{T} \mathbf{x}_i>=0$。可以看出这个判定边界是线性的。

对于逻辑回归模型的求解，也就是信号权重 $\mathbf{w}$，这个无法使用和线性回归模型相同的最小均方误差 $MSE$，因为根据最小均方误差计算的代价函数是非凸的，所以可能在寻优时容易陷入局部最优，也就是无法找到权重 $\mathbf{w}$ 使得代价函数 $Cost(\mathbf{w})$ 全局最小。 
> **TIPS**
> 对于凸函数而言有一个非常重要的性质：局部最小值就是全局最小值。这就是构建的代价函数要是凸函数的原因，而且这样也方便使用梯度下降算法求解全局最优解。

这里我们使用最大似然估计(这里也可以用信息熵)构建如下代价函数
$$
\begin{eqnarray*}
Cost(\mathbf{w})&=& \sum_{i=1}^{N} \left[y_{i} \log p(y=1|\mathbf{x}_i) + (1-y_i)\log p(y=0|\mathbf{x}_i) \right]\\\\
&=& \sum_{i=1}^{N}\left[y_{i} \log \left(h_{\mathbf{w}}(\mathbf{x}_i)\right)+\left(1-y_{i}\right) \log \left(1-h_{\mathbf{w}}(\mathbf{x}_i)\right)\right]\\\\
\end{eqnarray*} \tag{2.3}
$$
该代价函数也叫做对数损失（这里 log 的底数默认为 e）。因为该函数是凸函数([证明过程][convex_prof])，我们可以使用梯度下降算法求解最优解。根据代价函数 $Cost(\mathbf{w})$ 的偏导数不断迭代更新即可求解出代价函数取得最优解时的权重 $\mathbf{w}^*$（具体的偏导数推导过程见文末）。
$$
\frac{\partial Cost(\mathbf{w})}{\partial \mathbf{w}}  =  \sum_{i=1}^{N} \mathbf{x}_i\left(h_{\mathbf{w}}(\mathbf{x}_i)-y_i\right)\\\\
\mathbf{w}^*=\mathbf{w} - \alpha \frac{\partial Cost(\mathbf{w})}{\partial \mathbf{w}} \\\\
w_{j+1} = w_j - \alpha \sum_{i=1}^{N} x_{ij}\left(h_{\mathbf{w}}(\mathbf{x}_i)-y_i\right)\tag{2.4}
$$
从最后一个表达式我们可以看出要求解的权重向量 $\mathbf{w}$ 中的值 $w_j$ 和输入向量 $\mathbf{x_i}$ 中的 $x_{ij}$ 相关。在多元线性回归中，权重 $w_j$ 的解释是相关的信号量 $x_{ij}$ 每增加一个单位，预测的结果将会增加 $w_j$。而在逻辑回归模型中，权重 $w_j$ 的解释是相关信号量 $x_{ij}$ 每增加一个单位，它将使结果为1的事件发生比加 $(e^{w_j}-1)*100\%$ 倍。因为 $\log \left( \frac{p(y=1|\mathbf{x}_i) }{1-p(y=1|\mathbf{x}_i) } \right)=w_0+w_1x_{i1}+\cdots+w_mx_{im}$。
>  **TIPS:**
>  在统计和概率理论中，一个事件的发生比是指该事件发生和不发生的比率，公式就是 $\frac{p}{1-p}$ 。英文单词为 odds，也叫几率。比如用户点击物品的概率是 0.8，那么不点击的概率为 0.2，发生比就是0.8:0.2=4:1。
>  对数几率就是对几率取对数 $\log \frac{p}{1-p}=C$，其中 $p$ 解开来就是 $p=\frac{1}{1+e^{-C}}$


### 优化算法
在一元线性回归问题中，我们只有一个自变量，而观察的数据肯定大于自变量的个数，所以我们始终能够求解出结果。而在多元线性回归问题中，虽然我们通过置代价函数的偏导数为0求得$\mathbf{w}=(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$，但是现实中很可能$(\mathbf{X}^T\mathbf{X})$是不可逆的。还有在逻辑回归中无法通过代价函数的偏导数为0的方式求解$\mathbf{w}$。那有没有其他通用的方法来解决这样的求代价函数最小的问题呢。有，它就是梯度下降算法（Gradient Descent）。[维基百科][gradient_descent]上对梯度下降算法的解释是如果实值函数$F(x)$在点a处可微且有定义，那么函数$F(x)$在a点沿着梯度相反的方向$-\nabla F(\mathbf{a})$下降最快。这里以登山者下山来解释梯度下降算法：登山者如果想早点下山最简单的方式就是脚下哪个方向越陡峭就往哪个方向走，每走一步都要修正一下方向。这里也学维基百科使用如下图的等高线图来解释批量梯度下降，随机梯度下降以及自适应的随机梯度下降算法的区别。
![梯度下降算法间的比较](./gradient_descent.jpg)
图中黄色的路线表示批量梯度下降，蓝色的路线表示随机梯度下降，绿色的路线表示自适应的随机梯度下降算法。

#### 批量梯度下降`Batch Gradient Descent`
批量梯度下降的迭代公式如下：
$$
w_{j+1} = w_j - \alpha \sum_{i=1}^{N} x_{ij}\left(h_{\mathbf{w}}(\mathbf{x}_i)-y_i\right) \tag{2.5}
$$
在每一次的迭代，也就是登山者每次确定方向的时候，批量梯度下降算法会使用全部的训练样本，从而得到一个当前能最快到最优解的地方，也就是登山者能当前能最快下山的方向。接着按照确定的步长$\alpha$往前走一步。图中黄色路线表示了批量梯度下降算法的迭代过程。该算法虽然每步都能得到当前最优方向（图中是等高线法线方向），但是因为每步使用全量数据（样本数目可能非常大，大到内存放不下），将会导致每一步迭代需要的运算时间非常长。

#### 随机梯度下降`Stochastic Gradient Descent`
随机梯度下降的迭代公式如下：
$$
w_{j+1} = w_j - \alpha * x_{ij}\left(h_{\mathbf{w}}(\mathbf{x}_i)-y_i\right) \tag{2.6}
$$
在每一次的迭代时候，随机梯度下降算法只通过一个样本就完成了方向的确定，接着按照确定的步长$\alpha$往前走一步。图中蓝色路线表示了随机梯度下降算法的迭代过程。该算法虽然每步的方向都不一定是当前最优方向（最优方向是等高线法线方向），但是因为每步使用很少的样本数据，使得权重$\mathbf{w}$能够快速的迭代到最优解。但是它也是有缺点的，就是在最优解附近会存在明显的锯齿震荡现象，即损失函数的值会在最优解附近上下震荡一段时间才最终收敛。该现象发生的原因一方面因为确定的步长导致容易错过最优解，有种乘坐飞机跨越了两站公交车站距离的感觉；另一方面是因为每一步在确定方向的时候并不一定是最优解的方向，有种打高尔夫球一直在洞边徘徊的感觉。

图中还有一条绿色的线路表示自适应随机梯度下降算法。之所以有自适应这步的优化是因为批量梯度下降算法和随机梯度下降算法都是使用确定的步长，这就导致了初始时该步长太小，要走好几步才有明显的代价损失减少；而靠近最优解时步长太长，经常会错过最优解。自适应算法是为了让梯度下降算法在开始时大步向前迈，在靠近最优解时采用小碎步。至于如何实现自适应将在下章实现技巧中介绍。

## LR的实现技巧
俗话说得好：理想很丰满，现实很骨感。在掌握了上述关于逻辑回归算法的理论基础以及权重求解后，我们的确可以构建一个逻辑回归模型来解决实际问题。但是要想使模型更加健壮高效，还需要学习两个技巧，一个是将散列技巧用在信号表示上，这样可以为前期的预处理节省非常多的时间；另一个是使用自适应步长来加快收敛以及降低因为步长太大导致在最优解附近锯齿震荡的现象。
### 散列
在我们预处理逻辑回归的数据时经常采用one-hot序列编码的方式，简单来说就是将训练样本中出现过的每个特征按照顺序递增编号。这样有个非常明显的问题就是训练时要生成一个特征名到特征序号的映射表，并且要将这个映射表传递给预测阶段的特征预处理。这个映射表在数据样本很大或者数据结构设计不好时就会非常浪费时间。其次有些频次很低的特征置信度不高，特别像一些异常数据（用户年龄100多岁等），单独出现对模型无益。而通过散列的方式，只要训练和预测时的散列算法一致，那么对于同一特征的编码结果也将一致，也就是数据一致性以及预处理的速度将大大提升，特别是针对极大规模特征和在线学习的情况。

那么怎么使用散列呢？比较常见的用法是将特征名散列到int64的整数，然后再取模m。大家可能会担心散列可能会发生碰撞，而模m后使得碰撞的概率更大了。这里要以vowpal wabbit里举例，它有一个参数`-b`代表掩码mask的位数，默认取18，最大到61。每个特征号的计算方式是 $feat_{id} = hash(feat_{name}) \\& mask, mask \in [1,2^{61}]$。该掩码$mask$代表了最多包含的特征数目，上限非常高。所以散列方法能够在尽量不损失原始特征的表达能力的前提下，把原始高维特征向量压缩成较低维的特征向量。从这点上来看，散列技巧有点介于one-hot编码和enbedding之间。而关于它能够尽量多的保留原始特征的表达能力在[这篇论文][collision_prof]的4.4章节里有论述。

相较于one-hot序列编码，散列技巧不需要预先维护一个特征映射表，使在线学习更方便，而且大大降低了模型训练的时间和空间上的复杂度。对于它的不足也有一些针对的”补丁”。比如为解决散列的冲突引入了命名空间的概念，在不同命名空间中的特征序号不会发生冲突；还将特征空间的大小作为模型参数，通过观察损失函数来调节该参数。还比如为解决原特征反推难的问题，vowpal wabbit中生成命令`vw-varinfo `来展示所有的原特征，散列值，取值范围，权重等模型信息。

### 自适应
自适应梯度下降算法在上一章节介绍过，并且用绿色的线路演示了算法的优化过程。其关键就是对步长$\alpha$的自适应，也就是算法观察的训练数据越多，自适应策略要使步长越小，这样才能让算法达到初期快速收敛，后期缓慢接近接近全局最优解的目的。

从梯度下降的迭代公式中我们可以看出每个模型参数，也就是权重$w_j$只和对应维度的信号量$x_{ij}$相关，与其他维度无关。而我们期望的步长，也就是学习率$\alpha$是越来越小的。所以[谷歌的这篇论文][adapt_learning]以及vowpal wabbit都采用如下的学习率：
$$
\alpha_j = \frac{\alpha}{\sqrt{n_j}+1} \tag{3.1}
$$
其中$\alpha$表示初始的学习率，$n_j$表示从训练开始特征$x_{ij}$出现的次数。在随机梯度下降算法中，每处理一个样本，权重$\mathbf{w}$都会更新一次，也就是登山者下山走的一步。我们假定初始的学习率 $\alpha=0.1$，在处理10000个样本后，特征“年龄=100岁”在这10000个样本中只出现了2次，而特征“年龄=30岁”在这10000个样本中出现了300次。那么，在处理第10001个样本时，特征“年龄=100岁”的自适应学习率是$0.1/(\sqrt{2}+1) = 0.041$，特征“年龄=30岁”的自适应学习率是$0.1/(\sqrt{300}+1)=0.0055$。特征"年龄=100岁”将比特征“年龄=30岁”拥有更大的步长。

上面两个关于逻辑回归的技巧到底有没有用呢？[tinrtgu][lr_in_practise]在参加kaggle比赛Display Advertising Challenge时使用这两个技巧的逻辑回归超过了基准线（不是第一名，第一名用的是FFM方法）。这是一个广告CTR预估的比赛，由知名广告公司Criteo赞助举办。数据包括4千万训练样本，500万测试样本，解压缩以后，train.txt文件11.7G，test.txt文件1.35G。特征包括13个数值特征，26个类别特征，评价指标为logloss。 该程序只使用了200M内存，在tinrtgu机器上跑了大约30分钟。这里将其大概50行的代码贴出来并详细注解一下。
```python
'''
           DO WHAT THE FUCK YOU WANT TO PUBLIC LICENSE
                   Version 2, December 2004

Copyright (C) 2004 Sam Hocevar <sam@hocevar.net>

Everyone is permitted to copy and distribute verbatim or modified copies of this license document, and changing it is allowed as long as the name is changed.

           DO WHAT THE FUCK YOU WANT TO PUBLIC LICENSE
  TERMS AND CONDITIONS FOR COPYING, DISTRIBUTION AND MODIFICATION

 0. You just DO WHAT THE FUCK YOU WANT TO.
'''

from datetime import datetime
from csv import DictReader
from math import exp, log, sqrt


# 变量定义#################################################################

train = 'train.csv'  # 训练文件路径
test = 'test.csv'  # 测试文件路径

D = 2 ** 20   # 特征维度，维度固定是因为会使散列技巧
alpha = .1    # 随机梯度下降算法的初始学习率


# 函数定义 #######################################################

# A. 对数损失函数 $y_{i} \log p(y=1|\mathbf{x}_i) + (1-y_i)\log p(y=0|\mathbf{x}_i)$
# 输入参数:
#     p: 预测值
#     y: 真实值
# 输出：
#     预测值和真实值之间的对数损失
def logloss(p, y):
    p = max(min(p, 1. - 10e-12), 10e-12)
    return -log(p) if y == 1. else -log(1. - p)


# B. 应用散列技巧将样本编码成one-code
# hash(特征名,特征值)：0/1
# 输入参数:
#     csv_row: 单个样本数据, 比如: {'Lable': '1', 'I1': '357', 'I2': '', ...}
#     D: 样本空间的维度，散列的模
# 输出:
#     x: 特征索引列表，其对应的值都是1
def get_x(csv_row, D):
    x = [0]  # 偏差b的位置0都有
    for key, value in csv_row.items():
        index = int(value + key[1:], 16) % D  # 散列函数（比较弱，可以优化）
        x.append(index)
    return x 


# C. 对数几率函数 p(y = 1 | x; w)
# 输入参数:
#     x: 特征列表
#     w: 权重列表
# 输出:
#     给定特征和权重，预测为1的概率
def get_p(x, w):
    wTx = 0.
    for i in x:  
        wTx += w[i] * 1.  # x中所有index对应的特征值都是1
    return 1. / (1. + exp(-max(min(wTx, 20.), -20.)))  # 带边界的sigmod函数


# D. 更新模型
# 输入参数:
#     w: 权重列表w
#     n: 特征出现次数列表，用于自适应学习率
#     x: 样本-特征索引列表
#     p: 当前模型对样本x的预测值
#     y: 样本x的标签
# OUTPUT:
#     w: 更新的模型
#     n: 更新的特征出现次数列表
def update_w(w, n, x, p, y):
    for i in x:
        # alpha / (sqrt(n) + 1) 是自适应的学习率
        # (p - y) * x[i] 是当前下降梯度
        w[i] -= (p - y) * alpha / (sqrt(n[i]) + 1.)
        n[i] += 1.

    return w, n


# 训练和测试 #######################################################

# 初始化模型参数
w = [0.] * D  # 权重列表
n = [0.] * D  # 特征出现次数列表

# 训练数据
loss = 0.  # 累计损失
for t, row in enumerate(DictReader(open(train))):
    y = 1. if row['Label'] == '1' else 0.

    del row['Label']  # 已经转换成y
    del row['Id']  # 不需要Id信息

    # 主要的训练过程
    # S1: 获取散列后的特征索引列表
    x = get_x(row, D)

    # S2: 计算当前模型对当前样本的预测值
    p = get_p(x, w)

    # 过程中打印训练的样本数和累计损失
    loss += logloss(p, y)
    if t % 1000000 == 0 and t > 1:
        print('%s\tencountered: %d\tcurrent logloss: %f' % (
            datetime.now(), t, loss/t))

    # S3: 更新模型参数
    w, n = update_w(w, n, x, p, y)

# 测试
with open('submission1234.csv', 'w') as submission:
    submission.write('Id,Predicted\n')
    for t, row in enumerate(DictReader(open(test))):
        Id = row['Id']
        del row['Id']
        x = get_x(row, D)
        p = get_p(x, w)
        submission.write('%s,%f\n' % (Id, p))
```
## LR的工业级实现Vowpal Wabbit
本来这一章节准备详细介绍一下Vowpal Wabbit的用法的，但是考虑到[官方提供的wiki][vw_doc]已经非常详细了，从安装到使用，从输入格式到算法细节，从命令参数说明到服务搭建，基本上能支撑我们把vowpal wabbit用到如臂使指。唯一美中不足的是文档完全是英文的，后续会结合它的[源码学习][src_vw_doc]尝试做一些[翻译工作][vw_doc_zh]。所以这章节主要介绍一下为啥选择vowpal wabbit作为逻辑回归工业级的工具。

首先，vowpal wabbit在使用逻辑回归算法上非常专业。该开源项目是由雅虎发起，目前是由微软研究院维护的一个快速核外学习系统。虽然它目前的定位是在线的机器学习系统，涉及在线服务，散列，交互学习等，但是它最初就是以线性回归为初始版本（2.3），后来支持了逻辑回归并加入了更多的优化算法，像L-BFGS，FTRL等。所以通过它的源码能完全掌握逻辑回归的算法理论和工程实践。其次，vowpal wabbit非常高效和内存消耗小。一方面它是用C++编码，代码运行非常快，另一方面它用高效且消耗内存少的L-BGFS优化算法，使得其在性能方面表现极佳。这一点在tinrtgu的样例中能够体现出来，而且vowpal wabbit能够比他做得更好。最后，vowpal wabbit是便捷易用的。它不仅支持命令行的形式来使用，还支持python使用，而且提供了像`vw-varinfo`等便捷的工具类脚本。


## 思考与总结
1. 快速的迭代 vs 缓慢的精确
从上面优化算法的比对中，我们可以看出随机梯度下降比批量梯度下降占有很大的优势，基本上工程的实现都会优先考虑随机梯度下降，因为训练时间短，占用内存小，能够快速的迭代更新模型参数。而批量梯度下降的训练时间和样本的数量成正比的，随着样本规模的扩大，批量梯度下降的训练时间终究会超出项目的忍耐限度。所以相较于缓慢精确的算法，快速迭代的算法更符合现在这个大数据时代。就像吴恩达在新书《机器学习训练秘籍》所说“不要在一开始就试图设计和构建一个完美的系统。相反，应尽可能快(例如在短短 几天内)地构建和训练一个系统雏形。然后使用误差分析法去帮助你识别出最有前景的方向，并据此不断迭代改进你的算法。”

2. 是否使用散列
在平时的推荐项目以及以往的工程中，很少见过通过散列的方式来预处理特征的。当我们样本很少时，散列的意义不大，感觉连优化算法都会选批量梯度下降，因为无论时间和空间的代价都付得起。而碰到样本比较大时，大家可能的第一反应是上hadoop，上spark等。但当我们限定了只能在当前的系统上平稳过度时（毕竟hadoop等所耗费的资源不是小数目），此时才会想到散列技巧。但是散列的冲突始终会让人觉得可能会搞丢了重要特征。好在最近几年embedding被越来越多的人接受，虽然人们还是无法解释对应的维度代表啥现实含义。而散列技巧算是一种维度可解释的embedding。而对于散列的冲突特性，我想说的是像vowpal wabbit默认的特征空间是$2^{18}\approx 26$万，最大支持$2^{61}\approx 2.3*10^{18}\approx230$亿亿。这样的特征空间对于小样本数据很难冲突，而对于大样本数据冲突几万个也无伤大雅。当然[严格的理论依据][collision_prof]还是有的。除了可以调节特征空间的大小外，vowpal wabbit还提供命名空间`NameSpace`，就是每一个命名空间有自己的散列函数，这将大大降低特征冲突，也方面做特征交叉。所以我觉得还是可以多用用散列技巧的，毕竟它可以给我们特征的预处理带来时间和空间上的极大便利，再不济可以将特征空间大小当做作为模型参数来进行优化。

3. 逻辑回归中的特征工程是否尽量离散化
在实际项目中，特征的数据类型不仅有离散型，还有连续型。在工业界，很少直接将连续值作为特征直接丢给逻辑回归模型，而是将连续特征离散化，这样有如下优点：首先，离散化的特征对异常数据有很强的鲁棒性，也就是当我们将年龄>60当做老年用户时，年龄=200岁的异常数据不会给模型带来很大影响。其次，将年龄从1维扩展为幼儿，青年，壮年，老年4维，每个维度都有单独的权重，相当于为模型引入了非线性，提高模型的表达能力，交叉特征也是相同的道理。最后特征的离散化会使模型更加稳定，比如在推荐项目中会使用item的实时ctr这一特征，它能在一定程度上反映出item在用户群体中的受欢迎程度，所以我们可以给ctr分段来降低实时ctr带来的受欢迎程度的影响。

4. 逻辑回归的优与劣
逻辑回归算是机器学习中最简单的模型，可解释性好，特征的权重大小清楚的表现了特征的重要性，此外工程上容易实现，训练耗时短。以上都是逻辑回归的优点。逻辑回归最大的缺点是非常依赖特征工程，而且特征工程基本决定了模型的好坏。这也解释了tinrtgu参加的比赛中是FFM算法取得了第一名，同样也解释了Facebook会使用GBDT来筛选特征，LR来模型训练，毕竟人力有时尽。

本文主要介绍了逻辑回归的理论推导以及工程实践技巧。理论推导部分从一元线性回归延伸到多元线性回归，最终升级为逻辑回归。此外还着重介绍了它的优化算法。而工程实践技巧主要介绍了散列技巧以及自适应的随机梯度下降算法，最后还展示了一个50行左右实现全部技巧的代码。在本文的最后安利了一波vowpal wabbit软件以及在学习逻辑回归过程中的一些思考。



## 附推导过程
### 一元线性回归推导过程
已知观察值$X=[x_1,x_2,\dots,x_n]$和$Y=[y_1,y_2,\dots,y_n]$，求解$a,b$使得代价函数$Cost(a,b)=\sum_{i=1}^n(ax_i+b-y_i)^2$值最小。西瓜书说求解的过程是线性回归模型的最小二乘参数估计。因为代价函数是关于a或b的二次函数，所以求解的过程就是求导使得导函数为0。
$$
\begin{eqnarray*}
Cost(a,b) &=& \sum_{i=1}^n x_i^2 a^2 + \sum_{i=1}^n(2bx_i-2x_iy_i)a+\sum_{i=1}^n(b^2+y_i^2-2by_i)    \\
              &=& \sum_{i=1}^n b^2 + \sum_{i=1}^n (2ax_i-2y_i)b+\sum_{i=1}^n(a^2x_i^2-2ax_iy_i+y_i^2)  \\
\frac{\partial Cost(a,b)}{\partial a} &=& 2(\sum_{i=1}^nx_i^2)a + 2\sum_{i=1}^nx_ib -2\sum_{i=1}^nx_iy_i \\
\frac{\partial Cost(a,b)}{\partial b} &=& 2nb + 2\sum_{i=1}^nx_ia -2\sum_{i=1}^ny_i
\end{eqnarray*}
$$
前两行是代价函数展开后分别以a，b为变量来表示代价函数，后两行分别对a,b求偏导得到关于a,b的两个二元一次方程组。通过令两个偏导为0，即可求得公式$(1.1)$

### 多元线性回归推导过程
该推导过程涉及到对矩阵求导，需要补充一些微积分的基本知识。
已知观察数据$\mathbf{X}$和$\mathbf{y}$，求解参数$\mathbf{w}$使得代价函数$Cost(\mathbf{w})=(\mathbf{y}-\mathbf{X}\mathbf{w})^T(\mathbf{y}-\mathbf{X}\mathbf{w})$取到最小值。推导就是令代价函数的导数为0。
$$
\begin{eqnarray*}
Cost(\mathbf{w}) &=& \mathbf{y}^T\mathbf{y} -\mathbf{y}^T\mathbf{Xw}-\mathbf{w}^T\mathbf{X}^T\mathbf{y} +\mathbf{w}^T\mathbf{X}^T\mathbf{Xw}  \\
\frac{\partial Cost(\mathbf{w})}{\partial \mathbf{w}} &=& 0  -\mathbf{X}^T\mathbf{y} -\mathbf{X}^T\mathbf{y} + (\mathbf{X}^T\mathbf{X}+\mathbf{X}\mathbf{X}^T)\mathbf{w} = 2\mathbf{X}^T(\mathbf{X}\mathbf{w}-\mathbf{y})  =  0\\
\end{eqnarray*}
$$

### 逻辑回归推导过程
该推导过程需要使用以下求偏导公式：
$$
\begin{eqnarray*}
\frac{\partial (e^{-ax})}{\partial x} &=& -ae^{-ax}\\
\frac{\partial (lnx)}{\partial x} &=& \frac{1}{x}\\
\end{eqnarray*}
$$
代价函数$Cost(\mathbf{w})= \sum_{i=1}^{N}\left[y_{i} \log \left(h_{\mathbf{w}}(\mathbf{x}_i)\right)+\left(1-y_{i}\right) \log \left(1-h_{\mathbf{w}}(\mathbf{x}_i)\right)\right]$推导过程如下：
$$
\begin{eqnarray*}
\frac{\partial \log \left(h_{\mathbf{w}}(\mathbf{x}_i)\right)}{\partial \mathbf{w}} &= & \frac{\partial }{\partial \mathbf{w}} \log \left(1+e^{-\mathbf{w}^T\mathbf{x}_i}\right)^{-1}\\ 
& =& \frac{\mathbf{x}_i e^{-\mathbf{w}^T\mathbf{x}_i}}{1+e^{-\mathbf{w}^T\mathbf{x}_i}} \\
\frac{\partial \log \left(1-h_{\mathbf{w}}(\mathbf{x}_i)\right)}{\partial \mathbf{w}} &= & \frac{\partial }{\partial \mathbf{w}} \log \left( \frac{e^{-\mathbf{w}^T\mathbf{x}_i}}{ 1+e^{-\mathbf{w}^T\mathbf{x}_i}}\right)\\
&=& \frac{\partial }{\partial \mathbf{w}} \log e^{-\mathbf{w}^T\mathbf{x}_i} -\log \left( 1+e^{-\mathbf{w}^T\mathbf{x}_i} \right) \\
&=& -\mathbf{x}_i +  \frac{\mathbf{x}_i e^{-\mathbf{w}^T\mathbf{x}_i}}{1+e^{-\mathbf{w}^T\mathbf{x}_i}} \\
\frac{\partial Cost(\mathbf{w})}{\partial \mathbf{w}}  &=& \sum_{i=1}^{N} \left[ y_i *\frac{\mathbf{x}_i e^{-\mathbf{w}^T\mathbf{x}_i}}{1+e^{-\mathbf{w}^T\mathbf{x}_i}}  + \left(1-y_i\right)*\left(-\mathbf{x}_i +  \frac{\mathbf{x}_i e^{-\mathbf{w}^T\mathbf{x}_i}}{1+e^{-\mathbf{w}^T\mathbf{x}_i}} \right) \right] \\
&=&   \sum_{i=1}^{N} \left(-\mathbf{x}_i + \frac{\mathbf{x}_i e^{-\mathbf{w}^T\mathbf{x}_i}}{1+e^{-\mathbf{w}^T\mathbf{x}_i}} +\mathbf{x}_iy_i \right) \\
&=&  \sum_{i=1}^{N} \mathbf{x}_i\left(h_{\mathbf{w}}(\mathbf{x}_i)-y_i\right)
\end{eqnarray*} 
$$





## 参考文献
1. [基于Vowpal_Wabbit的大规模图像分类][vw_practise]
2. [Vowpal Wabbit GitHub主页][vowpal_wabbit]
3. [Deep  Dive Into Logistic Regression][lr_blog]
4. [逻辑回归代价函数凸证明][convex_prof]
5. [梯度下降_wiki][gradient_descent]
6. [散列技巧的冲突分析][collision_prof]
7. [自适应学习][adapt_learning]
8. [逻辑回归实战][lr_in_practise]
9. [vowpal wabbit源码阅读][src_vw_doc]
10. [vowpal wabbit帮助文档][vw_doc]
11. [vowpal wabbit帮助文档(翻译中)][vw_doc_zh]

[vw_practise]:  http://www.academia.edu/31836831/%E5%9F%BA%E4%BA%8EVowpal_Wabbit%E7%9A%84%E5%A4%A7%E8%A7%84%E6%A8%A1%E5%9B%BE%E5%83%8F%E5%88%86%E7%B1%BB
[vowpal_wabbit]: https://github.com/VowpalWabbit/vowpal_wabbit
[lr_blog]: http://www.philippeadjiman.com/blog/2017/12/09/deep-dive-into-logistic-regression-part-1/
[convex_prof]:http://qwone.com/~jason/writing/convexLR.pdf
[gradient_descent]:https://zh.wikipedia.org/wiki/%E6%A2%AF%E5%BA%A6%E4%B8%8B%E9%99%8D%E6%B3%95
[collision_prof]: http://people.csail.mit.edu/romer/papers/TISTRespPredAds.pdf
[adapt_learning]: http://static.googleusercontent.com/media/research.google.com/en//pubs/archive/41159.pdf
[lr_in_practise]:https://www.kaggle.com/c/criteo-display-ad-challenge/discussion/10322
[src_vw_doc]: https://doc.haojunyu.com/vw2.3/
[vw_doc]: https://github.com/VowpalWabbit/vowpal_wabbit/wiki
[vw_doc_zh]:https://book.haojunyu.com/vw_practise/

