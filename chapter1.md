---
marp: true
header: '**机器学习：第1章-概述**'
---
<!-- backgroundColor: silver -->
# 《机器学习》课程简介
## 教师：肖宇
## 授课时间：
## 答疑时间：
## 助教：


---
# 第一章、机器学习概述

## 

---
# 什么是机器学习？
## 维基百科概念
>  机器学习是近20多年兴起的一门多领域交叉学科，涉及**概率论、统计学、逼近论、凸分析、算法复杂度理论**等多门学科。机器学习理论主要是设计和分析一些让计算机可以自动“学习”的算法。机器学习算法是一类从数据中自动分析获得规律，并利用规律对未知数据进行预测的算法。因为学习算法中涉及了大量的统计学理论，机器学习与统计推断学联系尤为密切，也被称为**统计学习理论**。算法设计方面，机器学习理论关注可以实现的，行之有效的学习算法。很多推论问题属于无程序可循难度，所以部分的机器学习研究是开发容易处理的近似算法。

---
# 什么是机器学习？
## 机器学习的定义
>- 机器学习是一门人工智能的科学，该领域的主要研究对象是人工智能，特别是如何在经验学习中改善具体算法的性能。
>- 机器学习是用数据或以往的经验，以此优化计算机程序的性能标准。
>- A computer program is said to learn from **experience E** with respect to some class of **tasks T** and **performance measure P**, if its performance at tasks in T, as measured by P, improves with experience E.

---
# 什么是机器学习？
## 机器学习的应用
- 数据挖掘
- 计算机视觉
- 自然语言处理
- 生物特征识别
- 搜索引擎
- 医学诊断
- 检测信用卡欺诈
- 语音和手写识别
- 机器人
- ...

---
# 机器学习的发展历程
## “黑暗时代”，人工智能的诞生（1943年~1956年）
- Warren McCulloch和Walter Pitts在1943年发表了人工智能领域的开篇之作，提出了人工神经网络模型。
- John von Neumann。他在1930年加入了普林斯顿大学，在数学物理系任教，和阿兰·图灵是同事。
- Marvin Minsky和Dean Edmonds建造第一台神经网络计算机。
- 1956年：John McCarthy从普林斯顿大学毕业后去达特茅斯学院工作，说服了Marvin Minsky和Claude Shannon在达特茅斯学院组织一个暑期研讨会，召集了对机器智能、人工神经网络和自动理论感兴趣的研究者，参加由IBM赞助的研讨会。

---
# 机器学习的发展历程
![width:800px](pictures/1.1.jpg)

---
# 机器学习的发展历程
## 新的方向：
- 集成学习
- 可扩展机器学习（对大数据集、高维数据的学习等）
- 强化学习
- 迁移学习
- 概率网络
- 深度学习

---
# 国内外的研究者
## 国外学者: M. I. Jordan, Andrew Ng, Tommi Jaakkola, David Blei, D.Koller, Peter L. Bartlett, J. D. Lafferty
## 国内：李航, 周志华, 杨强, 王晓刚，唐晓鸥，唐杰，刘铁岩，何晓飞，朱筠，吴军，张栋，戴文渊，余凯，邓力，孙健

---
# 机器学习和数据挖掘
- 机器学习是数据挖掘的重要工具。
- 数据挖掘不仅仅要研究、拓展、应用一些机器学习方法，还要通过许多非机器学习技术解决数据仓储、大规模数据、数据噪音等等更为实际的问题。
- 机器学习的涉及面更宽，常用在数据挖掘上的方法通常只是“从数据学习”，然则机器学习不仅仅可以用在数据挖掘上，一些机器学习的子领域甚至与数据挖掘关系不大，例如增强学习与自动控制等等。
- 数据挖掘试图从海量数据中找出有用的知识。
- 大体上看，数据挖掘可以视为机器学习和数据库的交叉，它主要利用机器学习界提供的技术来分析海量数据，利用数据库界提供的技术来管理海量数据。

---
# 机器学习和数据挖掘
![width: 900px](/pictures/1.3.png)

---
# 为什么要研究大数据机器学习？
## 例如: “尿布→啤酒”关联规则
> 实际上，在面对少量数据时关联分析并不难，可以直接使用统计学中有关相关性的知识，这也正是机器学习界没有研究关联分析的一个重要原因。关联分析的困难其实完全是由海量数据造成的，因为数据量的增加会直接造成挖掘效率的下降，当数据量增加到一定程度，问题的难度就会产生质变，例如，在关联分析中必须考虑因数据太大而无法承受多次扫描数据库的开销、可能产生在存储和计算上都无法接受的大量中间结果等。

---
# 机器学习和统计学习
## 机器学习
>机器学习(machine learning)致力于研究通过**计算**的手段，利用经验改善系统自身的性能。在计算机系统中，“经验”通常以“数据”形式存在，因此，机器学习所研究的主要内容，是关于在计算机上从数据中产生“模型”（model）的算法，即“学习算法”（learning algorithm）。如果说计算机科学是研究关于“算法”的学问，那么类似地，可以说机器学习是研究关于“学习算法”的学问。（周志华，《机器学习》，p1）


## 统计学习
>统计学习(statistical learning)是关于**计算机**基于**数据**构建**概率统计模型**并运用模型对数据进行预测与分析的一门学科。统计学习也称为统计机器学习(statistical machine learning)。(李航，《统计学习方法》，p1)
  
---
# 机器学习和统计学习

## 机器学习
>强调了通过机器设备（如计算机）进行学习以提升系统，由机器实施“计算”，因此重点在于“算法”。

## 统计学习
>强调了通过统计理论与模型（如线性回归）学习以提升系统，重点在于“模型”。

## 两者是统一的
>都是为了构建一个可通过经验进行自我提升的系统，“模型”和“算法”都是这个过程中涉及的不同或缺组成部分。

---
# 机器学习和统计学习
|machine learning| statistics |
|---:|---:|
|instance | data point |
|feature | covariate|
|label | response|
|weights | parameters|
|learning | fitting/estimation |
|supervised learning | regression/classification|
|unsupervised learning | density estimation, clustering |

---
# 机器学习和统计学习
Simon Blomberg:
>From R’s fortunes package: To paraphrase provocatively, ‘machine learning is statistics minus any checking of models and assumptions’.

Andrew Gelman:
>In that case, maybe we should get rid of checking of models and assumptions more often. Then maybe we’d be able to solve some of the problems that the machine learning people can solve but we can’t!

---
# 机器学习和统计学习
- 研究方法差异
  - 统计学研究形式化和推导
  - 机器学习更容忍一些新方法
- 维度差异
  - 统计学强调低维空间问题的统计推导（confidence intervals, hypothesis tests, optimal estimators）
  - 机器学习强调高维预测问题
- 统计学和机器学习各自更关心的领域：
  - 统计学: survival analysis, spatial analysis, multiple testing, minimax theory, deconvolution,  semiparametric inference, bootstrapping, time series.
  - 机器学习: online learning, semisupervised learning, manifold learning, active learning, boosting.

---
# 机器学习的分类
机器学习是一个范围宽阔、内容繁多、应用广泛的领域，并不存在一个统一的理论体系涵盖所有内容。下面从几个角度对机器学习方法进行分类。
## 基本分类
- 监督学习
- 无监督学习
- 强化学习
- 半监督学习
- 主动学习

---
# 机器学习的分类
## 按模型分类
- 概率模型与非概率模型
- 线性模型与非线性模型
- 参数化模型与非参数化模型

---
# 机器学习的分类
## 按算法分类
- 在线学习
- 批量学习

## 按技巧分类
- 贝叶斯学习
- 核方法

---
# 监督学习
输入实例$x$的特征向量：
$$
x=(x^{(1)},x^{(2)},x^{(3)}, ..., x^{(n)})^T
$$
注意$x^{(i)}$与$x_i$的区别，后者表示多个特征变量的第i个
$$
x_i=(x_i^{(1)},x_i^{(2)},...,x_i^{(n)})^T
$$
训练集
$$
T=\{(x_1,y_1),(x_2, y_2),..., (x_N,y_N)\}
$$
输入变量和输出变量：分类问题、回归问题、标注问题

---
# 监督学习
**联合概率分布**
- 假设输入与输出的随机变量$X$和$Y$遵循联合概率分布$P(X,Y)$
- $P(X,Y)$为分布函数或分布密度函数
- 对于学习系统来说，联合概率分布是未知的
- 训练数据和测试数据被看作是依联合概率分布$P(X,Y)$独立同分布产生的。

**假设空间**
- 监督学习目的是学习一个由输入到输出的映射，称为模型
- 模式的集合就是假设空间（`hypothesis space`）
- 概率模型:条件概率分布$P(Y|X)$, 决策函数：$Y=f(X)$

---
# 监督学习
问题的形式化


---
# 机器学习方法三要素
机器学习方法都是由模型、策略和算法构成的，可以简单地表示为：
$$方法=模型+策略+算法$$




---
# 学习资源
## [斯坦福机器学习](http://v.163.com/special/opencourse/machinelearning.html)
## CMU 机器学习课程
- http://www.cs.cmu.edu/~epxing/Class/10715/  
- http://www.cs.cmu.edu/~epxing/Class/10708/
- http://www.cs.cmu.edu/~epxing/Class/10701
- https://sites.google.com/site/10601a14spring/syllabus 
