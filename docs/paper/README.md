

# 应用于工业场景的大规模自动化特征工程



[TOC]

# 摘要

机器学习技术已经成为我们日常生活的重要组成部分， 广泛应用于各种任务中。然而，构建性能良好的机器学习应用程序需要高度专业化数据科学家和领域专家。作为一种重要的驱动力，自动机器学习技术（AutoML）寻求自动化构建机器学习应用程序的整个过程，使得非专家也能够执行此过程，来减少对数据科学家的需求 。 特征工程已被公认为构建机器学习系统的关键环节。然而，特征工程在机器学习开发过程中是一个非常乏味且通常是最耗时的过程。 近年来，人们越来越多地致力于自动特征工程方法的开发，以使大量繁琐的手工工作得以解放。然而，对于工业任务，这些方法的效率和可扩展性仍然远远不能令人满意。在本文中，我们提出了一个通用的自动化机器学习框架 AutoTabular ， 它能够自动进行特征生成和特征选择，不仅涵盖了大多数现有特征预处理方法，而且还能进行人机交互，利用专家先验知识指导新的特征生成。 它可以提供出色的效率和可扩展性，以及必要的可解释性和良好的性能。大量的实验结果表明，与其他现有的自动化工具箱相比，AutoTabular 在多个数据集上具有显著优势。此外，我们所提出的方法具有足够的可扩展性，确保了它能够部署在大规模工业任务中。

# 自动机器学习

自动机器学习（AutoML）是一种新兴的技术，用于自动执行重复的机器学习任务。这些任务的自动化将加快流程，减少错误和成本，并提供更准确的结果，因为它使企业能够选择性能最佳的算法。以下是维基百科对autoML的定义：

自动机器学习（AutoML）是将机器学习应用于实际问题的端到端过程自动化的过程。

##　自动机器学习步骤

![img](https://img2018.cnblogs.com/blog/1473228/201902/1473228-20190214212402566-383152380.png)

AutoML服务旨在自动化机器学习过程的部分或所有步骤，包括：

- 数据预处理：此过程包括提高数据质量，并使用数据清理、数据集成、数据转换等方法将非结构化原始数据转换为结构化格式。
- 特征工程：AutoML可以通过分析输入数据自动创建与机器学习算法更兼容的特征。
- 特征提取：此过程包括组合不同的特征或数据集，以生成新的特征，从而实现更准确的结果并减少正在处理的数据的大小。
- 特征选择：AutoML可以自动选择有用特征。
- 算法选择和超参数优化：AutoML工具可以选择最佳的超参数和算法，无需人工干预。



## 自动化机器学习为什么重要

### 需要更多的数据科学家

随着数据科学越来越融入我们的生活，企业在这一领域需要更多的解决方案，并需要更多的数据科学家来构建这些解决方案。如果没有数据科学方法，公司可能无法理解其流程、监控绩效水平或采取某些措施来防止巨大损失。
2017年IBM的一份报告指出，到2020年，对数据科学家的需求将增加28%。该报告还指出，填补一个数据科学家职位平均需要43-51天。考虑到数据科学家的稀缺性和构建数据科学解决方案的时间，AutoML解决方案可以帮助企业满足对数据科学家的需求。

### 应用机器学习算法人为错误

传统的数据分析和建模由数据科学家来实现机器学习算法的全部过程，并选择最适合商业案例的方法。然而，实施过程容易出现人为错误和偏见。AutoML工具可以自动化这一过程，还可以运行一组更广泛的机器学习算法来选择最佳算法，这可能是数据科学家以前没有考虑过的。这些功能将加速机器学习过程，AutoML解决方案将提高机器学习项目的投资回报率（ROI）

## 自动化机器的收益

- 降低成本
  - 提高数据科学家的生产力
  - 机器学习的自动化减少了对数据科学家的需求
- 增加收入和客户满意度
- 以更高的准确性推出更多的模型也可以改善其他不太实际的业务成果。例如，模型带来了自动化，从而提高了员工的敬业度，使他们能够专注于更有趣的任务。

## 既然有自动ML方法，为什么我们如此依赖数据科学家？
与当前的Auto ML方法相比，数据科学家在模型构建方面有两个优势：

- 符合定制规范：大多数autoML工具都会优化模型性能，但这只是现实生活中机器学习项目的规范之一。例如：
  - 如果模型需要嵌入边缘设备，计算和存储需求迫使公司选择更简单的模型。
  - 如果需要解释性，则只能使用某些类型的模型。
- 模型性能：在机器学习竞赛社区Kaggle上，人类仍然很容易击败autoML工具生成的模型。AutoML工具尚未赢得任何数据科学竞赛。



## AutoML 开源工具箱

### AutoTabular 和其他框架对比

|              | Classificaton | Regression | Clustering | Ranking | Time-Series | Anomly Detection | Text-Transformer |      |
| ------------ | ------------- | ---------- | ---------- | ------- | ----------- | ---------------- | ---------------- | ---- |
| AutoGluon    | Yes           | Yes        |            |         |             |                  | Yes              |      |
| Auto-sklearn | Yes           | Yes        |            |         |             |                  |                  |      |
| H2O-AutoML   | Yes           | Yes        |            |         |             |                  | Yes              |      |
| TPOT         | Yes           | Yes        |            |         |             |                  |                  |      |
| AutoTabular  | Yes           | Yes        | Yes        | Yes     | Yes         | Yes              | Yes              |      |



#### Classification Example

```python
from autotabular import AutoML
from sklearn.datasets import load_iris
# Initialize an AutoML instance
automl = AutoML()
# Specify automl goal and constraint
automl_settings = {
    "time_budget": 10,  # in seconds
    "metric": 'accuracy',
    "task": 'classification',
    "log_file_name": "test/iris.log",
}
X_train, y_train = load_iris(return_X_y=True)
# Train with labeled input data
automl.fit(X_train=X_train, y_train=y_train,
           **automl_settings)
# Predict
print(automl.predict_proba(X_train))
# Export the best model
print(automl.model)
```

#### Regression

```python
from autotabular import AutoML
from sklearn.datasets import load_boston
# Initialize an AutoML instance
automl = AutoML()
# Specify automl goal and constraint
automl_settings = {
    "time_budget": 10,  # in seconds
    "metric": 'r2',
    "task": 'regression',
    "log_file_name": "test/boston.log",
}
X_train, y_train = load_boston(return_X_y=True)
# Train with labeled input data
automl.fit(X_train=X_train, y_train=y_train,
           **automl_settings)
# Predict
print(automl.predict(X_train))
# Export the best model
print(automl.model)
```

#### Time-Series

```python
import numpy as np
from autotabular import AutoML
X_train = np.arange('2014-01', '2021-08', dtype='datetime64[M]')
y_train = np.random.random(size=72)
automl = AutoML()
automl.fit(X_train=X_train[:72],  # a single column of timestamp
           y_train=y_train,  # value for each timestamp
           period=12,  # time horizon to forecast, e.g., 12 months
           task='forecast', time_budget=15,  # time budget in seconds
           log_file_name="test/forecast.log",
          )
print(automl.predict(X_train[72:]))
```

####  Clustering

```python
from sklearn.datasets import fetch_openml
from autotabular import AutoML

n_samples = 1500
random_state = 170
X, y = make_blobs(n_samples=n_samples, random_state=random_state)

automl = AutoML()
automl.fit(
    X_train, n_clusters=3,
    task='clustering', time_budget=10,    # in seconds
)
# Incorrect number of clusters
y_pred =automl.predict(X)
```

####  Rank

```python
from sklearn.datasets import fetch_openml
from autotabular import AutoML
X_train, y_train = fetch_openml(name="credit-g", return_X_y=True, as_frame=False)
y_train = y_train.cat.codes
# not a real learning to rank dataaset
groups = [200] * 4 + [100] * 2    # group counts
automl = AutoML()
automl.fit(
    X_train, y_train, groups=groups,
    task='rank', time_budget=10,    # in seconds
)
```

### Auto-sklearn

Auto-sklearn 是由德国 AutoML 团队基于著名机器学习工具包sklearn开发的自动化机器学习框架，是目前最成熟且功能相对完备的AutoML框架之一。Auto-sklearn集成了16种分类模型、13种回归模型、18种特征预处理方法和5种数据预处理方法，共组合产生了超过110个超参数的结构化假设空间，并采用基于序列模型的贝叶斯优化器搜索最优模型。Auto-Sklearn的使用方法也和scikit-learn库基本一致， 这让熟悉sklearn的开发者很容易切换到Auto-Sklearn。在模型方面，除了sklearn提供的机器学习模型，还加入了xgboost、lightgbm 等算法支持。

#### Auto-Sklearn 优点

- Auto-sklearn的最大优势在于它建立在sklearn的生态上，所以具有非常好的可扩展性以及兼容性，毕竟sklearn是目前为止最为流行的机器学习工具。
- Auto-skearn 可以极大地减少对于领域专家和算法专家的依赖： 一方面 Auto-skearn 可以自动进行模型选择和参数调优； 另一方面， Auto-skearn 根据单次训练时长和总体训练时间设置，最大化的利用机器性能和时间。
- Auto-Sklearn支持切分训练/测试集的方式，也支持使用交叉验证。从而减少了训练模型的代码量和程序的复杂程度。另外，Auto-Sklearn支持加入扩展模型以及扩展预测处理方法。

#### Auto-Sklearn 缺点

- Auto-sklearn 基于 sklearn库开发，所支持的模型必须有类似 sklearn 的接口， 对于其他的算法库，扩展性较差。
- Auto-sklearn 目前不支持深度学习模型。

- Auto-sklearn 的计算很慢，对于一个小数据集，计算时长往往一个小时以上
- Auto-sklearn 的数据清洗上所包含的方法有限，复杂情况下还需要人为参与，目前对非数值型数据不友好
- 但相反，对于自然语言处理的数据，缺乏一些有效的工具。



### TPOT

宾夕法尼亚大学研发的TPOT也是一个优秀的基于sklearn的自动化机器学习工具箱。	机器学习 Pipeline 包括 原始数据，到数据清洗，特征工程，模型选择，参数优化，模型验证等过程，整个过程可以有各自各样的组合。基于 **遗传编程**， TPOT 自动地智能地从数以千记的 pipeline 组合中探索出最优的处理流程。 在 TPOT 完成 搜索之后， 将会自动生成最优 Pipeline 的 Python 代码。TPOT 使用的是**遗传编程**的技术。**遗传编程与传统机器学习的区别：**机器学习主要是在一个参数空间上，通过调整参数获得最佳的预测结果，重点在找参数。遗传编程可以理解为一个能构造算法的算法，重点在找算法。TPOT 首先会分析数据的多项式特征和主成分特征，然后通过遗传算法迭代搜索交叉熵最小的特征子集，最后建立随机森林模型。

#### 优点：

- TPOT 使用遗传算法进行优化， 优势是 pipeline 的长度和结构可以是非常灵活的，而传统的优化方法一般都是在一个固定的 pipeline 结构上做参数优化。

- TPOT 基于 sklearn 来构建，能极大的利用 scikit-learn库的优势
- TPOT 中的特征预处理算子： StandardScaler, RobustScaler, MinMaxScaler, MaxAbsScaler, RandomizedPCA, Binarizer, and PolynomialFeatures.
- TPOT 的特征选择算子：VarianceThreshold, SelectKBest, SelectPercentile, SelectFwe, and Recursive Feature Elimination (RFE).
- TPOT 在 pipeline 优化完成后, 可以生成解决方案的Python代码，可以显示的看到整个 pipeline , 用户可以在此基础上进一步做分析与优化。

#### 缺点：

- TPOT 实现的 数据预处理和特征工程非常有限
- 需要手动进行数据预处理工作
- 缺少对文本特征的处理

### H2O

H2O 是 H2O.ai 公司的完全开源的分布式内存机器学习平台。H2O同时支持 R 和 Python，支持最广泛使用的统计和机器学习算法，包括梯度提升（Gradient Boosting）机器、广义线性模型、深度学习模型等。

H2O 包括一个自动机器学习模块，使用自己的算法来构建管道。它对特征工程方法和模型超参数采用了穷举搜索，优化了管道。

H2O 自动化了一些最复杂的数据科学和机器学习工作，例如特征工程、模型验证、模型调整、模型选择 和 模型部署。除此之外，它还提供了自动可视化以及机器学习的解释能力（MLI）。

H2O AutoML H2O（H2O.ai，2019）是一个分布式ML框架，用于协助数据科学家。在本文中，仅考虑H2O AutoML组件。H2O AutoML能够选择和调整分类算法，而无需自动预处理。可用的算法以专家定义的固定顺序或通过随机网格搜索选定的超参数进行测试。最后，对性能最佳的配置进行聚合，以创建一个集成。与所有其他经过评估的框架相比，H2O是在Java中使用Python绑定开发的，不使用scikit learn。

------------------

AutoFE作为AutoML的一环，常被集成在大型AutoML平台中，国外的知名互联网公司均在AutoML领域有所投入，其中最有代表性也最成熟的产品是谷歌的Cloud AutoML，不过该平台主要致力于深度AutoML，面向传统AutoML的平台有微软的NNI平台和AML平台，其中NNI内置了基于梯度和决策树的自动特征选择算法，其次是一些专注于AutoML的AI创业公司，比如H2O.ai开源的H2O AutoML是一个Java实现的AutoML平台，H2O支持常见的机器学习模型的自动化构建，其中AutoFE被转化为了超参优化的问题，即统一采用启发式搜索的方式搜索最优的特征、模型和超参数，同时还支持训练指标的可视化。

除了这些大型平台，也有一些具有针对性的低阶AutoML工具包被研发出来，最早的AutoML库是发布于2013年的AutoWEKA，它实现了模型选择，超参搜索等基本功能。依赖于著名机器学习工具包sklearn的Auto-sklearn是目前最成熟且功能相对完备的AutoML框架之一，Auto-sklearn集成了16种分类模型、13种回归模型、18种特征预处理方法和5种数据预处理方法，共组合产生了超过110个超参数的结构化假设空间，并采用基于序列模型的贝叶斯优化器搜索最优模型。宾夕法尼亚大学研发的TPOT也是一个优秀的基于sklearn的AutoFE工具包，TPOT首先会分析数据的多项式特征和主成分特征，然后通过遗传算法迭代搜索交叉熵最小的特征子集，最后建立随机森林模型。

专门面向AutoFE的工具包还比较少，其中最为著名的是麻省理工学院开发的FeatureTools，FeatureTools主要实现了关系型数据库的多表特征自动融合，FeatureTools使用了一种叫做深度特征合成(Deep Feature Synthesis, DFS)的算法，该算法能遍历通过关系数据库中模式描述的关系路径，当DFS遍历这些路径时，它通过数据操作(如求和、取平均值、计数等)生成合成特征，IBM也研发出了面向关系型数据库的自动特征融合工具OneBM，OneBM通过基于深度优先搜索的复杂关系图挖掘方法，实现了关系数据库特征工程的自动化以及高阶特征的自动合成和抽取等功能。除此之外，还存在一些Boruta-py、Tsfresh这类单独实现某种AutoFE算法的工具包。但目前这些落地的AutoFE工具包主要面向的特定场景下的关系型数据库，适用范围并不是特别广泛。

除了AutoFE算法的研究，以众包的方式实现AutoFE也是值得思考的方式，如麻省理工学院建立的用于管理数据科学协作的FeatureHub平台使得数据科学家能在特征工程开发过程中互相协作，这个系统能自动对生成特征评分，以确定当前模型的总体价值，这种以众包方式进行特征工程和机器学习的方法在测试时取得了不错的效果。

## 国内研究

与国外研究现状相似的是，AutoML相关技术的研究以互联网大公司为主导，比如阿里巴巴于2018年推出了商业化云端机器学习平台PAI，PAI包括了整套AutoML引擎，致力于在最大限度上减少机器学习业务的搭建成本，PAI包含了从数据预处理、特征工程到算法训练和模型评估的整套自动化处理流程。百度也推出了基于PaddlePaddle的EasyDL平台，主要支持自动模型搜索、迁移学习、神经架构搜索、模型部署的自动化，基本实现了零算法训练模型的效果。华为的NAIE AutoML平台是以华为诺亚实验室Vega AutoML为原型开发的工具包，其中涵盖了AutoML框架中的数据预处理、特征工程、算法模型、超参优化、集成学习五个模块，其中数据预处理、特征工程、算法模型均被视为超参优化模块的调优对象。除此之外，一些AI创业公司也在AutoML技术上有所投入，比如第四范式发布的自动机器学习平台AI Prophet AutoML和探智立方发布的DarwinML 2.0，这些平台在不同程度上都支持AutoFE。

## AutoML 相关研究论文

在学术界，AutoFE的研究也主要集中于基于特征枚举与评估的启发式算法(Heuristic)，比如IBM提出的Cognito是面向监督学习的AutoFE算法，它递归地在特征集上应用预先定义的数学变换获取新特征，并采用指定的特征选择策略移除冗余特征，避免特征数量的指数级增长。伯克利大学提出的ExploreKit框架放弃了采用模型评估的方式来评估候选特征的质量，而是生成所有可能的候选特征后使用一个学习到的排序函数对它们进行排序，大大降低了AutoFE固有的计算复杂度。不过基于启发式算法的AutoFE常常具有很高的计算成本，简单的特征生成方法也常常导致模型的过拟合，虽然深度学习是一个很好的特征学习方式，但由于其缺乏解释性以及过参数化的特性，导致深度学习在许多应用领域都不够有效。近几年，基于元学习(Meta Learing)的AutoFE得到了越来越多的研究，比如IBM提出的LFE(Learning Feature Engineering)构建了一个AutoFE模型，该模型可以从过去的特征工程经验中学习概括出不同的特征变换对模型性能的影响模式，从而自动完成具有可解释性的特征工程，但LFE只支持分类任务下的特征转换，不支持特征融合。基于强化学习的AutoFE也得到了一些研究，比如IBM提出的基于Q-Learning的TransGraph，但其同样会遇到特征组合爆炸的问题。

### 国内研究

与国外不同，国内的学术成果目前更多集中于高校。目前的绝大部分工作都将AutoFE视为离散空间上的黑盒优化过程，计算成本通常都难以接受，借鉴可微神经架构搜索的研究成果，南京大学于2021年提出了名为DIFER的首个可微的AutoFE模型，DIFER由三部分神经网络构成：Encoder-Predictor-Decoder，DIFER通过将离散的特征映射到连续空间，使得可以用基于梯度的优化方式端到端地优化神经网络，能够有效地抽取低阶和高阶特征。

总体来说，目前的AutoFE技术和AutoML技术大多都面向于非常成熟的应用场景，也就是说如果一个机器学习项目的解决方案是明确的，但特征工程、模型选择和超参调整等步骤的人力成本过高，就可以考虑借助AutoML技术，通过科学化的建模算法来有效地提高建模的效率及质量，减少人力成本。不过目前AutoML技术的效率还无法替代初级的模型调参人员，但对于毫无调参经验的业务人员来说，AutoML技术依旧是很有帮助的。

作为自动机器学习[13]-[16]中一个不容忽视的问题，自动特征工程近年来受到了广泛关注，并从不同角度提出了许多方法来解决这一任务[17]-[24]，[26]-[28]。在这一部分中，我们主要讨论了三种典型的策略，包括生成选择策略、基于强化学习的策略和基于迁移学习的策略。给定一个有监督的学习数据集，自动特征工程的典型方法是遵循生成-选择过程。FICUS算法[26]通过构造一组候选特征进行初始化，并迭代改进，直到计算预算耗尽。在每次迭代过程中，它执行波束搜索来构造新特征，并通常使用基于决策树中信息增益的启发式度量来选择特征。TFC[27]还通过迭代框架解决了这一任务。在每次迭代中，它基于当前特征库和所有可用的操作符生成所有合法特征，然后使用信息增益从所有候选特征中选择最佳特征，并将其保留为新特征库。有了这个框架，随着迭代的进行，可以获得更高阶的特征组合。然而，每次迭代中的穷举搜索会导致特征空间的组合爆炸，从而导致这种方法不可扩展。为了避免穷举搜索，提出了基于学习的方法，如FCTree算法[28]。FCTree通过对原始特征应用多个顺序变换来训练决策树并执行特征生成，并根据决策树每个节点上的信息增益选择特征。一旦建立了一棵树，在内部决策节点上选择的特征就会被删除用于获取构造的特征。[24]是一种基于回归的算法，它通过挖掘成对特征关联、识别每对特征之间的线性或非线性关系、应用回归并选择稳定的关系和提高预测性能来学习表示。由于时间的消耗，这些算法总是遇到性能和可伸缩性瓶颈如果没有巧妙的设计，特征生成和选择过程中的资源可能非常不令人满意。

本文还探讨了强化学习策略。
[17] 将特征选择形式化为强化学习问题，并引入蒙特卡罗树搜索的适应性。这里，选择可用功能子集的问题被转换为单人游戏，其状态都是功能的可能子集，操作包括选择功能并将其添加到子集。[18] 处理这个问题的方法是在一个有向无环图上进行探索，该图表示不同转换版本的并通过Qlearning学习在给定预算下探索可用特征工程选择的有效策略。[19] 将此任务形式化为异构转换图（HTG）上的优化问题。它在HTG上提出了一个深度Q学习，以支持细粒度和广义FE策略的有效学习，这些策略可以从集合中传递工程“良好”特性的知识将数据集转换为其他看不见的数据集。



#  交互式AutoML

## Alpine Meadow

- Alpine Meadow : A System for Interactive AutoML

AutoML已被没有机器学习知识的领域专家广泛用于从数据中提取可操作的见解。然而，以前的研究只强调最终答案的高准确性，这可能需要几个小时甚至几天才能完成。在本文中，我们介绍了Alpine Meadow，第一个交互式自动机器学习工具。使我们的系统独一无二的不仅仅是对交互性的关注，还有系统和算法设计方法的结合。我们设计了新的AutoML搜索算法，并共同设计了执行运行时，以高效地执行ML工作负载。

我们在300个数据集上评估了我们的系统，并与其他AutoML工具（包括当前的NIPS赢家）以及专家解决方案进行了比较。Alpine Meadow不仅能够显著优于其他AutoML系统，没有交互延迟，而且在80%的情况下，在从未见过的数据集上，我们的系统优于专家解决方案。

**Contributions：**

(1) We present a novel architecture of an AutoML system with interactive responses;

 (2) We show rule-based optimization, can be combined with multi-armed bandits, Bayesian optimization and meta-learning to find more efficiently the best ML pipeline for a given problem. We devise an adaptive pipeline selection algorithm to prune unpromising pipelines early.

(3) We co-design the runtime with the decision process and decouple these two components to achieve better scalability, and devise sampling, caching and scheduling strategies to further promote interactivity.

(4) We show in our evaluation that Alpine Meadow significantly outperforms other AutoML systems while — in contrast to the other systems — provides interactive latencies on over 300 real world datasets.

Furthermore, Alpine Meadow outperforms expert solutions in 80% of the cases for datasets we have never seen before



![image-20210916173841268](C:\Users\jianzh\AppData\Roaming\Typora\typora-user-images\image-20210916173841268.png)



![image-20210916173915140](C:\Users\jianzh\AppData\Roaming\Typora\typora-user-images\image-20210916173915140.png)



![image-20210916174548646](C:\Users\jianzh\AppData\Roaming\Typora\typora-user-images\image-20210916174548646.png)



**建议细读**： 建议

## AutoAIViz

- AutoAIViz: Opening the Blackbox of Automated Artificial Intelligence with Conditional Parallel Coordinates

**ABSTRACT**

Artificial Intelligence (AI) can now automate the algorithm selection, feature engineering, and hyperparameter tuning steps in a machine learning workflow. Commonly known as AutoML or AutoAI, these technologies aim to relieve data scientists from the tedious manual work. However, today’s AutoAI systems often present only limited to no information about the process of how they select and generate model results. Thus, users often do not understand the process, neither do they trust the outputs. In this short paper, we provide a first user evaluation by 10 data scientists of an experimental system, AutoAIViz, that aims to visualize AutoAI’s model generation process. We find that the proposed system helps users to complete the data science tasks, and increases their understanding, toward the goal of increasing trust in the AutoAI system

人工智能（AI）现在可以自动完成机器学习工作流中的算法选择、特征工程和超参数调整步骤。这些技术通常被称为AutoML或AutoAI，旨在将数据科学家从繁琐的手工工作中解放出来。然而，今天的AutoAI系统通常只提供关于如何选择和生成模型结果，而没有提供过程信息。因此，用户通常不了解流程，也不信任输出。在这篇短文中，我们提供了10位数据科学家对实验系统AutoAIViz的首次用户评估，该系统旨在可视化AutoAI的模型生成过程。我们发现，建议的系统有助于用户完成数据科学任务，并增加他们的理解。

**建议细读**： 不建议

## AutoDS

- AutoDS: Towards Human-Centered Automation of Data Science

**ABSTRACT** Data science (DS) projects often follow a lifecycle that consists of laborious tasks for data scientists and domain experts (e.g., data exploration, model training, etc.). Only till recently, machine learning(ML) researchers have developed promising automation techniques to aid data workers in these tasks. This paper introduces AutoDS, an automated machine learning (AutoML) system that aims to leverage the latest ML automation techniques to support data science projects. Data workers only need to upload their dataset, then the system can automatically suggest ML configurations, preprocess data, select algorithm, and train the model. These suggestions are presented to the user via a web-based graphical user interface and a notebook-based programming user interface. We studied AutoDS with 30 professional data scientists, where one group used AutoDS, and the other did not, to complete a data science project. As expected, AutoDS improves productivity; Yet surprisingly, we find that the models produced by the AutoDS group have higher quality and less errors, but lower human confidence scores. We reflect on the findings by presenting design implications for incorporating automation techniques into human work in the data science lifecycle.

**Contributions**•

- We present an automated data science prototype system with various novel feature designs (e.g., end-to-end, human-inthe-loop, and automatically exporting models to notebooks);
-  We offer a systematic investigation of user interaction and perceptions of using an AutoDS system in solving a data science task, which yields many expected (e.g., higher productivity) and novel findings (e.g., performance is not equal to confidence, and shift of focus);
-  Based on these novel findings, we present design implications for AutoDS systems to better fit into data science workers’ workflow

**建议细读**： 建议

### Human-in-the-Loop AutoDS

AutoDS refers to a group of technologies that can automate the manual processes of data pre-processing, feature engineering, model selection, etc. [71]. Several technologies have been developed with different specializations.

For example, Google has developed a set of AutoDS products under the umbrella of Cloud AutoML, such that even non-technical users can build models for visual, text, and tabular data [19].

 H2O is java-based software for data modelling that provides a python module, which data scientists can import into their code file in order to use the automation capability.

# 特征工程

## 背景

当前， 机器学习技术广泛应用于各个领域， 例如 推荐系统[^1-3] ,  欺诈检测[^4-6], 广告推荐， 癌症诊断等， 在机器学习技术的帮助下，这些领域取得了显著的提升。

一般来说，要构建一个机器学习系统，通常需要一个专业而复杂的ML流程，它通常包括数据准备、数据预处理、特征工程、模型生成和模型评估等。特征工程是将原始数据转换为有用特征的过程。人们普遍认为，机器学习方法的性能在很大程度上取决于特征的质量，生成良好的特征集成为追求高性能的关键步骤。

然而，令人沮丧的是，特征工程通常是人工干预ML流程中最不可或缺的部分，因为人类的直觉和经验是非常必要的，因此，它变得单调乏味、任务相关且具有挑战性。因此，大多数机器学习工程师在构建机器学习系统时都会花费大量精力来获取有用的特征。

另一方面，随着工业任务中对ML技术需求的不断增长，在所有领域中手动执行特征工程变得不切实际。这促进了自动特征工程的诞生，这是自动机器学习（AutoML）[13]–[16]的一个重要主题。自动特征工程的发展不仅可以将机器学习工程师从繁重乏味的过程中解放出来，而且为机器学习技术在越来越多的应用提供了动力。

在工业任务中，真实业务数据的大小总是非常巨大，这对空间和时间复杂性提出了极高的要求。同时，由于业务的快速变化，对算法的灵活性和可扩展性也提出了很高的要求。此外，还有更多的要求需要解决。

*在完整的机器学习 Pipeline 中，特征工程通常占据了数据科学家很大一部分的精力，一方面是因为特征工程能够显著提升模型性能，高质量的特征能够大大简化模型复杂度，让模型变得高效且易理解、易维护，如果没有具有代表性的特征，即使有最佳的超参数，模型性能也会十分糟糕；另一方面因为扎实的特征工程不仅要求数据科学家拥有坚实的数据挖掘和统计背景，还要求具有相关的领域知识，这对数据科学家提出了很高的要求。因此，自动化特征工程(Automated Feature Engineering, AutoFE)的概念应运而生，特征工程的自动化旨在减少数据科学家反复试错带来的时间成本，辅助他们更高效地尝试想法与排除错误，扩大特征空间的搜索广度和深度，改善模型表现。*

AutoFE可以看作自动化机器学习技术(Automated Machine Learning, AutoML)的一环，AutoML主要包括自动化特征工程、自动化模型选择、自动化超参调优三块，根据侧重点的不同，AutoML又分为传统AutoML和深度AutoML，在数据挖掘领域，AutoML偏向于特征工程和模型选择(如Auto-sklearn)，而在深度学习领域，AutoML偏向于超参优化和神经架构搜索(如Auto-Keras)。这样的差异很大程度与深度学习方法强大的特征融合能力有关，实际上深度学习也被称为无监督特征学习(Unsupervised Feature Learning)，但是深度学习仅仅是AutoFE的一种可能的解决方案，深度学习虽然避免了繁琐的特征工程，但又带来了大量的超参调优工作，因此深度学习工程师常被戏称为“调参工程师”。

- 适用性强：适应性强的工具意味着用户友好且易于使用。自动特征工程算法的性能不应依赖于大量的超参数，或者其超参数配置之一可以应用于不同的数据集。
- 分布式计算：实际业务任务中的样本和功能数量相当大，这使得分布式计算成为必要。自动特征工程算法的大部分部分应该能够并行计算。
- 实时推理：实时推理涉及许多实际业务。在这种情况下，一旦输入了一个实例，就应该立即生成特征，并随后执行预测。



## 特征工程的主要步骤

特征工程的主要步骤包括：
特征构造：从原始数据构造新特征。特征构建需要对数据和要解决的潜在问题有很好的了解。
特征选择：选择与模型训练问题最相关的可用特征子集。
特征提取：通过组合和减少现有特征的数量来创建新的和更有用的特征。主成分分析（PCA）和嵌入是一些特征提取方法。

## 特征工程常见技术

特征工程的一些常见技术包括：

### One-Hot编码

大多数ML算法不能处理类别（字符）型数据，需要转化为数值。例如，如果表格数据集中有一个“颜色”列，且观测值为“红色”、“蓝色”和“绿色”，则可能需要将这些值转换为数值，以便模型更好地处理。然而，标记“红色”=1、“蓝色”=2，“绿色”=3是不够的，因为颜色之间没有顺序关系（即蓝色不是红色的两倍）。
相反，一个热编码涉及创建两列“红色”和“蓝色”。如果观察值为红色，则在“红色”列中取1，在“蓝色”列中取0。如果它是绿色的，则在两列中都取0，并且模型推断它是绿色的。

### 对数变换

对数变换是将列中的每个值替换为其对数。这是一种处理扭曲数据的有用方法。对数变换可以将分布转换为近似正态分布，并减少异常值的影响。例如，拟合一个线性模型会在转换后给出更准确的结果，因为两个变量之间的关系在转换后更接近线性。

![img](https://research.aimultiple.com/wp-content/uploads/2021/06/logtransform.gif)

### 异常值处理

离群值是指与其他观测值相距较远的观测值。它们可能是由于错误造成的，也可能是真实的观察结果。无论是什么原因，识别它们都很重要，因为机器学习模型对值的范围和分布很敏感。下图展示了异常值如何改变线性模型的拟合。

![img](https://research.aimultiple.com/wp-content/uploads/2021/06/outliers_effect-1160x580.png)

异常值处理方法取决于数据集。假设您使用一个地区房价数据集。如果你知道某个地区的房价不能超过某个特定的数值，如果有人观察到房价高于这个数值，你可以这样做：

- 删除这些观察，因为它们可能是错误的
- 用属性的平均值或中值替换异常

### 分桶

 分桶是将观察结果分组到“bins”下。将个人年龄转换为年龄组或根据其所在大陆对国家进行分组就是分类的例子。装箱的决定取决于您试图从数据中获得什么。

### 处理缺失值

缺失值是数据处理过程中最常见的问题之一。这可能是由于错误、数据不可用或隐私原因造成的。机器学习算法的很大一部分是为了处理完整的数据而设计的，因此您应该处理数据集中缺失的值。如果没有，模型可以自动删除那些可能不受欢迎的观察值。
要处理缺少的值，可以：
-　如果属性是数值型，用属性的平均值/中值填充缺失的观测值。
-　如果属性是分类的，则填充最频繁的类别。
-　使用ML算法捕获数据结构并相应地填充缺失值。
-　如果有关于数据的领域知识，可以根据经验预测缺少的值。
-　删除丢失的观察结果。



## 自动化特征工程

### 摘要

特征生成是指利用已有特征衍生对模型有用的特征，是特征工程中的重要组成部分。衍生的特征可分为低阶特征和高阶特征，二者结合更有利于提升模型效果。这一步可能比实际上使用的模型更重要，因为一个机器学习算法只能从我们给定的数据中学习，所以构造一个和任务相关的特征是至关重要的，参见优质论文《A Few Useful Things to Know about Machine Learning》。

通常，特征工程是一个冗长的人工过程，依赖于领域知识、直觉和数据操作。这个过程可能是极其枯燥的，同时最终得到的特征将会受到人的主观性和时间的限制。特征工程自动化旨在通过从数据集中自动构造候选特征，并从中选择最优特征用于训练来帮助数据科学家。

机器学习越来越多地从人工设计模型转向使用 H20、TPOT 和 Auto-sklearn 等工具自动优化的工具。这些库以及随机搜索（参见《Random Search for Hyper-Parameter Optimization》）等方法旨在通过寻找匹配数据集的最优模型来简化模型选择和机器学习调优过程，而几乎不需要任何人工干预。然而，特征工程作为机器学习流程中可能最有价值的一个方面，却被现在的自动化机器学习工具所忽略。

针对这一现状，提出了新的AutoML框架Auto-Tabular，通过特征交叉组合和聚合操作进行特征交互，通过神经网络生成高阶特征，合并低阶和高阶特征，有效提升模型效果。在Kaggle和OpenML的数据集进行实验，结果表明，与Auto-Sklearn、mljar和 AutoGluon-Tabular等框架相比，Auto-Tabular具有更优异的性能。

自动化特征工程(Automated Feature Engineering, AutoFE)主要涉及了feature extraction和feature selection两个子问题. 目前feature selection已经有较多的解决方案, 所以主要谈谈feature extraction.

- 类别特征比较注重特征交叉这一步, 并且因为稀疏且维度高, 一般自动化构造主要是奔着效率去的(比如 AutoCross, 来自第四范式的工作);
- 时序特征比较特殊, 特征算子是在时间维度上利用滑动窗口等方法进行统计量的计算, 目前的自动化库有tsfresh这些;
- 适用于关系型数据(多表join情况)的自动化特征工程, 比如featuretools这个库;
- 数值特征比较经典, 因为可用的算子较多且特征维度大, 一般会导致组合爆炸的问题, 这方面的工作也很多, 从expand-reduction这种暴力搜索, 到现在微软的NFS这类模仿NAS, 利用深度学习进行搜索, 但还是面临一个效率的问题.

我看来, 自动化特征工程出发点是通过某个strategy来替代人类专家, 针对特定的问题域(domain), 在设计好的特征空间中搜索特征, 且通过某种方式评估特征的效果. 因此主要需要解决三个问题:

- 搜索空间: 不同问题域一般有不同的搜索空间, 需要人为进行定义
- 搜索策略: 暴力策略 / 启发式搜索 / neural-based
- 评估方式: 直接评估(将特征加入数据集, 训练学习器, 以学习器效果作为评价) / 简介评估(一些信息论相关的评价 或 利用树模型的特征重要性等)
- 目前来说, 自动化特征工程应该关注于设计更高效的搜索策略, 比如仿照NAS, 以梯度信息 / agent控制器等, 对搜索过程进行指导.



### GBDT+LR：利用树模型自动化特征工程

![img](https://pic2.zhimg.com/80/v2-e26ce7bd3df33a61010189049cb856c5_1440w.jpg)



在FM/FFM之外，2014年Facebook提出的GBDT+LR 是实现用模型自动化特征工程的一项经典之作， 其本质上是通过Boosting Tree模型本身的特征组合能力来替代原先算法工程师们手动组合特征的过程。GBDT等这类Boosting Tree模型本身具备了特征筛选能力（每次分裂选取增益最大的分裂特征与分裂点）以及高阶特征组合能力（树模型天然优势），因此通过GBDT来自动生成特征向量就成了一个非常自然的思路。

GBDT+LR是级联模型，主要思路是先训练一个GBDT模型，然后利用训练好的GBDT对输入样本进行特征变换，最后把变换得到的特征向量作为LR模型的输入，对LR模型进行训练。

举个具体例子，上图是由两颗子树构成的GBDT，将样本x输入，假设经过自上而下的节点判别，左边子树落入到第二个叶子节点，右边子树落入到第一个叶子节点，那么两颗子树分别得到向量[1, 0, 0]和[0, 1]，将各子树的输出向量进行concat，得到的[1, 0, 0, 1, 0]就是由GBDT变换得到的特征向量，最后将此向量作为LR模型的输入，训练LR的权重w0, w1，...。在此过程中，从根节点到叶子节点的遍历路径，就是一种特征组合，LR模型参数wi，对应一种特征组合的权重。



### tsfresh 构建时间序列特征

tsfresh是基于可伸缩假设检验的时间序列特征提取工具。该包包含多种特征提取方法和鲁棒特征选择算法。

tsfresh可以自动地从时间序列中提取100多个特征。这些特征描述了时间序列的基本特征，如峰值数量、平均值或最大值，或更复杂的特征，如时间反转对称性统计量等。

 ![img](https://img2018.cnblogs.com/blog/1473228/201902/1473228-20190214204914311-1113301258.png)

这组特征可以用来在时间序列上构建统计或机器学习模型，例如在回归或分类任务中使用。

时间序列通常包含噪声、冗余或无关信息。因此，大部分提取出来的特征对当前的机器学习任务没有用处。为了避免提取不相关的特性，tsfresh包有一个内置的过滤过程。这个过滤过程评估每个特征对于手头的回归或分类任务的解释能力和重要性。它建立在完善的假设检验理论的基础上，采用了多种检验方法。

需要注意的是，在使用tsfresh提取特征时，需要提前把结构进行转换，一般上需转换为(None,2)的结构，例如下图所示：

![img](https://img2018.cnblogs.com/blog/1473228/201902/1473228-20190214204948088-1936775105.png)





### Featuretools 提取关联数据表特征

Featuretools使用一种称为深度特征合成（Deep Feature Synthesis，DFS）的算法，该算法遍历通过关系[数据库](https://cloud.tencent.com/solution/database?from=10680)的模式描述的关系路径。当DFS遍历这些路径时，它通过应用于数据的操作（包括和、平均值和计数）生成综合特征。例如，对来自给定字段client_id的事务列表应用sum操作，并将这些事务聚合到一个列中。尽管这是一个深度操作，但该算法可以遍历更深层的特征。Featuretools最大的优点是其可靠性和处理信息泄漏的能力，同时可以用来对时间序列数据进行处理。



**ExploreKit** is based on the intuition that highly informative features often result from manipulations of elementary ones, they identify common operators to transform each feature individually or combine several of them together. it uses these operators to generate many candidate features, and chooses the subset to add based on the empirical performance of models trained with candidate features added.

ExploreKit 基于直觉，即信息量大的特征通常来自于对基本特征的操作，它们识别常见的运算符来分别变换每个特征或将其中的几个特征组合在一起。它使用这些算子生成许多候选特征，并根据添加了候选特征的模型的经验性能选择要添加的子集。



![img](https://miro.medium.com/max/1483/1*cF-MjVs5VDgwPD2FTRXSUg.png)

参考：

[^1]: J. Davidson, B. Liebald, J. Liu, P. Nandy, T. V. Vleet, U. Gargi, S. Gupta, Y. He, M. Lambert, B. Livingston, and D. Sampath, “The youtube video recommendation system,” in Proceedings of the 2010 ACM Conference on Recommender Systems, RecSys 2010, Barcelona, Spain, September 26-30, 2010, 2010, pp. 293–296.
[^2]: X. He, J. Pan, O. Jin, T. Xu, B. Liu, T. Xu, Y. Shi, A. Atallah, R. Herbrich, S. Bowers, and J. Q. Candela, “Practical lessons from predicting clicks on ads at facebook,” in Proceedings of the Eighth International Workshop on Data Mining for Online Advertising, ADKDD 2014, August 24, 2014, New York City, New York, USA, 2014, pp. 5:1– 5:9.
