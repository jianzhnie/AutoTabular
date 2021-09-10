

# 应用于工业场景的大规模自动化特征工程技术

[TOC]

## 摘要1

特征工程是一种将原始数据列转换为有意义的内容的技术，有助于预测机器学习任务的结果。特征工程在机器学习生命周期中可能是一个非常乏味且通常最耗时的过程。在过去几十年中，机器学习在构建数据驱动的实验中发挥了重要作用。基于从各种来源提取的信息，可以识别新模式，更容易做出预测，更快更有效地做出决策。机器学习解决方案的专门应用需要数学、计算和统计等领域的专门知识，而且时间成本极高，在这一过程中极有可能出现任何人为错误。自动机器学习技术（AutoML）寻求自动化构建机器学习应用程序的部分过程， 使得非专家也能够执行此过程。这类问题的一个重要部分是特征工程部分，该部分在数据中创建转换，使其更能代表最终模型。本文系统地回顾了机器学习问题中的自动特征工程解决方案。主要目的是识别和分析现有的方法和技术，以便在机器学习问题的框架内执行自动特征工程步骤。



## 摘要2

机器学习技术已广泛应用于互联网公司的各种任务中，作为一种重要的驱动力，特征工程已被公认为构建机器学习系统的关键环节。近年来，人们越来越多地致力于自动特征工程方法的开发，以使大量繁琐的手工工作得以解放。然而，对于工业任务，这些方法的效率和可扩展性仍然远远不能令人满意。在本文中，我们提出了一种称为SAFE（Scalable Automatic Feature Engineering）的分阶段方法，它可以提供出色的效率和可扩展性，以及必要的可解释性和良好的性能。大量的实验结果表明，与其他方法相比，该方法具有显著的效率和竞争优势。此外，所提出的方法具有足够的可扩展性，确保了它能够部署在大规模工业任务中。



## 背景

当前， 机器学习技术广泛应用于各个领域， 例如 推荐系统[^1-3] ,  欺诈检测[^4-6], 广告推荐， 癌症诊断等， 在机器学习技术的帮助下，这些领域取得了显著的提升。

一般来说，要构建一个机器学习系统，通常需要一个专业而复杂的ML流程，它通常包括数据准备、特征工程、模型生成和模型评估等。人们普遍认为，机器学习方法的性能在很大程度上取决于特征的质量，生成良好的特征集成为追求高性能的关键步骤。因此，大多数机器学习工程师在构建机器学习系统时都会花费大量精力来获取有用的特征。

然而，令人沮丧的是，特征工程通常是人工干预ML流程中最不可或缺的部分，因为人类的直觉和经验是非常必要的，因此，它变得单调乏味、任务相关且具有挑战性，因此非常耗时。另一方面，随着工业任务中对ML技术需求的不断增长，在所有领域中手动执行特征工程变得不切实际。这促进了自动特征工程的诞生，这是自动机器学习（AutoML）[13]–[16]的一个重要主题。自动特征工程的发展不仅可以将机器学习工程师从繁重乏味的过程中解放出来，而且为机器学习技术在越来越多的应用提供了动力。

对于一个常规的有监督学习任务，问题可以表述为使用训练示例来寻找函数F:X→ Y，定义为返回获得最高分数的Y值：F（x）=arg max S（x，Y），其中x是输入空间，Y是输出空间，S: x×Y→ R是评分函数。自动特征工程的目标是学习特征表示ψ：X→ Z、 在原有特征x的基础上构造新的特征表示Z，以尽可能提高后续机器学习工具的性能。

目前，已经就这一方法进行了几项研究。举几个例子，一些方法使用基于强化学习的策略来执行自动特征工程[17]–[19]。这些方法需要多次尝试，并且有必要生成一个新的特征集，并在每一轮中对其进行评估，使其在工业任务中不可行。基于迁移学习或元学习的策略也被用于自动特征工程[20]，[21]。然而，为了训练这些方法，需要在不同的数据集上进行大量的实验，并且很难引入新的算子或增加父特征的数量。一些方法遵循生成选择程序[22]–[24]进行自动特征工程。然而，这些方法始终在特征生成阶段生成所有合法特征，然后从中选择一个子集特征，因此时间和空间复杂度极高，不适用于数据量大或特征维度高的任务。

在工业任务中，真实业务数据的大小总是非常巨大，这对空间和时间复杂性提出了极高的要求。同时，由于业务的快速变化，对算法的灵活性和可扩展性也提出了很高的要求。此外，还有更多的要求需要解决。



- 适用性强：适应性强的工具意味着用户友好且易于使用。自动特征工程算法的性能不应依赖于大量的超参数，或者其超参数配置之一可以应用于不同的数据集。
- 分布式计算：实际业务任务中的样本和功能数量相当大，这使得分布式计算成为必要。自动特征工程算法的大部分部分应该能够并行计算。
- 实时推理：实时推理涉及许多实际业务。在这种情况下，一旦输入了一个实例，就应该立即生成特征，并随后执行预测。

在本文中，我们从典型的两阶段的角度来研究这个问题，并提出了一种称为SAFE（Scalable Automatic Feature Engineering）的方法来执行有效的自动特征工程，该方法包括特征生成阶段和特征选择阶段。我们保证计算效率、可扩展性和上述要求。
本文的主要贡献总结如下：

- 在特征生成阶段，与以往的方法不同，以往的方法侧重于使用什么操作员或如何生成所有合法特征，我们侧重于挖掘原始特征对，以更高的概率生成更有效的新特征，以提高效率。

- 在特征选择阶段，我们提出了一个特征选择 pipeline ，考虑了单个特征的能力、特征对的冗余度以及通过典型树模型评估的特征重要性。它适用于多个不同的业务数据集和各种机器学习算法。
- 我们已经在大数据集和多分类器上实验证明了我们的算法的优势。与原始特征空间相比，预测精度平均提高了6.50%



## 相关研究



作为自动机器学习[13]-[16]中一个不容忽视的问题，自动特征工程近年来受到了广泛关注，并从不同角度提出了许多方法来解决这一任务[17]-[24]，[26]-[28]。在这一部分中，我们主要讨论了三种典型的策略，包括生成选择策略、基于强化学习的策略和基于迁移学习的策略。给定一个有监督的学习数据集，自动特征工程的典型方法是遵循生成-选择过程。FICUS算法[26]通过构造一组候选特征进行初始化，并迭代改进，直到计算预算耗尽。在每次迭代过程中，它执行波束搜索来构造新特征，并通常使用基于决策树中信息增益的启发式度量来选择特征。TFC[27]还通过迭代框架解决了这一任务。在每次迭代中，它基于当前特征库和所有可用的操作符生成所有合法特征，然后使用信息增益从所有候选特征中选择最佳特征，并将其保留为新特征库。有了这个框架，随着迭代的进行，可以获得更高阶的特征组合。然而，每次迭代中的穷举搜索会导致特征空间的组合爆炸，从而导致这种方法不可扩展。为了避免穷举搜索，提出了基于学习的方法，如FCTree算法[28]。FCTree通过对原始特征应用多个顺序变换来训练决策树并执行特征生成，并根据决策树每个节点上的信息增益选择特征。一旦建立了一棵树，在内部决策节点上选择的特征就会被删除用于获取构造的特征。[24]是一种基于回归的算法，它通过挖掘成对特征关联、识别每对特征之间的线性或非线性关系、应用回归并选择稳定的关系和提高预测性能来学习表示。由于时间的消耗，这些算法总是遇到性能和可伸缩性瓶颈如果没有巧妙的设计，特征生成和选择过程中的资源可能非常不令人满意。

本文还探讨了强化学习策略。
[17] 将特征选择形式化为强化学习问题，并引入蒙特卡罗树搜索的适应性。这里，选择可用功能子集的问题被转换为单人游戏，其状态都是功能的可能子集，操作包括选择功能并将其添加到子集。[18] 处理这个问题的方法是在一个有向无环图上进行探索，该图表示不同转换版本的并通过Qlearning学习在给定预算下探索可用特征工程选择的有效策略。[19] 将此任务形式化为异构转换图（HTG）上的优化问题。它在HTG上提出了一个深度Q学习，以支持细粒度和广义FE策略的有效学习，这些策略可以从集合中传递工程“良好”特性的知识将数据集转换为其他看不见的数据集。

参考：

[^1]: J. Davidson, B. Liebald, J. Liu, P. Nandy, T. V. Vleet, U. Gargi, S. Gupta, Y. He, M. Lambert, B. Livingston, and D. Sampath, “The youtube video recommendation system,” in Proceedings of the 2010 ACM Conference on Recommender Systems, RecSys 2010, Barcelona, Spain, September 26-30, 2010, 2010, pp. 293–296.
[^2]: X. He, J. Pan, O. Jin, T. Xu, B. Liu, T. Xu, Y. Shi, A. Atallah, R. Herbrich, S. Bowers, and J. Q. Candela, “Practical lessons from predicting clicks on ads at facebook,” in Proceedings of the Eighth International Workshop on Data Mining for Online Advertising, ADKDD 2014, August 24, 2014, New York City, New York, USA, 2014, pp. 5:1– 5:9.



## ExploreKit

**ExploreKit** is based on the intuition that highly informative features often result from manipulations of elementary ones, they identify common operators to transform each feature individually or combine several of them together. it uses these operators to generate many candidate features, and chooses the subset to add based on the empirical performance of models trained with candidate features added.

ExploreKit 基于直觉，即信息量大的特征通常来自于对基本特征的操作，它们识别常见的运算符来分别变换每个特征或将其中的几个特征组合在一起。它使用这些算子生成许多候选特征，并根据添加了候选特征的模型的经验性能选择要添加的子集。



![img](https://miro.medium.com/max/1483/1*cF-MjVs5VDgwPD2FTRXSUg.png)

**Advantages :**

· Uses meta learning to rank candidate features rather than running feature selection on all created features which can sometimes be very large.

**Limitations :**

· No open source implementation either in Python or R.



## **OneBM** 

**OneBM** works directly with multiple raw tables in a database. It joins the tables incrementally, following different paths on the relational graph. It automatically identifies data types of the joint results, including simple data types (numerical or categorical) and complex data types (set of numbers, set of categories, sequences, time series and texts), and applies corresponding pre-defined feature engineering techniques on the given types. In doing so, new feature engineering techniques could be plugged in via an interface with its feature extractor modules to extract desired types of features in specific domain. it supports data scientists by automating the most popular feature engineering techniques on different structured and unstructured data.

Feature selection is used to remove irrelevant features extracted in the prior steps. First, duplicated features are removed. Second, if the training and test data have an implicit order defined by a column, e.g. timestamp, then drift features are detected by comparing the distribution between the value of features in the training and a validation set. If two distributions are different, the feature is identified as a drift feature which may cause over-fitting. Drift features are all removed from the feature set. Besides, we also employ Chi-square hypothesis testing to test whether there exists a dependency between a feature and the target variable. Features that are marginally independent from the target variable are removed.



**Advantages :**

· Works well with both relational as well as non-relational data.

· Generates simple as well as complex features as compared to FeatureTools.

· Tested in Kaggle competitions where it out performed state of the art models.

· Can be used to create feature for big data also.



[^1]:
[^2]:
[^3]:

