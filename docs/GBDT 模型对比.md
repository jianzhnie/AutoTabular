

# XGBoost, LightGBM or CatBoost



## Splits

在模型训练之前，所有算法都会为所有特征创建feature-split 的 pairs 。例如：（年龄<5岁），（年龄>10岁），（数量>500岁）。这些特征分割对都是基于直方图构建的，在学习过程中用作可能的分割节点。这种预处理方法比精确的贪婪算法更快，贪婪算法线性地枚举连续特征的所有可能分割。

首先让我们理解预分类算法如何工作：

- 对于每个节点，遍历所有特征
- 对于每个特征，根据特征值分类样例
- 进行线性扫描，根据当前特征的基本信息增益，确定最优分割
- 选取所有特征分割结果中最好的一个

在过滤数据样例寻找分割值时：

- **XGBoost** 通过预分类算法和直方图算法来确定最优分割。

- **LightGBM** 使用的是全新的技术：基于梯度的单边采样（GOSS）；

  lightGBM提供了基于梯度的单边采样（GOSS），它使用具有大梯度（即，大误差）的所有实例和具有小梯度的实例的随机样本选择分割。为了在计算信息增益时保持相同的数据分布，GOSS为具有小梯度的数据实例引入了一个常数乘数。因此，GOSS在通过减少数据实例的数量来提高速度和保持学习决策树的准确性之间实现了良好的平衡。

- **Catboost** 提供了一种称为最小方差采样（MVS）的新技术

  它是随机梯度提升的加权采样版本。在这种技术中，加权采样发生在树级别，而不是拆分级别。对每棵推进树的观测值进行采样，以最大限度地提高分割评分的准确性。

**XGboost 没有使用任何加权采样技术，这使得其拆分过程比GOSS和MVS慢。**简单说，直方图算法在某个特征上将所有数据点划分到离散区域，并通过使用这些离散区域来确定直方图的分割值。虽然在计算速度上，和需要在预分类特征值上遍历所有可能的分割点的预分类算法相比，直方图算法的效率更高，但和 GOSS 算法相比，其速度仍然更慢。

### **为什么 GOSS 方法如此高效？**

在 Adaboost 中，样本权重是展示样本重要性的很好的指标。但在梯度提升决策树（GBDT）中，并没有天然的样本权重，因此 Adaboost 所使用的采样方法在这里就不能直接使用了，这时我们就需要基于梯度的采样方法。

梯度表征损失函数切线的倾斜程度，所以自然推理到，如果在某些意义上数据点的梯度非常大，那么这些样本对于求解最优分割点而言就非常重要，因为算其损失更高。

GOSS 保留所有的大梯度样例，并在小梯度样例上采取随机抽样。比如，假如有 50 万行数据，其中 1 万行数据的梯度较大，那么我的算法就会选择（这 1 万行梯度很大的数据+x% 从剩余 49 万行中随机抽取的结果）。如果 x 取 10%，那么最后选取的结果就是通过确定分割值得到的，从 50 万行中抽取的 5.9 万行。

在这里有一个基本假设：如果训练集中的训练样例梯度很小，那么算法在这个训练集上的训练误差就会很小，因为训练已经完成了。

为了使用相同的数据分布，在计算信息增益时，GOSS 在小梯度数据样例上引入一个常数因子。因此，GOSS 在减少数据样例数量与保持已学习决策树的准确度之间取得了很好的平衡。



![img](https://pic3.zhimg.com/v2-5a05e40c18467a2ac5703b7088de4d92_b.jpg)



高梯度/误差的叶子，用于 LGBM 中的进一步增长



## Leaf growth



![img](https://miro.medium.com/max/1575/1*E006sjlIjabDJ3jNixRSnA.png)

**Catboost** grows a balanced tree. In each level of such a tree, the feature-split pair that brings to the lowest loss (according to a penalty function) is selected and is used for all the level’s nodes. It is possible to change its policy using the *grow-policy* parameter.

**LightGBM** uses leaf-wise (best-first) tree growth. It chooses to grow the leaf that minimizes the loss, allowing a growth of an imbalanced tree. Because it doesn’t grow level-wise, but leaf-wise, overfitting can happen when data is small. In these cases, it is important to control the tree depth.

**XGboost** splits up to the specified *max_depth* hyperparameter and then starts pruning the tree backwards and removes splits beyond which there is no positive gain. It uses this approach since sometimes a split of no loss reduction may be followed by a split with loss reduction. XGBoost can also perform leaf-wise tree growth (as LightGBM).

## Missing values handling

**Catboost** has two modes for processing missing values, “Min” and “Max”. In “Min”, missing values are processed as the minimum value for a feature (they are given a value that is less than all existing values). This way, it is guaranteed that a split that separates missing values from all other values is considered when selecting splits. “Max” works exactly the same as “Min”, only with maximum values.

In **LightGBM** and **XGBoost** missing values will be allocated to the side that reduces the loss in each split.

## Feature importance methods

**Catboost** has two methods: The first is “PredictionValuesChange”. For each feature, PredictionValuesChange shows how much, on average, the prediction changes if the feature value changes. A feature would have a greater importance when a change in the feature value causes a big change in the predicted value. This is the default feature importance calculation method for non-ranking metrics. The second method is “LossFunctionChange”. This type of feature importance can be used for any model, but is particularly useful for ranking models. For each feature the value represents the difference between the loss value of the model with this feature and without it. Since it is computationally expensive to retrain the model without one of the features, this model is built approximately using the original model with this feature removed from all the trees in the ensemble. The calculation of this feature importance requires a dataset.

**LightGBM** and **XGBoost** have two similar methods: The first is “Gain” which is the improvement in accuracy (or total gain) brought by a feature to the branches it is on. The second method has a different name in each package: “split” (LightGBM) and “Frequency”/”Weight” (XGBoost). This method calculates the relative number of times a particular feature occurs in all splits of the model’s trees. This method can be biased by categorical features with a large number of categories.

**XGBoost** has one more method, “Coverage”, which is the relative number of observations related to a feature. For each feature, we count the number of observations used to decide the leaf node for.

## Categorical features handling

**Catboost** uses a combination of one-hot encoding and an advanced mean encoding. For features with low number of categories, it uses one-hot encoding. The maximum number of categories for one-hot encoding can be controlled by the *one_hot_max_size* parameter. For the remaining categorical columns, CatBoost uses an efficient method of encoding, which is similar to mean encoding but with an additional mechanism aimed at reducing overfitting. Using CatBoost’s categorical encoding comes with a downside of a slower model. We won’t go into how exactly their encoding works, so for more details see CatBoost’s documentation.

**LightGBM** splits categorical features by partitioning their categories into 2 subsets. The basic idea is to sort the categories according to the training objective at each split. From our experience, this method does not necessarily improve the LightGBM model. It has comparable (and sometimes worse) performance than other methods (for example, target or label encoding).

**XGBoost** doesn’t have an inbuilt method for categorical features. Encoding (one-hot, target encoding, etc.) should be performed by the user.



### **CatBoost**

Catboost 结合使用一种独热编码和一种更高级的均值编码。对于类别数较少的功能，它使用一个独热编码。一个独热编码的最大类别数可由 `one_hot_max_size`参数控制。对于其余的分类变量，CatBoost 使用了一种有效的编码方法，这种方法和均值编码类似，但有一种减少过拟合的附加机制。

![img](https://pic3.zhimg.com/v2-db9b088dfc8df09fe9dee753c2daed22_b.jpg)’

它的具体实现方法如下：

1. 将输入样本集随机排序，并生成多组随机排列的情况。

2. 将浮点型或属性值标记转化为整数。

3. 将所有的分类特征值结果都根据以下公式，转化为数值结果。



![img](https://pic3.zhimg.com/v2-ea103a4657076960d8ed62a7af3ff172_b.jpg)



其中 CountInClass 表示在当前分类特征值中，有多少样本的标记值是「1」；Prior 是分子的初始值，根据初始参数确定。TotalCount 是在所有样本中（包含当前样本），和当前样本具有相同的分类特征值的样本数量。

可以用下面的数学公式表示：

![img](https://pic4.zhimg.com/v2-459515a4d098da506c60f742204e702f_b.jpg)





CatBoost 可通过传递分类变量指标，进而通过独热最大量得到独热编码形式的结果（独热最大量：在所有特征上，对小于等于某个给定参数值的分类的变量列使用独热编码）。

如果在 CatBoost 语句中没有设置「跳过」，CatBoost 就会将所有列当作数值变量处理。

注意，如果某一列数据中包含字符串值，CatBoost 算法就会抛出错误。另外，带有默认值的 int 型变量也会默认被当成数值数据处理。在 CatBoost 中，必须对变量进行声明，才可以让算法将其作为分类变量处理。

### **LightGBM**

和 CatBoost 类似，LighGBM 也可以通过使用特征名称的输入来处理属性数据；它没有对数据进行独热编码，因此速度比独热编码快得多。LGBM 使用了一个特殊的算法来确定属性特征的分割值。

![img](https://pic3.zhimg.com/v2-33c024104f0ef117d057833f59e18ec2_b.jpg)

注意，在建立适用于 LGBM 的数据集之前，需要将分类变量转化为整型变量；此算法不允许将字符串数据传给分类变量参数。

### **XGBoost**

和 CatBoost 以及 LGBM 算法不同，XGBoost 本身无法处理分类变量，而是像随机森林一样，只接受数值数据。因此在将分类数据传入 XGBoost 之前，必须通过各种编码方式：例如标记编码、均值编码或独热编码对数据进行处理。

## ##**超参数中的相似性**

所有的这些模型都需要调节大量参数，但我们只谈论其中重要的。以下是将不同算法中的重要参数按照功能进行整理的表格。

![img](https://pic3.zhimg.com/v2-eb51dd20913df9ff1d4140eb5c9e9cce_b.jpg)
