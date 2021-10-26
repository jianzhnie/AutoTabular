# 数据挖掘竞赛利器——TabNet模型浅析



随着深度神经网络的不断发展，DNN在图像、文本和语音等类型的数据上都有了广泛的应用，然而对于同样非常常见的一种数据——表格数据，DNN却似乎并没有取得像它在其他领域那么大的成功。参加过Kaggle等数据挖掘竞赛的同学应该都知道，对于采用表格数据的任务，基本都是决策树模型的主场，像XGBoost和LightGBM这类提升（Boosting）树模型已经成为了现在数据挖掘比赛中的标配。相比于DNN，这类树模型好处主要有：

- 模型的决策流形（decision manifolds）是可以看成是超平面边界的，对于表格数据的效果很好
- 可以根据决策树追溯推断过程，可解释性较好
- 训练起来更快

而对于DNN，它的优势在于：

- 类似于图像和文本，可以对表格数据进行编码（encode），从而得到一个能够表征表格数据的方法，这种表征学习（representation learning）可以用在很多地方
- 可以减少对于特征工程（feature engineering）的依赖（相信打过比赛的同学都知道这有多重要）
- 可以通过online learning的方式来更新模型，而树模型只能用整个数据集重新训练

然而对于传统的DNN，一味地堆叠网络层很容易导致模型过参数化（overparametrized），导致DNN在表格数据集上表现并不尽如人意。因此，如果能够设计这样一种DNN，它既吸收了树模型的长处，又继承了DNN的优点，那么这样的模型无疑是针对于表格数据的一大利器，而这次介绍的论文就巧妙地设计出了这样的模型——TabNet，它**在保留DNN的end-to-end和representation learning特点的基础上，还拥有了树模型的可解释性和稀疏特征选择的优点**，这使得它在具备DNN优点的同时，在表格数据任务上也可以和目前主流的树模型相媲美，接下来我们就开始具体介绍TabNet。

## 用DNN构造决策树

既然想要让DNN具有树模型的优点，那么我们首先需要解决的一个问题就是：**如何构建一个与树模型具有相似决策流形的神经网络？**下图是一个决策树流形的简单示例。

![img](https://pic4.zhimg.com/80/v2-aec8d6ddeb8cbed899d40b6d1718998b_1440w.jpg)

这个决策流形很容易理解，输入两个特征 ![[公式]](https://www.zhihu.com/equation?tex=x_1) 和 ![[公式]](https://www.zhihu.com/equation?tex=x_2) ，决策树分别以 ![[公式]](https://www.zhihu.com/equation?tex=a) 和 ![[公式]](https://www.zhihu.com/equation?tex=d) 为阈值来对他们进行划分，这样就得到了图中所示的决策流形。那么我们如何用神经网络来构建出一个类似的决策流形呢？论文给出了一种方法，如下图所示。

![img](https://pic1.zhimg.com/80/v2-22d876a63f96fe1db4e8472f11f35380_1440w.jpg)

分析一下这个神经网络的流程，输入是特征向量 ![[公式]](https://www.zhihu.com/equation?tex=%5Bx_1%2Cx_2%5D) ，首先分别通过两个Mask层来将 ![[公式]](https://www.zhihu.com/equation?tex=x_1) 和 ![[公式]](https://www.zhihu.com/equation?tex=x_2) 单独筛选出来，然后通过一个weight和bias都被专门设定过的全连接层，并将两个FC层的输出通过ReLU激活函数后相加起来，最后经过一个Softmax激活函数作为最终输出。如果与决策树的流程进行对比，我们可以发现其实这个神经网络的每一层都对应着决策树的相应步骤：Mask层对应的是决策树中的**特征选择**，这个很好理解；FC层+RelU对应阈值判断，以 ![[公式]](https://www.zhihu.com/equation?tex=x_1) 为例，通过一个特定的FC层+ReLU之后，可以保证输出的向量里面只有一个是正值，其余全为0，而这就对应着决策树的**条件判断**；最后将所有条件判断的结果加起来，再通过一个Softmax层得到最终的输出。

这里说一下个人的理解，这个神经网络其实可以看做一个**加性模型**（Additive Model），它是由两个Mask+FC+ReLU组合相加构成的，而一个组合其实就是一个基本决策树，两棵树分别挑选的 ![[公式]](https://www.zhihu.com/equation?tex=x_1) 和 ![[公式]](https://www.zhihu.com/equation?tex=x_2) 作为划分特征，然后输出各自的结果，最后模型将两个结果加起来作为最终输出。这个输出向量可以理解成一个权重向量，它的每一维代表**某个条件判断的对最终决策影响的权重**；以图中的模型为例，对于输出向量 ![[公式]](https://www.zhihu.com/equation?tex=%5B0.1%2C0.4%2C0.2%2C0.3%5D%5ET) ，则第1维就代表如果条件 ![[公式]](https://www.zhihu.com/equation?tex=x_1%3Ea) 成立的话，它对于最终决策的影响占0.1权重，第3维就代表条件 ![[公式]](https://www.zhihu.com/equation?tex=x_2%3Ed) 成立占0.2权重，这一点与决策树有些不同，在决策树里，如果某个条件成立的话，那么它的权重就是1，所以对于决策树的加性模型，如果两个基本树的权重一样，若条件 ![[公式]](https://www.zhihu.com/equation?tex=x_1%3Ea) 和![[公式]](https://www.zhihu.com/equation?tex=x_2%3Ed) 成立，那么输出的向量就应该是![[公式]](https://www.zhihu.com/equation?tex=%5B0.5%2C0%2C0.5%2C0%5D%5ET)。因此，从这一点来看，图中的神经网络其实就是一个**“软”版本的决策树加性模型**。

## 模型架构

为了理解起来比较容易，上面的那个神经网络构造得比较简单，作为一个加性模型它只有两步，Mask层是人为设置好的，特征计算用的也是一个简单的FC层，而接下来介绍的TabNet就对这些地方做了改进，它的基本结构如下所示。

![img](https://pic3.zhimg.com/80/v2-24d90e6099f976b535f6fac2072fd5a6_1440w.jpg)

先不关注下方的Feature attribute输出，这个模型与前面的神经网络的框架是基本一致的。还是可以看成一个有很多step的加性模型，模型的输入是维度为 ![[公式]](https://www.zhihu.com/equation?tex=B+%5Ctimes+D)的Features，其中 ![[公式]](https://www.zhihu.com/equation?tex=B) 是batch size， ![[公式]](https://www.zhihu.com/equation?tex=D) 是feature的维数；而模型输出的是一个向量或一个数（分类或回归任务)。

现在具体讨论一下TabNet中各个层的作用：

- BN层：即batch normalization层
- Feature transformer层：其作用与之前的FC层类似，都是做**特征计算**，只不过复杂一些，结构如下所示：

![img](https://pic3.zhimg.com/80/v2-79e32895cdd49a0672e8e5b1bc0268f2_1440w.jpg)

其中GLU是gated linear unit，它其实就是在原始FC层的基础上再加上一个门控，其计算公式为 ![[公式]](https://www.zhihu.com/equation?tex=h%28X%29%3D%28W%2AX%2Bb%29%E2%8A%97%CF%83%28V%2AX%2Bc%29) 。

可以看出Feature transformer层由两个部分组成，前半部分层的参数是共享的，也就是说它们是在所有step上共同训练的；而后半部分则没有共享，在每一个step上是分开训练的。这样做是考虑到对于每一个step，输入的是同样的features（Mask层只是屏蔽了一些feature，并没有改变其它feature），因此我们可以先用同样的层来做特征计算的**共性部分**，之后再通过不同的层做每一个step的**特性部分**。另外，可以看到层中用到了残差连接，乘 ![[公式]](https://www.zhihu.com/equation?tex=%5Csqrt%7B0.5%7D) 是为了保证网络的稳定性。

- Split层：该层比较简单，就是将Feature transformer层输出的向量切成两部分，用公式表示为 ![[公式]](https://www.zhihu.com/equation?tex=%5Bd%5Bi%5D%2Ca%5Bi%5D%5D%3Df_i%28M%5Bi%5D%5Ccdot+f%29) ，其中 ![[公式]](https://www.zhihu.com/equation?tex=d%5Bi%5D) 将用于计算模型的最终输出，而 ![[公式]](https://www.zhihu.com/equation?tex=a%5Bi%5D) 则用来计算下一个step的Mask层。
- Attentive transformer层：该层的作用是根据上一个step的结果，计算出当前step的Mask层，具体结果如下：

![img](https://pic1.zhimg.com/80/v2-147835adfa59c88d75dc4d433425bc54_1440w.jpg)

其中Sparsemax层可以理解为Softmax的稀疏化版本，这里简要介绍一下，对它感兴趣的同学可以参看Sparsemax原论文。不像Softmax的平滑变换，Sparsemax通过直接将向量投影到一个simplex来实现稀疏化，其计算公式为： ![[公式]](https://www.zhihu.com/equation?tex=%5Ctext%7BSparsemax%7D%28z%29%3D%5Ctext%7Barg%7D%5Cmin_%7Bp%5Cin+%5CDelta%7D%5C%7C+p-z%5C%7C) 。

根据Attentive transformer层的结构，可以将其计算公式写做：

![[公式]](https://www.zhihu.com/equation?tex=M%5Bi%5D%3D%5Ctext%7BSparsemax%7D%5Cleft%28+P%5Bi-1%5D%5Ccdot+h_i%28a%5Bi-1%5D%29+%5Cright%29%5C%5C)

其中 ![[公式]](https://www.zhihu.com/equation?tex=a%5Bi-1%5D) 是上一个step中Split层划分出来的， ![[公式]](https://www.zhihu.com/equation?tex=h_i%28%5Ccdot%29) 代表FC+BN层，而 ![[公式]](https://www.zhihu.com/equation?tex=P%5Bi%5D) 是Prior scales项，其形式为： ![[公式]](https://www.zhihu.com/equation?tex=%5Cmathrm%7BP%7D%5B%5Cmathrm%7Bi%7D%5D%3D%5Cprod_%7Bj%3D1%7D%5E%7Bi%7D%28%5Cgamma-%5Cmathrm%7BM%7D%5B%5Cmathrm%7Bj%7D%5D%29) ，它用来表示某一个feature在之前的step中的**运用程度**，按照正常的直觉，如果一个feature已经在之前的step中用了很多次，那么它就不应该再被模型选中了，因此模型通过这个Prior scales项来减小这类feature的权重占比，从式子可以看出，如果令 ![[公式]](https://www.zhihu.com/equation?tex=%5Cgamma%3D1) ，那么每个feature只能被用一次，而当 ![[公式]](https://www.zhihu.com/equation?tex=%5Cgamma) 增加时，这个约束就会变“软”一些。最后因为得到的Mask矩阵是![[公式]](https://www.zhihu.com/equation?tex=B%5Ctimes+D) 维的，根据Sparsemax的性质有 ![[公式]](https://www.zhihu.com/equation?tex=%5Csum_%7Bj-1%7D%5E%7BD%7DM%5Bi%5D_%7Bb%2Cj%7D%3D1) ，因此 ![[公式]](https://www.zhihu.com/equation?tex=M%5Bi%5D) 可以理解为模型在当前step上，对于batch样本的**注意力权重分配**，值得注意的是，对于不同的样本，Attentive transformer层输出的注意力权重也不同，这个特点在论文中被叫做**instance-wise**。

接着是正则项，为了增强模型对feature稀疏选择的能力，这里额外引入一个正则项，具体形式如下：![[公式]](https://www.zhihu.com/equation?tex=L_%7B%5Ctext+%7Bsparse%7D%7D%3D%5Csum_%7Bi%3D1%7D%5E%7BN_%7Bs+t+e+p+s%7D%7D+%5Csum_%7Bb%3D1%7D%5E%7BB%7D+%5Csum_%7Bj%3D1%7D%5E%7BD%7D+%5Cfrac%7B-M_%7Bb%2C+j%7D%5B%5Cmathrm%7Bi%7D%5D%7D%7BN_%7Bs+t+e+p+s%7D+%5Ccdot+B%7D+%5Clog+%5Cleft%28M_%7Bb%2C+j%7D%5Bi%5D%2B%5Cepsilon%5Cright%29%5C%5C)

这个正则项也很好理解，就是算了一个平均的熵，其目的是希望 ![[公式]](https://www.zhihu.com/equation?tex=M%5Bi%5D) 的分布尽量趋近于0或1，而考虑到 ![[公式]](https://www.zhihu.com/equation?tex=%5Csum_%7Bj-1%7D%5E%7BD%7DM%5Bi%5D_%7Bb%2Cj%7D%3D1)，则 ![[公式]](https://www.zhihu.com/equation?tex=L_%7Bsparse%7D) 反映的就是![[公式]](https://www.zhihu.com/equation?tex=M%5Bi%5D)的**稀疏程度**，![[公式]](https://www.zhihu.com/equation?tex=L_%7Bsparse%7D)越小，![[公式]](https://www.zhihu.com/equation?tex=M%5Bi%5D)越稀疏。

最后再看一下之前忽略掉的Feature attribute输出，它其实刻画的是feature的**全局重要性**。模型先对模型一个step的输出向量求和，得到一个标量，这个标量反映的是这个step对于最终结果的重要性，那么它乘以这个step的Mask矩阵就反映了这个step中每个feature的重要性，将所有step的结果加起来，就得到了feature的全局重要性。

总结一下，TabNet采用了**顺序多步**（sequential multi-step）框架，构造了一种类似于加性模型的神经网络，模型中比较关键的是Attentive transformer层和Feature transformer层，它们分别实现了下面两种功能：

- 特征选择：Attentive transformer层可以根据上一个step的结果得到当前step的Mask矩阵，并尽量使得Mask矩阵是**稀疏且不重复**的。值得注意的一点是，不同样本的Mask向量可以不同，也就是说TabNet可以让不同的样本选择不同的特征（instance-wise），而这个特点是树模型所不具备的，对于XGBoost这类加性模型，一个step就是一棵树，而这棵决策树用到的特征是在所有样本上挑选出来的（例如通过计算信息增益），它没有办法做到instance-wise。
- 特征计算：Feature transformer层实现了对于当前step步所选取特征的计算处理。还是类比于决策树，对于给定的一些特征，一棵决策树构造的是**单个特征的大小关系的组合**，也就是上面提到的决策流形，而之前那个简单神经网络就是通过一个FC层来模仿这个决策流形，但FC层只是构造了一组简单的线性关系，并没有考虑更加复杂的情况，因此TabNet通过更复杂的Feature transformer层来进行特征计算，个人感觉它的决策流形不一定和决策树的相似，在一些特征组合上它可能比决策树做得更好。

## 自监督学习

前面提到了DNN的一个好处就是可以进行表征学习，而TabNet就应用了自监督学习的方法，通过encoder-decoder框架来获得表格数据的representation，从而也有助于分类和回归任务，如下图所示：

![img](https://pic4.zhimg.com/80/v2-b8973da542872f102d427c5752ca0cc3_1440w.jpg)针对表格数据的自监督学习

简单来说，我们认为同一样本的不同特征之间是有关联的，因此自监督学习就是先人为mask掉一些feature，然后通过encoder-decoder模型来对mask掉的feature进行预测。我们认为通过这样的方式训练出来的encoder模型，可以有效地将表征样本的feature（可以理解为对数据进行了编码或压缩），这时再将encoder模型由于回归或分类任务，就能够事半功倍。自监督学习时的encoder模型就是上图中的模型，decoder模型如下所示：

![img](https://pic4.zhimg.com/80/v2-ea261726fb779ac1e3660e9e8ca08d7b_1440w.jpg)

这里的encoded representation就是encoder中**没有经过FC层的加和向量**，将它作为decoder的输入，decoder同样利用了Feature transformer层，只不过这次的目的是将representation向量重构为feature，然后类似地经过若干个step的加和，得到最后的重构feature。

设在一开始对feature做mask的矩阵是 ![[公式]](https://www.zhihu.com/equation?tex=S%5Cin+%5C%7B0%2C1%5C%7D%5E%7BB%5Ctimes+D%7D) ，特征数据是 ![[公式]](https://www.zhihu.com/equation?tex=f) ，则encoder的输入是 ![[公式]](https://www.zhihu.com/equation?tex=%281-S%29%5Ccdot+f) ，若最后decoder的输出是 ![[公式]](https://www.zhihu.com/equation?tex=%5Chat%7Bf%7D) ，那么自监督学习就是减小真实值 ![[公式]](https://www.zhihu.com/equation?tex=S%5Ccdot+f) 与重构值 ![[公式]](https://www.zhihu.com/equation?tex=S%5Ccdot+%5Chat%7Bf%7D) 之间的差别，考虑到不同的feature的量级不一定相同，因此采用正则化后的MSE作为loss，形式如下：![[公式]](https://www.zhihu.com/equation?tex=%5Csum_%7Bb%3D1%7D%5E%7BB%7D+%5Csum_%7Bj%3D1%7D%5E%7BD%7D%5Cleft%7C%5Cleft%28%5Chat%7Bf%7D_%7Bb%2C+j%7D-f_%7Bb%2C+j%7D%5Cright%29+%5Ccdot+S_%7Bb%2C+j%7D+%2F+%5Csqrt%7B%5Csum_%7Bb%3D1%7D%5E%7BB%7D%5Cleft%28f_%7Bb%2C+j%7D-1+%2F+B+%5Csum_%7Bb%3D1%7D%5E%7BB%7D+f_%7Bb%2C+j%7D%5Cright%29%5E%7B2%7D%7D%5Cright%7C%5E%7B2%7D%5C%5C)

另外，为了让模型学到的是整个feature数据的表征方法，而不仅仅是某些feature，在自监督学习的训练过程中，每一轮都会对矩阵 ![[公式]](https://www.zhihu.com/equation?tex=S) **重新采样**，以此来保证encoder模型的整体表征能力。

总结一下，自监督学习可以让encoder模型学到feature数据的representation，通过representation再对任务进行分类和回归就会变得更加容易了。类似于NLP中的pre-training和fine-tuning过程，自监督学习可以**很好地利用到没有label的数据**，从而使得模型在有label的样本比较少时也能取得不错的效果，并且还能加快模型的收敛速度。

## 实验

为了证明TabNet确实具有上文中提到的种种优点，这篇文章在不同的数据集上进行了各种类型的实验，这里只介绍一部分，其它实验以及具体实验细节可以看论文原文，写得也很详细。

1. **Instance-wise feature selection**

第一个实验考察的是TabNet能够根据不同样本来选择相应特征的能力，用的是6个人工构建的数据集Syn1-6，它们的feature大多是无用的，只有一小部分**关键feature**是与label相关的。对于Syn1-3，这些关键feature对数据集上的所有样本都是一样的，例如对于Syn2数据集，![[公式]](https://www.zhihu.com/equation?tex=%5C%7BX_3%2CX_4%2CX_5%2CX_6+%5C%7D) 是关键feature，因此只需要全局的特征选择方法就可以得到最优解；而Syn4-6则更困难一些，样本的关键feature并不相同，它们取决于另外一个**指示feature**（indicator），例如对于Syn4数据集， ![[公式]](https://www.zhihu.com/equation?tex=X_%7B11%7D) 是指示feature， ![[公式]](https://www.zhihu.com/equation?tex=X_%7B11%7D) 的取值，决定了 ![[公式]](https://www.zhihu.com/equation?tex=%5C%7BX_%7B1%7D%2CX_%7B2%7D%5C%7D) 和 ![[公式]](https://www.zhihu.com/equation?tex=%5C%7BX_3%2CX_4%2CX_5%2CX_6+%5C%7D) 哪一组是关键feature，显然，对于这样的数据集，简单的全局特征选择并不是最优的。

下表展示的是TabNet与一些baseline模型的在测试集上的AUC均值+标准差，可以看出TabNet表现不错，在Syn4-6数据集上，相较于全局特征选择方法（Global）有所改善。

![img](https://pic3.zhimg.com/80/v2-8e1e2aaedc4edf7d6cf4ccc6b6b0e282_1440w.jpg)



2. **真实数据集**

**Forest Cover Type**：这个数据集是一个分类任务——根据cartographic变量来对森林覆盖类型进行分类，实验的baseline采用了如XGBoost等目前主流的树模型、可以自动构造高阶特征的AutoInt、以及AutoML Tables这种用了神经网络结构搜索 (Neural Architecture Search)的强力模型（node hours的数量反映了模型的复杂性），对比结果如下：

![img](https://pic3.zhimg.com/80/v2-45fe44e0f1d9dff54774e32a75ae3036_1440w.jpg)

**Higgs Boson**：这是一个物理领域的数据集，任务是将产生希格斯玻色子的信号与背景信号分辨开来，由于这个数据集很大，因此DNN比树模型的表现更好，下面是对比结果，其中Sparse evolutionary MLP应用了目前最好的evolutionary sparsification算法，能够有效减小原始MLP模型的大小，不过可以看出，和它大小相近的TabNet-S的性能也只是稍弱一点，这说明轻量级的TabNet表现依旧很好。

![img](https://pic3.zhimg.com/80/v2-c43478ebbbd042335253f3f4522d79de_1440w.jpg)



3. **可解释性**

为了展示TabNet的可解释性，就需要用到上文提到的Feature attribute。具体来说，我们用 ![[公式]](https://www.zhihu.com/equation?tex=%5Ceta_%7B%5Cmathrm%7Bb%7D%7D%5B%5Cmathrm%7Bi%7D%5D%3D%5Csum_%7Bc%3D1%7D%5E%7BN_%7Bd%7D%7D+%5Coperatorname%7BReLU%7D%5Cleft%28%5Cmathrm%7Bd%7D_%7B%5Cmathrm%7Bb%7D%2C+%5Cmathrm%7Bc%7D%7D%5B%5Cmathrm%7Bi%7D%5D%5Cright%29) 来表示第i个step对最后结果的贡献，那么根据之前的讨论可知，归一化后的feature全局重要性可以表示为：

![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbf%7BM%7D_%7B%5Cmathrm%7Bagg%7D-%5Cmathbf%7Bb%7D%2C+%5Cmathbf%7Bj%7D%7D%3D%5Csum_%7Bi%3D1%7D%5E%7BN_%7Bs+t+e+p+s%7D%7D+%5Ceta_%7B%5Cmathbf%7Bb%7D%7D%5B%5Cmathbf%7Bi%7D%5D+%5Cmathbf%7BM%7D_%7B%5Cmathbf%7Bb%7D%2C+%5Cmathbf%7Bj%7D%7D%5B%5Cmathbf%7Bi%7D%5D+%2F+%5Csum_%7Bj%3D1%7D%5E%7BD%7D+%5Csum_%7Bi%3D1%7D%5E%7BN_%7Bs+t+e+p+s%7D%7D+%5Ceta_%7B%5Cmathbf%7Bb%7D%7D%5B%5Cmathbf%7Bi%7D%5D+%5Cmathbf%7BM%7D_%7B%5Cmathbf%7Bb%7D%2C+%5Cmathbf%7Bj%7D%7D%5B%5Cmathbf%7Bi%7D%5D%5C%5C)

这样我们就可以展示TabNet中的feature重要性了，对于第一个实验中的人工数据集Syn1-6，可将feature重要性可视化，如下图所示：

![img](https://pic3.zhimg.com/80/v2-45c1abf55722ed1a95719d14d894ac1a_1440w.jpg)

其中 ![[公式]](https://www.zhihu.com/equation?tex=M_%7Bagg%7D) 反映的是feature的全局重要性， ![[公式]](https://www.zhihu.com/equation?tex=M%5Bi%5D) 是第i个step的feature重要性。可以看出，对于Syn2，TabNet在每一个step分别选择了一个feature，最后的全局重要性也集中在 ![[公式]](https://www.zhihu.com/equation?tex=%5C%7BX_3%2CX_4%2CX_5%2CX_6+%5C%7D) 上，这与Syn2的关键feature是全局一致的设定相符；对于Syn6，TabNet在一个step中选择的feature并不一致，而最后的全局重要性则集中在 ![[公式]](https://www.zhihu.com/equation?tex=%5C%7BX_1%2CX_2%5C%7D) 、![[公式]](https://www.zhihu.com/equation?tex=%5C%7BX_3%2CX_4%2CX_5%2CX_6+%5C%7D) 以及 ![[公式]](https://www.zhihu.com/equation?tex=X_%7B11%7D) 上，这也与Syn6的Instance-wise设定相符。这些可视化结果说明TabNet可以准确地捕捉到指示feature与关键feature之间的联系。

论文也在真实数据上做了实验，采用的数据集是Adult Census Income，任务是判断一个人的收入是否高于5万美元，下图是TabNet和其他一些可解释模型的feature重要性排序，表中数字代表重要性排名。

![img](https://pic3.zhimg.com/80/v2-411aeeaa4194dee5c556c0474249cf42_1440w.jpg)



4. **自监督学习**

前面已经提到了，自监督学习可以提高模型的小样本学习能力，还能加快模型的收敛速度。为了验证这一点，这里我们采用Higgs Boson数据集，其中用全部样本来做自监督学习（pre-training），而只用部分样本做监督学习（fine-tuning），该方法与直接全样本监督学习的对比结果如下所示：

![img](https://pic1.zhimg.com/80/v2-281de07af94e43151a5351efcf65a1b4_1440w.jpg)

![img](https://pic3.zhimg.com/80/v2-037dc35b07e7ea0fca19f06e29777e8e_1440w.jpg)

从结果中可以看出，通过自监督学习进行预训练之后，模型的收敛速度明显更快，小样本学习的结果也变得更好。

## 总结

这篇论文提出的TabNet是一种针对于表格数据的神经网络，它通过类似于加性模型的顺序注意力机制（sequential attention mechanism）实现了instance-wise的特征选择，还通过encoder-decoder框架实现了自监督学习，从而将树模型的可解释性与DNN的表征能力很好地结合到了一起，相信这种兼具两者优点的模型将会成为数据挖掘竞赛中的一大利器，也对未来的研究提供了一个很好的思路。



## 参考资料

[1] [TabNet: Attentive Interpretable Tabular Learning](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1908.07442)
