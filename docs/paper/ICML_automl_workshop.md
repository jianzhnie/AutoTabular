## 拒稿论文

- Automatically Learning Feature Crossing from Model Interpretation for Tabular Data [link](https://openreview.net/forum?id=Sye2s2VtDr&noteId=BkxGEFUsir)

  > 作者在论文中缺少相关实验基线对比， 至少要对比 FM 模型

- Parsed Categoric Encodings with Automunge [link](https://openreview.net/forum?id=Dh29CAlnMW&noteId=C2cjMyyPqxd)

  > 本文介绍了一个用于预处理表格数据的python库“Automunge”。作者开发了一个有用的库，可供表格数据预处理。论文缺少相关实验证明该工具箱在实际使用过程中对模型性能的提升。

- SpreadsheetCoder: Formula Prediction from Semi-structured Context  [link](https://openreview.net/forum?id=refmbBH_ysO)

  > 这篇文章显然有很好的想法, 然而，论文缺乏可以验证的实验（只有一个专有数据集的实验）。而且，没有公开数据集上的基准实验， 我们担心论文中报告的准确率是否可信。

## 中稿论文

- Multimodal AutoML on Structured Tables with Text Fields [link](https://openreview.net/forum?id=OHAIVOOl7Vl)

**Abstract:** We  design  automated  supervised  learning  systems  for  data  tables  that  not  only  contain numeric/categorical columns,  but  text  fields  as  well.  Here  we  assemble  15  multimodal data tables that each contain some text fields and stem from a real business application. Over this benchmark, we evaluate numerous multimodal AutoML strategies, including a standard two-stage approach where NLP is used to featurize the text such that AutoML for tabular data can then be applied. We propose various practically superior strategies based on multimodal adaptations of Transformer networks and stack ensembling of these networks with classical tabular models.  Beyond performing the best in our benchmark, our proposed (fully automated) methodology manages to rank 1st place (against human data scientists) when fit to the raw tabular/text data in two MachineHack prediction competitions  and 2nd place (out of 2380 teams) in Kaggle’s Mercari Price Suggestion Challenge.

**Ethics Statement:** The paper proposed the first benchmark for multimodal AutoML on structured data tables with text fields. This can help researchers evaluate their multimodal AutoML solutions and will boost innovations in this area. Also, the network fusion strategies and model ensembling techniques discussed and compared in the paper can provide insights in how to design a good multimodal AutoML system. In addition, the practical AutoML system proposed in the paper can help people that are less familiar with state-of-the-art ML techniques solve real world problems via ML. This democratizes machine learning and improves fairness of the area.



**Review:** This paper proposes the use of modern NLP architecture to handle text fields in tabular data for AutoML.

Positives:

- The authors provide a new benchmark set of 15 datasets with a focus on text fields
- Solid ablation analysis of separate components of their approach
- Clearly and well written

Negatives (all rather minor):

- I am a bit unsure how much I dislike averaging over different performance measures (R2, AUC, ACC) and all datasets without any form of scaling in the benchmark.
- Only one other AutoML tool is used for comparison, maybe add a sentence why you selected H2O as a baseline.

Minor comments:

- In Sec 2. Methods: The figure reference for fuse-late in the text should be 1c) instead of 1d)

Overall I really enjoyed reading the paper. The addressed problem is of high practical relevance and their approach is very reasonable. The publishing of their benchmarks allows for easy and fair comparison of potential further work.



**Review:**The paper presents an AutoML system for tabular data with text fields. Furthermore, the paper introduces a benchmark for tabular data with text fields and evaluates the AutoML system successfully on this benchmark but also on two competitions and a Kaggle Challange. The benchmark and the code for the system are available.

The paper uses pre-trained Transformer models to build a multimodal model. The paper describes and evaluates multiple ways to embed text and combine text and tabular data. The methods are well-written and comprehensible.

The experiment is based on the introduced benchmark and contains a detailed comparison of AutoML strategies on tabular data with text fields.The paper lacks a conclusion and could analyze more the difference between pre-embedding and n-grams/word2ve
