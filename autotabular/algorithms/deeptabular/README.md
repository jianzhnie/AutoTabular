
# DeepTabular

DeepLearning model for Tabular data.


## Available Models

1. **TabMlp**: a simple MLP that receives embeddings representing the
categorical features, concatenated with the continuous features.
2. **TabResnet**: similar to the previous model but the embeddings are
passed through a series of ResNet blocks built with dense layers.
3. **TabNet**: details on TabNet can be found in
[TabNet: Attentive Interpretable Tabular Learning](https://arxiv.org/abs/1908.07442)
4. **Node** is a model presented in ICLR 2020 and according to the authors have beaten well-tuned Gradient Boosting models on many datasets. [Neural Oblivious Decision Ensembles for Deep Learning on Tabular Data](https://arxiv.org/abs/1909.06312).
5. **AutoInt** is a model which tries to learn interactions between the features in an automated way and create a better representation and then use this representation in downstream task. [AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks](https://arxiv.org/abs/1810.11921) 
6. **MDN** [Mixture Density Networks](https://publications.aston.ac.uk/id/eprint/373/1/NCRG_94_004.pdf) is a regression model which uses gaussian components to approximate the target function and  provide a probabilistic prediction out of the box.

And the ``Tabformer`` family, i.e. Transformers for Tabular data:

1. **TabTransformer**: details on the TabTransformer can be found in
[TabTransformer: Tabular Data Modeling Using Contextual Embeddings](https://arxiv.org/pdf/2012.06678.pdf).
2. **SAINT**: Details on SAINT can be found in
[SAINT: Improved Neural Networks for Tabular Data via Row Attention and Contrastive Pre-Training](https://arxiv.org/abs/2106.01342).
6. **FT-Transformer**: details on the FT-Transformer can be found in
[Revisiting Deep Learning Models for Tabular Data](https://arxiv.org/abs/2106.11959).
7. **TabFastFormer**: adaptation of the FastFormer for tabular data. Details
on the Fasformer can be found in
[FastFormers: Highly Efficient Transformer Models for Natural Language Understanding](https://arxiv.org/abs/2010.13382)
8. **TabPerceiver**: adaptation of the Perceiver for tabular data. Details on
the Perceiver can be found in
[Perceiver: General Perception with Iterative Attention](https://arxiv.org/abs/2103.03206)


### Reference

1. pytorch-fm
   - https://rixwew.github.io/pytorch-fm
2. pytorch_tabular
   - https://github.com/manujosephv/pytorch_tabular
3. pytorch-widedeep
   - https://github.com/jrzaurin/pytorch-widedeep
4. tabnet
   - https://github.com/dreamquark-ai/tabnet