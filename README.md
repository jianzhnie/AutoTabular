# AutoTabular   

[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539)
[![Conference](http://img.shields.io/badge/NeurIPS-2019-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)
[![Conference](http://img.shields.io/badge/ICLR-2019-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)
[![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)  


AutoTabular automates machine learning tasks enabling you to easily achieve strong predictive performance in your applications.  With just a few lines of code, you can train and deploy high-accuracy machine learning and deep learning models tabular data.

<p align="center">
  <img src="https://apulis-gitlab.apulis.cn/apulis/AutoTabular/autotabular/docs/autotablar.png" width="100%" />
</p>


## What's good in it? 

- It is using the RAPIDS as back-end support, gives you the ability to execute end-to-end data science and analytics pipelines entirely on GPUs. 
- It Supports many anomaly detection models: , 
- It using meta learning to accelerate  model selection and parameter tuning.
- It is using many Deep Learning models for tabular data: `Wide&Deep`,  `DCN(Deep & Cross Network)`, `FM`, `DeepFM`, `PNN` ...
- It is using many machine learning algorithms: `Baseline`, `Linear`, `Random Forest`, `Extra Trees`, `LightGBM`, `Xgboost`, `CatBoost`, and `Nearest Neighbors`.
- It can compute Ensemble based on greedy algorithm from [Caruana paper](http://www.cs.cornell.edu/~alexn/papers/shotgun.icml04.revised.rev2.pdf).
- It can stack models to build level 2 ensemble (available in `Compete` mode or after setting `stack_models` parameter).
- It can do features preprocessing, like: missing values imputation and converting categoricals. What is more, it can also handle target values preprocessing.
- It can do advanced features engineering, like: [Golden Features](https://supervised.mljar.com/features/golden_features/), [Features Selection](https://supervised.mljar.com/features/features_selection/), Text and Time Transformations.
- It can tune hyper-parameters with `not-so-random-search` algorithm (random-search over defined set of values) and hill climbing to fine-tune final models.


## Example

First, install dependencies   
```bash
# clone project   
git clone https://apulis-gitlab.apulis.cn/apulis/AutoTabular/autotabular.git

# install project   
cd autotabular
pip install -e .   
pip install -r requirements.txt
```
 Next, navigate to any file and run it.   
 ```bash
# module folder
cd example

# run module (example: mnist as your main contribution)   
python demo.py    
 ```

### Citation  
If you use AutoTabular in a scientific publication, please cite the following paper:

Robin, et al. ["AutoTabular: Robust and Accurate AutoML for Structured Data."](https://arxiv.org/abs/2003.06505) arXiv preprint arXiv:2003.06505 (2021).

BibTeX entry:

```bibtex
@article{agtabular,
  title={AutoTabular: Robust and Accurate AutoML for Structured Data},
  author={JianZheng, WenQi},
  journal={arXiv preprint arXiv:2003.06505},
  year={2021}
}
```

## License

This library is licensed under the Apache 2.0 License.

## Contributing to AutoTabular

We are actively accepting code contributions to the AutoTabular project. If you are interested in contributing to AutoTabular, please contact me.