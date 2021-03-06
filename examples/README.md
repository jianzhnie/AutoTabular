
## Benchmark Datasets

| Datasets | Category | Link |
| -------- | -------- | ---- |
| Adult | binary Classification | https://archive.ics.uci.edu/ml/datasets/Adult |
| Bank-market | binary Classification | https://archive.ics.uci.edu/ml/datasets/Bank+Marketing |
| Amazon.com - Employee Access Challenge | binary Classification | https://www.kaggle.com/c/amazon-employee-access-challenge/data |
| Credit | binary Classification | https://www.kaggle.com/c/GiveMeSomeCredit/data |
| shelter-animal-outcomes | multiclass | https://www.kaggle.com/c/shelter-animal-outcomes/data |
| house-prices-advanced-regression-techniques | Regession | https://www.kaggle.com/c/house-prices-advanced-regression-techniques. |
| Bike Sharing Demand | Regression | https://www.kaggle.com/c/bike-sharing-demand/data |
| Prudential Life Insurance Assessment | Classification | https://www.kaggle.com/c/prudential-life-insurance-assessment/data |
| Display Advertising Challenge | Classification | https://www.kaggle.com/c/criteo-display-ad-challenge/data |
| nyc-taxi-trip-duration| regression|  kaggle|
| fb_comments | regression|  https://archive.ics.uci.edu/ml/machine-learning-databases/00363/Dataset.zip|
|covtype|Classification|https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz|
|UCI_Credit_Card|Classification|UCI|



# **Adult**

## LightGBM

```sh
{
    "runtime": 0.25020575523376465,
    "acc": 0.9009516410953583,
    "auc": 0.6169500687269301,
    "f1": 0.35929648241206025
}
```

##

```sh
Evaluation: accuracy on test data: 0.5867998689813299
Evaluations on test data:
{
    "accuracy": 0.5867998689813299
}

```sh

                  model  score_test  score_val  pred_time_test  pred_time_val   fit_time  pred_time_test_marginal  pred_time_val_marginal  fit_time_marginal  stack_level  can_infer  fit_order
0   WeightedEnsemble_L2    0.586800   0.577396        0.324756       0.118791  51.265472                 0.002953                0.000990           0.584490            2       True         13
1              LightGBM    0.586636   0.575102        0.092299       0.021177   0.776972                 0.092299                0.021177           0.776972            1       True          5
2         LightGBMLarge    0.585653   0.570516        0.044704       0.026019   1.422465                 0.044704                0.026019           1.422465            1       True         12
3               XGBoost    0.583524   0.574939        0.141645       0.017584   0.952234                 0.141645                0.017584           0.952234            1       True         11
4              CatBoost    0.581559   0.571499        0.012733       0.013407   9.210831                 0.012733                0.013407           9.210831            1       True          8
5            LightGBMXT    0.579921   0.569697        0.041396       0.035489   1.185030                 0.041396                0.035489           1.185030            1       True          4
6       NeuralNetFastAI    0.577301   0.567240        0.075126       0.065633  39.740945                 0.075126                0.065633          39.740945            1       True          3
7      RandomForestEntr    0.543727   0.535627        0.725883       0.118161   1.602522                 0.725883                0.118161           1.602522            1       True          7
8      RandomForestGini    0.538487   0.531368        0.884620       0.118408   1.365430                 0.884620                0.118408           1.365430            1       True          6
9        ExtraTreesGini    0.533246   0.527273        0.491555       0.108277   0.825006                 0.491555                0.108277           0.825006            1       True          9
10       ExtraTreesEntr    0.531444   0.522031        0.631304       0.118686   0.907277                 0.631304                0.118686           0.907277            1       True         10
11       KNeighborsUnif    0.472486   0.471744        0.132595       0.108453   0.217091                 0.132595                0.108453           0.217091            1       True          1
12       KNeighborsDist    0.443662   0.439640        0.159426       0.107895   0.216901                 0.159426                0.107895           0.216901            1       True          2
```



# **BANK MARKETING**

##  LightGBM

Accuracy: 0.9009516410953583. F1: 0.35929648241206025. ROC_AUC: 0.6169500687269301

## Autogluon

Evaluation: roc_auc on test data: 0.7932188141901805
Evaluations on test data:
```sh
{
    "roc_auc": 0.7932188141901805,
    "accuracy": 0.899980578753156,
    "balanced_accuracy": 0.6055423635886269,
    "mcc": 0.3350921434155827,
    "f1": 0.33376455368693403,
    "precision": 0.6201923076923077,
    "recall": 0.22831858407079647
}
```

```sh
                  model  score_test  score_val  pred_time_test  pred_time_val   fit_time  pred_time_test_marginal  pred_time_val_marginal  fit_time_marginal  stack_level  can_infer  fit_order
0              LightGBM    0.794310   0.800945        0.060899       0.011755   0.376998                 0.060899                0.011755           0.376998            1       True          4
1   WeightedEnsemble_L2    0.793219   0.805910        0.310814       0.213801  37.933798                 0.011362                0.001433           1.795885            2       True         14
2               XGBoost    0.792844   0.798676        0.071560       0.014120   0.329626                 0.071560                0.014120           0.329626            1       True         11
3         LightGBMLarge    0.790024   0.804079        0.023960       0.015402   0.541690                 0.023960                0.015402           0.541690            1       True         13
4            LightGBMXT    0.789667   0.791630        0.017457       0.013695   0.372125                 0.017457                0.013695           0.372125            1       True          3
5      RandomForestEntr    0.786927   0.782968        0.222352       0.110278   1.097945                 0.222352                0.110278           1.097945            1       True          6
6              CatBoost    0.784190   0.792967        0.010077       0.010843   2.254603                 0.010077                0.010843           2.254603            1       True          7
7       NeuralNetFastAI    0.784134   0.791027        0.078635       0.063625  32.907988                 0.078635                0.063625          32.907988            1       True         10
8      RandomForestGini    0.783973   0.778614        0.218709       0.107462   0.889401                 0.218709                0.107462           0.889401            1       True          5
9        NeuralNetMXNet    0.779303   0.790351        0.376103       0.373975  41.118439                 0.376103                0.373975          41.118439            1       True         12
10       ExtraTreesGini    0.778897   0.773716        0.245392       0.107401   0.689204                 0.245392                0.107401           0.689204            1       True          8
11       ExtraTreesEntr    0.778871   0.779230        0.222300       0.121855   0.691060                 0.222300                0.121855           0.691060            1       True          9
12       KNeighborsUnif    0.732413   0.745891        0.125880       0.110743   0.056635                 0.125880                0.110743           0.056635            1       True          1
13       KNeighborsDist    0.732380   0.743742        0.120806       0.110945   0.079304                 0.120806                0.110945           0.079304            1       True          2
```

## Autogluon + gbdt_embedding
Evaluation: accuracy on test data: 0.8986210914740727
Evaluations on test data:
```sh
{
    "accuracy": 0.8986210914740727,
    "balanced_accuracy": 0.5977959119059754,
    "mcc": 0.3186939313630284,
    "roc_auc": 0.7965227262197099,
    "f1": 0.31496062992125984,
    "precision": 0.6091370558375635,
    "recall": 0.21238938053097345
}
```
```sh
                  model  score_test  score_val  pred_time_test  pred_time_val   fit_time  pred_time_test_marginal  pred_time_val_marginal  fit_time_marginal  stack_level  can_infer  fit_order
0         LightGBMLarge    0.901146   0.902292        0.004069       0.018548   2.035835                 0.004069                0.018548           2.035835            1       True         13
1               XGBoost    0.900563   0.902486        0.034835       0.012483   0.140268                 0.034835                0.012483           0.140268            1       True         11
2              LightGBM    0.900175   0.902292        0.003725       0.011452   0.204145                 0.003725                0.011452           0.204145            1       True          4
3              CatBoost    0.899786   0.901709        0.002861       0.007307   0.507490                 0.002861                0.007307           0.507490            1       True          7
4   WeightedEnsemble_L2    0.898621   0.903458        4.082587       3.266741  54.753304                 0.007943                0.005700           1.333156            2       True         14
5       NeuralNetFastAI    0.897844   0.901709        0.067458       0.056914  27.811110                 0.067458                0.056914          27.811110            1       True         10
6            LightGBMXT    0.897650   0.901709        0.003487       0.010666   0.218025                 0.003487                0.010666           0.218025            1       True          3
7        NeuralNetMXNet    0.894931   0.900350        0.157967       0.145453  20.694171                 0.157967                0.145453          20.694171            1       True         12
8      RandomForestEntr    0.893183   0.896853        0.266650       0.126920   0.905921                 0.266650                0.126920           0.905921            1       True          6
9        ExtraTreesEntr    0.892795   0.894716        0.308957       0.123732   0.688299                 0.308957                0.123732           0.688299            1       True          9
10       ExtraTreesGini    0.892601   0.895105        0.341269       0.122086   0.686160                 0.341269                0.122086           0.686160            1       True          8
11     RandomForestGini    0.892406   0.896076        0.273823       0.125934   0.890113                 0.273823                0.125934           0.890113            1       True          5
12       KNeighborsUnif    0.888134   0.885198        3.259767       2.745364   0.013068                 3.259767                2.745364           0.013068            1       True          1
13       KNeighborsDist    0.882113   0.880148        4.198992       2.978011   0.012753                 4.198992                2.978011           0.012753            1       True          2
```

## Autogluon + network_embedding

Evaluations on test data:
```sh
{
    "roc_auc": 0.7935784722543977,
    "accuracy": 0.899980578753156,
    "balanced_accuracy": 0.607870005714374,
    "mcc": 0.3378522072655624,
    "f1": 0.3388960205391528,
    "precision": 0.616822429906542,
    "recall": 0.2336283185840708
}
```

```sh
                  model  score_test  score_val  pred_time_test  pred_time_val   fit_time  pred_time_test_marginal  pred_time_val_marginal  fit_time_marginal  stack_level  can_infer  fit_order
0              LightGBM    0.794228   0.799175        0.010258       0.010874   0.242152                 0.010258                0.010874           0.242152            1       True          4
1   WeightedEnsemble_L2    0.793578   0.800334        0.090156       0.041656   5.314435                 0.001894                0.001483           1.764923            2       True         14
2               XGBoost    0.790306   0.799265        0.064113       0.014503   0.344668                 0.064113                0.014503           0.344668            1       True         11
3            LightGBMXT    0.789666   0.790635        0.010910       0.012148   0.334874                 0.010910                0.012148           0.334874            1       True          3
4              CatBoost    0.786962   0.792819        0.011479       0.012089   4.406275                 0.011479                0.012089           4.406275            1       True          7
5         LightGBMLarge    0.786271   0.796902        0.013891       0.014795   2.962692                 0.013891                0.014795           2.962692            1       True         13
6       NeuralNetFastAI    0.781407   0.784038        0.136436       0.061103  29.894907                 0.136436                0.061103          29.894907            1       True         10
7        NeuralNetMXNet    0.781022   0.789550        0.356769       0.396969  50.516046                 0.356769                0.396969          50.516046            1       True         12
8      RandomForestEntr    0.777776   0.766655        0.170549       0.126777   0.906674                 0.170549                0.126777           0.906674            1       True          6
9        ExtraTreesGini    0.777572   0.764737        0.216180       0.121191   0.689899                 0.216180                0.121191           0.689899            1       True          8
10     RandomForestGini    0.777461   0.767026        0.224467       0.124629   0.792258                 0.224467                0.124629           0.792258            1       True          5
11       ExtraTreesEntr    0.774681   0.767422        0.190128       0.122386   0.694366                 0.190128                0.122386           0.694366            1       True          9
12       KNeighborsUnif    0.709145   0.726308        0.104594       0.211192   0.658563                 0.104594                0.211192           0.658563            1       True          1
13       KNeighborsDist    0.690714   0.703982        0.213173       0.212145   0.643552                 0.213173                0.212145           0.643552            1       True          2

```
