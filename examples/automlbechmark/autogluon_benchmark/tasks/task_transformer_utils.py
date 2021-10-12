import time

import openml
import pandas as pd
from openml.exceptions import OpenMLServerException
from pytorch_widedeep import Tab2Vec
from pytorch_widedeep.metrics import Accuracy
from pytorch_widedeep.models import FTTransformer, WideDeep
from pytorch_widedeep.preprocessing import TabPreprocessor
from pytorch_widedeep.training import Trainer
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

from ..frameworks.autogluon.run import run


def get_task(task_id: int):
    task = openml.tasks.get_task(task_id)
    return task


def get_dataset(task):
    X, y, _, _ = task.get_dataset().get_data(task.target_name)
    cat_col_names, cont_col_names = [], []
    for col in X.columns:
        # 50 is just a random number I choose here for this example
        if X[col].dtype == 'O':
            cat_col_names.append(col)
        else:
            cont_col_names.append(col)

    label_encoder = LabelEncoder()
    le_y = label_encoder.fit_transform(y)

    tab_preprocessor = TabPreprocessor(
        embed_cols=cat_col_names,
        continuous_cols=cont_col_names,
        for_transformer=True)

    X_tab = tab_preprocessor.fit_transform(X)

    ft_transformer = FTTransformer(
        column_idx=tab_preprocessor.column_idx,
        embed_input=tab_preprocessor.embeddings_input,
        continuous_cols=tab_preprocessor.continuous_cols,
        n_blocks=3,
        n_heads=6,
        input_dim=36)

    model = WideDeep(deeptabular=ft_transformer)
    trainer = Trainer(model, objective='binary', metrics=[Accuracy])
    trainer.fit(
        X_tab=X_tab, target=le_y, n_epochs=30, batch_size=256, val_split=0.2)
    t2v = Tab2Vec(model=model, tab_preprocessor=tab_preprocessor)
    # assuming is a test set with target col
    X_vec = t2v.transform(X)
    feature_names = ['nn_embed_' + str(i) for i in range(X_vec.shape[1])]
    X_vec = pd.DataFrame(X_vec, columns=feature_names)

    selector = SelectFromModel(
        estimator=LogisticRegression(), max_features=64).fit(X_vec, y)
    support = selector.get_support()
    col_names = X_vec.columns[support]
    X_enc = selector.transform(X_vec)
    X_enc = pd.DataFrame(X_enc, columns=col_names)

    return X_enc, y


def run_task(task,
             n_folds=None,
             n_repeats=1,
             n_samples=1,
             init_args=None,
             fit_args=None,
             print_leaderboard=True):
    if isinstance(task, int):
        task_id = task
        delay_exp = 0
        while True:
            try:
                print(f'Getting task {task_id}')
                task = openml.tasks.get_task(task_id)
                print(f'Got task {task_id}')
            except OpenMLServerException as e:
                delay = 2**delay_exp
                delay_exp += 1
                if delay_exp > 10:
                    raise ValueError('Unable to get task after 10 retries')
                print(e)
                print(f'Retry in {delay}s...')
                time.sleep(delay)
                continue
            break

    n_repeats_full, n_folds_full, n_samples_full = task.get_split_dimensions()
    if n_folds is None:
        n_folds = n_folds_full
    if n_repeats is None:
        n_repeats = n_repeats_full
    if n_samples is None:
        n_samples = n_samples_full

    X, y = get_dataset(task)
    predictors = []
    scores = []
    eval_dict = []
    if isinstance(n_folds, int):
        n_folds = list(range(n_folds))
    for repeat_idx in range(n_repeats):
        for fold_idx in n_folds:
            for sample_idx in range(n_samples):
                train_indices, test_indices = task.get_train_test_split_indices(
                    repeat=repeat_idx,
                    fold=fold_idx,
                    sample=sample_idx,
                )
                X_train = X.loc[train_indices]
                y_train = y[train_indices]
                X_test = X.loc[test_indices]
                y_test = y[test_indices]

                print('Repeat #{}, fold #{}, samples {}: X_train.shape: {}, '
                      'y_train.shape {}, X_test.shape {}, y_test.shape {}'.
                      format(
                          repeat_idx,
                          fold_idx,
                          sample_idx,
                          X_train.shape,
                          y_train.shape,
                          X_test.shape,
                          y_test.shape,
                      ))

                # Save and get_task_dict data to remove any pre-set dtypes, we want to observe performance from worst-case scenario: raw csv
                predictor = run(
                    X_train=X_train,
                    y_train=y_train,
                    label=task.target_name,
                    init_args=init_args,
                    fit_args=fit_args)
                predictors.append(predictor)
                X_test[task.target_name] = y_test
                eval_dict.append(predictor.evaluate(X_test))
                scores.append(
                    predictor.evaluate(X_test)[predictor.eval_metric.name])
                if print_leaderboard:
                    predictor.leaderboard(X_test)

    return predictors, scores, eval_dict
