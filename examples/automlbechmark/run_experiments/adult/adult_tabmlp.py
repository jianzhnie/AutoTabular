import os
import pickle
from datetime import datetime
from pathlib import Path
from time import time

import pandas as pd
import torch
from pytorch_widedeep import Tab2Vec, Trainer
from pytorch_widedeep.callbacks import EarlyStopping, LRHistory, ModelCheckpoint
from pytorch_widedeep.metrics import Accuracy
from pytorch_widedeep.models import FTTransformer, TabMlp, WideDeep
from pytorch_widedeep.preprocessing import TabPreprocessor
from sklearn.model_selection import train_test_split
from tabmlp_parser import parse_args
from utils import set_lr_scheduler, set_optimizer

SEED = 42

if __name__ == '__main__':
    pd.options.display.max_columns = 100
    use_cuda = torch.cuda.is_available()
    ROOTDIR = Path('/home/robin/jianzh/autotabular/examples/automlbechmark')
    PROCESSED_DATA_DIR = ROOTDIR / 'data/processed_data/adult/'
    RESULTS_DIR = ROOTDIR / 'results/adult/tabmlp'
    if not RESULTS_DIR.is_dir():
        os.makedirs(RESULTS_DIR)

    adult_data = pd.read_csv(PROCESSED_DATA_DIR / 'adult.csv')
    target_name = 'target'

    # 200 is rather arbitraty but one has to make a decision as to how to decide
    # if something will be represented as embeddings or continuous in a "kind-of"
    # automated way
    cat_col_names, cont_col_names = [], []
    for col in adult_data.columns:
        # 50 is just a random number I choose here for this example
        if adult_data[col].dtype == 'O' or adult_data[col].nunique(
        ) < 50 and col != 'target':
            cat_col_names.append(col)
        elif col != 'target':
            cont_col_names.append(col)

    IndList = range(adult_data.shape[0])
    train_list, test_list = train_test_split(IndList, random_state=SEED)
    val_list, test_list = train_test_split(
        test_list, random_state=SEED, test_size=0.5)

    train = adult_data.iloc[train_list]
    val = adult_data.iloc[val_list]
    test = adult_data.iloc[test_list]

    # all columns will be represented by embeddings
    tab_preprocessor = TabPreprocessor(
        embed_cols=cat_col_names,
        continuous_cols=cont_col_names,
        for_transformer=False)

    X_train = tab_preprocessor.fit_transform(train)
    y_train = train.target.values
    X_valid = tab_preprocessor.transform(val)
    y_valid = val.target.values

    args = parse_args()
    if args.mlp_hidden_dims == 'auto':
        n_inp_dim = sum([e[2] for e in tab_preprocessor.embeddings_input])
        mlp_hidden_dims = [4 * n_inp_dim, 2 * n_inp_dim]
    else:
        mlp_hidden_dims = eval(args.mlp_hidden_dims)

    print(tab_preprocessor.embeddings_input)
    print(tab_preprocessor.continuous_cols)

    ft_transformer = FTTransformer(
        column_idx=tab_preprocessor.column_idx,
        embed_input=tab_preprocessor.embeddings_input,
        continuous_cols=tab_preprocessor.continuous_cols,
        n_blocks=3,
        n_heads=6,
        input_dim=36)

    deeptabular = TabMlp(
        column_idx=tab_preprocessor.column_idx,
        mlp_hidden_dims=mlp_hidden_dims,
        mlp_activation=args.mlp_activation,
        mlp_dropout=args.mlp_dropout,
        mlp_batchnorm=args.mlp_batchnorm,
        mlp_batchnorm_last=args.mlp_batchnorm_last,
        mlp_linear_first=args.mlp_linear_first,
        embed_input=tab_preprocessor.embeddings_input,
        continuous_cols=tab_preprocessor.continuous_cols,
        embed_dropout=args.embed_dropout,
    )

    model = WideDeep(deeptabular=deeptabular)
    optimizers = set_optimizer(model, args)

    steps_per_epoch = (X_train.shape[0] // args.batch_size) + 1
    lr_schedulers = set_lr_scheduler(optimizers, steps_per_epoch, args)
    early_stopping = EarlyStopping(
        monitor=args.monitor,
        min_delta=args.early_stop_delta,
        patience=args.early_stop_patience,
    )
    model_checkpoint = ModelCheckpoint(
        filepath='models/adult_tabmlp_model',
        monitor=args.monitor,
        save_best_only=True,
        verbose=1,
        max_save=1,
    )

    trainer = Trainer(
        model,
        objective='binary',
        optimizers=optimizers,
        lr_schedulers=lr_schedulers,
        reducelronplateau_criterion=args.monitor.split('_')[-1],
        callbacks=[early_stopping,
                   LRHistory(n_epochs=args.n_epochs)],
        metrics=[Accuracy],
    )

    start = time()
    trainer.fit(
        X_train={
            'X_tab': X_train,
            'target': y_train
        },
        X_val={
            'X_tab': X_valid,
            'target': y_valid
        },
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        validation_freq=args.eval_every,
    )
    runtime = time() - start

    if args.save_results:
        suffix = str(datetime.now()).replace(' ', '_').split('.')[:-1][0]
        filename = '_'.join(['adult_tabmlp', suffix]) + '.pkl'
        results_d = {}
        results_d['args'] = args.__dict__
        results_d['early_stopping'] = early_stopping
        results_d['trainer_history'] = trainer.history
        results_d['runtime'] = runtime
        with open(RESULTS_DIR / filename, 'wb') as f:
            pickle.dump(results_d, f)

    t2v = Tab2Vec(model, tab_preprocessor)
    X_vec = t2v.transform(X_train)
    # assuming is a test set with target col
    X_vec, y = t2v.transform(train.sample(100), target_col=target_name)
    print(X_vec)
