'''
Author: jianzhnie
Date: 2021-11-08 11:57:03
LastEditTime: 2021-11-08 14:18:14
LastEditors: jianzhnie
Description: Use multimodal model for imdb datasets.
            Here's a data set of 1,000 most popular movies on IMDB in the last 10 years. The data points included are:
            Title, Genre, Description, Director, Actors, Year, Runtime, Rating, Votes, Revenue, Metascrore
            Feel free to tinker with it and derive interesting insights.
'''

import os
import numpy as np
import pandas as pd

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pytorch_widedeep import Trainer
from pytorch_widedeep.preprocessing import WidePreprocessor, TabPreprocessor, TextPreprocessor
from pytorch_widedeep.models import Wide, TabMlp, WideDeep, DeepText
from pytorch_widedeep.metrics import Accuracy

if __name__ == "__main__":

    data_dir = '/media/robin/DATA/datatsets/structure_data/imdb'
    file_name = 'IMDB-Movie-Data.csv'
    file_path = os.path.join(data_dir, file_name)
    df = pd.read_csv(file_path)
    df["target"] = (df["Genre"].apply(lambda x: "Drama" in x)).astype(int)
    df.drop("Genre", axis=1, inplace=True)
    df_train, df_test = train_test_split(df, test_size=0.2, stratify=df.target)

    print(df_train)
    cont_cols = [
        "Runtime (Minutes)", "Rating", "Votes", "Revenue (Millions)",
        "Metascore"
    ]
    target_col = "target"
    # target
    target = df_train[target_col].values

    # deeptabular
    tab_preprocessor = TabPreprocessor(continuous_cols=cont_cols)
    X_tab = tab_preprocessor.fit_transform(df_train)
    deeptabular = TabMlp(
        mlp_hidden_dims=[64, 32],
        column_idx=tab_preprocessor.column_idx,
        continuous_cols=cont_cols,
    )

    text_preprocessor = TextPreprocessor(
        text_col='Description', max_vocab=256, min_freq=1, maxlen=80)
    X_text = text_preprocessor.fit_transform(df_train)
    print(X_text)
    deeptext = DeepText(
        vocab_size=256, hidden_dim=64, n_layers=3, padding_idx=0, embed_dim=4)
    # wide and deep
    model = WideDeep(deeptabular=deeptabular, deeptext=deeptext)
    print(model)
    # train the model
    trainer = Trainer(model, objective="binary", metrics=[Accuracy])
    trainer.fit(
        X_tab=X_tab,
        X_text=X_text,
        target=target,
        n_epochs=5,
        batch_size=64,
        val_split=0.1,
    )
    # predict
    X_text_te = text_preprocessor.transform(df_test)
    X_tab_te = tab_preprocessor.transform(df_test)
    preds = trainer.predict(X_tab=X_tab_te, X_text=X_text_te)
    print(preds)