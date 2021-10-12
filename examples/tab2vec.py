import string
from random import choices

import numpy as np
import pandas as pd
from pytorch_widedeep import Tab2Vec
from pytorch_widedeep.models import TabMlp, WideDeep
from pytorch_widedeep.preprocessing import TabPreprocessor

if __name__ == '__main__':
    colnames = list(string.ascii_lowercase)[:4]
    cat_col1_vals = ['a', 'b', 'c']
    cat_col2_vals = ['d', 'e', 'f']
    # Create the toy input dataframe and a toy dataframe to be vectorised
    cat_inp = [
        np.array(choices(c, k=5)) for c in [cat_col1_vals, cat_col2_vals]
    ]
    cont_inp = [np.round(np.random.rand(5), 2) for _ in range(2)]
    df_inp = pd.DataFrame(
        np.vstack(cat_inp + cont_inp).transpose(), columns=colnames)
    cat_t2v = [
        np.array(choices(c, k=5)) for c in [cat_col1_vals, cat_col2_vals]
    ]
    cont_t2v = [np.round(np.random.rand(5), 2) for _ in range(2)]
    df_t2v = pd.DataFrame(
        np.vstack(cat_t2v + cont_t2v).transpose(), columns=colnames)
    # fit the TabPreprocessor
    embed_cols = [('a', 2), ('b', 4)]
    cont_cols = ['c', 'd']
    tab_preprocessor = TabPreprocessor(
        embed_cols=embed_cols, continuous_cols=cont_cols)
    X_tab = tab_preprocessor.fit_transform(df_inp)
    # define the model (and let's assume we train it)
    tabmlp = TabMlp(
        column_idx=tab_preprocessor.column_idx,
        embed_input=tab_preprocessor.embeddings_input,
        continuous_cols=tab_preprocessor.continuous_cols,
        mlp_hidden_dims=[8, 4])
    model = WideDeep(deeptabular=tabmlp)
    # ...train the model...
    # vectorise the dataframe
    t2v = Tab2Vec(model, tab_preprocessor)
    X_vec = t2v.transform(df_t2v)
    print(X_vec)
