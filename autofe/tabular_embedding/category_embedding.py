# Necessary imports:
import warnings

import numpy as np
import pandas as pd
from keras import backend as K
from keras.layers import Activation, Dense, Embedding, Flatten, Input, concatenate
from keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import column_or_1d
from sklearn.utils.validation import check_is_fitted

warnings.filterwarnings('ignore')


# Helper functions:
class __LabelEncoder__(LabelEncoder):

    def transform(self, y):

        check_is_fitted(self, 'classes_')
        y = column_or_1d(y, warn=True)

        unseen = len(self.classes_)

        e = np.array([
            np.searchsorted(self.classes_, x) if x in self.classes_ else unseen
            for x in y
        ])

        if unseen in e:
            self.classes_ = np.array(self.classes_.tolist() + ['unseen'])

        return e


def get_embedding_info(data, categorical_variables=None):
    """this function identifies categorical variables and its embedding size.

    :data: input data [dataframe]
    :categorical_variables: list of categorical_variables [default: None]
    if None, it automatically takes the variables with data type 'object'

    embedding size of categorical variables are determined by minimum of 50 or half of the no. of its unique values.
    i.e. embedding size of a column  = Min(50, # unique values of that column)
    """
    if categorical_variables is None:
        categorical_variables = data.select_dtypes(include='object').columns

    return {
        col: (data[col].nunique(), min(50, (data[col].nunique() + 1) // 2))
        for col in categorical_variables
    }


def get_label_encoded_data(data, categorical_variables=None):
    """this function label encodes all the categorical variables using
    sklearn.preprocessing.labelencoder and returns a label encoded dataframe
    for training.

    :data: input data [dataframe]
    :categorical_variables: list of categorical_variables [Default: None]
    if None, it automatically takes the variables with data type 'object'
    """
    encoders = {}

    df = data.copy()

    if categorical_variables is None:
        categorical_variables = [
            col for col in df.columns if df[col].dtype == 'object'
        ]

    for var in categorical_variables:
        encoders[var] = __LabelEncoder__()
        df.loc[:, var] = encoders[var].fit_transform(df[var])

    return df, encoders


def r2(y_true, y_pred):
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return (1 - SS_res / (SS_tot + K.epsilon()))


def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


# Main function:


def get_embeddings(X_train,
                   y_train,
                   categorical_embedding_info,
                   is_classification,
                   epochs=100,
                   batch_size=256):
    """this function trains a shallow neural networks and returns embeddings of
    categorical variables.

    :X_train: training data [dataframe]
    :y_train: target variable
    :categorical_embedding_info: output of get_embedding_info function [dictionary of categorical variable and it's embedding size]
    :is_classification: True for classification tasks; False for regression tasks
    :epochs: num of epochs to train [default:100]
    :batch_size: batch size to train [default:256]

    It is a 2 layer neural network architecture with 1000 and 500 neurons with 'ReLU' activation
    for classification: loss = 'binary_crossentropy'; metrics = 'accuracy'
    for regression: loss = 'mean_squared_error'; metrics = 'r2'
    """

    numerical_variables = [
        x for x in X_train.columns
        if x not in list(categorical_embedding_info.keys())
    ]

    inputs = []
    flatten_layers = []

    for var, sz in categorical_embedding_info.items():
        input_c = Input(shape=(1, ), dtype='int32')
        embed_c = Embedding(*sz, input_length=1)(input_c)
        flatten_c = Flatten()(embed_c)
        inputs.append(input_c)
        flatten_layers.append(flatten_c)

    print(flatten_layers)
    input_num = Input(shape=(len(numerical_variables), ), dtype='float32')
    flatten_layers.append(input_num)
    inputs.append(input_num)

    flatten = concatenate(flatten_layers, axis=-1)
    print(flatten)

    fc1 = Dense(100, kernel_initializer='normal')(flatten)
    fc1 = Activation('relu')(fc1)

    fc2 = Dense(50, kernel_initializer='normal')(fc1)
    fc2 = Activation('relu')(fc2)

    if is_classification:
        output = Dense(1, activation='sigmoid')(fc2)

    else:
        output = Dense(1, kernel_initializer='normal')(fc2)

    nnet = Model(inputs=inputs, outputs=output)

    x_inputs = []
    for col in categorical_embedding_info.keys():
        x_inputs.append(X_train[col].values)

    x_inputs.append(X_train[numerical_variables].values)

    if is_classification:
        loss = 'binary_crossentropy'
        metrics = 'accuracy'
    else:
        loss = 'mean_squared_error'
        metrics = r2

    nnet.compile(loss=loss, optimizer='adam', metrics=[metrics])
    nnet.fit(
        x_inputs,
        y_train.values,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.2,
        verbose=0)

    embs = list(
        map(lambda x: x.get_weights()[0],
            [x for x in nnet.layers if 'Embedding' in str(x)]))
    embeddings = {
        var: emb
        for var, emb in zip(categorical_embedding_info.keys(), embs)
    }
    return embeddings


def get_embeddings_in_dataframe(embeddings, encoders):
    """this function return the embeddings in pandas dataframe.

    :embeddings: output of 'get_embeddings' function
    :encoders: output of 'get_embedding_info' function
    """

    assert len(embeddings) == len(
        encoders
    ), 'Categorical variables in embeddings does not match with those of encoders'

    dfs = {}
    for cat_var in embeddings.keys():
        df = pd.DataFrame(embeddings[cat_var])
        df.index = encoders[cat_var].classes_
        df.columns = [cat_var + '_embedding_' + str(num) for num in df.columns]
        dfs[cat_var] = df

    return dfs


def fit_transform(data, embeddings, encoders, drop_categorical_vars=False):
    """this function includes the trained embeddings into your data.

    :data: input data [dataframe]
    :embeddings: output of 'get_embeddings' function
    :encoders: output of 'get_embedding_info' function
    :drop_categorical_vars: False to keep the categorical variables in the data along with the embeddings
    if True - drops the categorical variables and replaces them with trained embeddings
    """

    assert len(embeddings) == len(
        encoders
    ), 'Categorical variables in embeddings does not match with those of encoders'

    for cat_var in embeddings.keys():
        df = pd.DataFrame(embeddings[cat_var])
        df.index = encoders[cat_var].classes_
        df.columns = [cat_var + '_embedding_' + str(num) for num in df.columns]
        data = data.merge(df, how='left', left_on=cat_var, right_index=True)

    if drop_categorical_vars:
        return data.drop(list(embeddings.keys()), axis=1)
    else:
        return data


if __name__ == '__main__':
    df = pd.read_csv('autotabular/datasets/data/Titanic.csv')
    X = df.drop(['Survived'], axis=1)
    y = df['Survived']
    embedding_info = get_embedding_info(X)
    print(embedding_info)
    X_encoded, encoders = get_label_encoded_data(X)
    print(X_encoded)
    print(encoders)
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y)
    embeddings = get_embeddings(
        X_train,
        y_train,
        categorical_embedding_info=embedding_info,
        is_classification=True,
        epochs=1,
        batch_size=64)
    print(embeddings)
