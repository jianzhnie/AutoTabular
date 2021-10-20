import sys

import flash
import pandas as pd
import torch
from autotabular.algorithms.deepnets.tabnet import TabularClassifier
from flash.tabular import TabularClassificationData

sys.path.append('../')

sys.path.append('../')

# 1. load data
datamodule = TabularClassificationData.from_csv(
    ['Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'Embarked'],
    'Fare',
    target_fields='Survived',
    train_file='../autotabular/datasets/data/Titanic.csv',
    batch_size=256,
    val_split=0.1,
)

# 2. Build the task
model = TabularClassifier.from_data(datamodule)
# 3. Create the trainer and train the model
trainer = flash.Trainer(max_epochs=1, gpus=torch.cuda.device_count())
trainer.fit(model, datamodule=datamodule)

# 4. Generate predictions from a CSV
predictions = model.predict('../autotabular/datasets/data/Titanic.csv')
predictions = pd.core.frame.DataFrame(predictions)
predictions.rename(columns={0: 'Not_suvived', 1: 'Suvived'}, inplace=True)
print(predictions)
# 5. Save the model!
trainer.save_checkpoint('tabular_classification_model.pt')


def get_category_columns(df, target):
    cat_col_names = []
    for col in df.columns:
        if df[col].dtype in ['object'] and col != target:
            cat_col_names.append(col)
    return cat_col_names


def get_numerical_columns(df, target):
    num_col_names = []
    for col in df.columns:
        if df[col].dtype in ['float', 'int'] and col != target:
            num_col_names.append(col)
    return num_col_names


data_file = '/home/robin/jianzh/autotabular/examples/automlbechmark/data/processed_data/bank_marketing/bankm.csv'
bank_maket = pd.read_csv(data_file)
target_name = 'target'

categorical_fields = get_category_columns(bank_maket, target_name)
numerical_fields = get_numerical_columns(bank_maket, target_name)

# 1. load data
datamodule = TabularClassificationData.from_csv(
    categorical_fields=categorical_fields,
    numerical_fields=numerical_fields,
    target_fields=target_name,
    train_file=data_file,
    batch_size=256,
    val_split=0.1,
)

# 2. Build the task
model = TabularClassifier.from_data(datamodule)
# 3. Create the trainer and train the model
trainer = flash.Trainer(max_epochs=30, gpus=torch.cuda.device_count())
trainer.fit(model, datamodule=datamodule)

# 4. Generate predictions from a CSV
predictions = model.predict(data_file)
predictions = pd.core.frame.DataFrame(predictions)
predictions.rename(columns={0: 'Not_suvived', 1: 'Suvived'}, inplace=True)
print(predictions)
# 5. Save the model!
trainer.save_checkpoint('tabular_classification_model.pt')
