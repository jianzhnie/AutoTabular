import sys

import flash
import pandas as pd
import torch
from autotabular.algorithms.deepnets.tabnet import TabularClassifier
from flash.core.data.utils import download_data
from flash.tabular import TabularClassificationData

sys.path.append('../')

# 1. load data
datamodule = TabularClassificationData.from_csv(
    ['Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'Embarked'],
    'Fare',
    target_fields='Survived',
    train_file='../autotabular/datasets/data/Titanic.csv',
    batch_size=32,
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
