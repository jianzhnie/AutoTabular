# Deep Learning For Tabular Data

- [1 Deep Learning For Tabular Data](http://ethen8181.github.io/machine-learning/deep_learning/tabular/tabular.html#Deep-Learning-For-Tabular-Data)
  - [1.1 Data Preprocessing](http://ethen8181.github.io/machine-learning/deep_learning/tabular/tabular.html#Data-Preprocessing)
  - [1.2 PyTorch Dataset](http://ethen8181.github.io/machine-learning/deep_learning/tabular/tabular.html#PyTorch-Dataset)
  - [1.3 PyTorch Lightning Module](http://ethen8181.github.io/machine-learning/deep_learning/tabular/tabular.html#PyTorch-Lightning-Module)
  - [1.4 Evaluation](http://ethen8181.github.io/machine-learning/deep_learning/tabular/tabular.html#Evaluation)
  - [1.5 ONNX Runtime](http://ethen8181.github.io/machine-learning/deep_learning/tabular/tabular.html#ONNX-Runtime)
  - [1.6 Ending Notes](http://ethen8181.github.io/machine-learning/deep_learning/tabular/tabular.html#Ending-Notes)
- [2 Reference](http://ethen8181.github.io/machine-learning/deep_learning/tabular/tabular.html#Reference)

In [1]:

```
# code for loading the format for the notebook
import os

# path : store the current path to convert back to it later
path = os.getcwd()
os.chdir(os.path.join('..', '..', 'notebook_format'))

from formats import load_style
load_style(css_style='custom2.css', plot_style=False)
```

Out[1]:

In [2]:

```
os.chdir(path)

# 1. magic for inline plot
# 2. magic to print version
# 3. magic so that the notebook will reload external python modules
# 4. magic to enable retina (high resolution) plots
# https://gist.github.com/minrk/3301035
%matplotlib inline
%load_ext watermark
%load_ext autoreload
%autoreload 2
%config InlineBackend.figure_format='retina'

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# change default style figure and font size
plt.rcParams['figure.figsize'] = 8, 6
plt.rcParams['font.size'] = 12

# prevent scientific notations
pd.set_option('display.float_format', lambda x: '%.3f' % x)

%watermark -a 'Ethen' -d -t -v -p numpy,pandas,sklearn,matplotlib,torch,onnxruntime,pytorch_lightning
Ethen 2020-10-26 20:04:27 

CPython 3.6.4
IPython 7.15.0

numpy 1.18.5
pandas 1.0.5
sklearn 0.23.1
matplotlib 3.1.0
torch 1.6.0
onnxruntime 1.5.1
pytorch_lightning 1.0.1
```

# DEEP LEARNING FOR TABULAR DATA

The success of deep learning is often times mentioned in domains such as computer vision and natural language processing, another use-case that is also powerful but receives far less attention is to use deep learning on tabular data. By tabular data, we are referring to data that we usually put in a dataframe or a relational database, which is one of the most commonly encountered type of data in the industry.

One key technique to make the most out of deep learning for tabular data is to use embeddings for our categorical variables. This approach allows for relationship between categories to be captured, e.g. Given a categorical feature with high cardinality (number of distinct categories is large), it often works best to embed the categories into a lower dimensional numeric space, the embeddings might be able to capture zip codes that are geographically near each other without us needing to explicitly tell it so. By converting our raw categories into embeddings, our goal/hope is that these embeddings can capture more rich/complex relationships that will ultimately improve the performance of our models.

Another interesting thing about embeddings is that once we train them, we can leverage them in other scenarios. e.g. use these learned embeddings as features for our tree-based models.

## Data Preprocessing

We'll be using the credit card default dataset from UCI, we can download this dataset from [Kaggle](https://www.kaggle.com/uciml/default-of-credit-card-clients-dataset) as well.

There are many ways to implement the data preprocessing step, the high-level idea is to implement the following workflow:

- perform train/validation/test split.
- encode categorical columns as distinct numerical ids.
- standardize numerical columns.
- save the preprocesed data.

Here, we chose to preprocess the data once and save the output, so we won't have to go through these preprocessing steps every time.

In [3]:

```python
input_path = 'UCI_Credit_Card.csv'
df = pd.read_csv(input_path)
print(df.shape)
df.head()
(30000, 25)
```

Out[3]:

|      |   ID |  LIMIT_BAL |  SEX | EDUCATION | MARRIAGE |  AGE | PAY_0 | PAY_2 | PAY_3 | PAY_4 |  ... | BILL_AMT4 | BILL_AMT5 | BILL_AMT6 | PAY_AMT1 |  PAY_AMT2 |  PAY_AMT3 | PAY_AMT4 | PAY_AMT5 | PAY_AMT6 | default.payment.next.month |
| ---: | ---: | ---------: | ---: | --------: | -------: | ---: | ----: | ----: | ----: | ----: | ---: | --------: | --------: | --------: | -------: | --------: | --------: | -------: | -------: | -------: | -------------------------: |
|    0 |    1 |  20000.000 |    2 |         2 |        1 |   24 |     2 |     2 |    -1 |    -1 |  ... |     0.000 |     0.000 |     0.000 |    0.000 |   689.000 |     0.000 |    0.000 |    0.000 |    0.000 |                          1 |
|    1 |    2 | 120000.000 |    2 |         2 |        2 |   26 |    -1 |     2 |     0 |     0 |  ... |  3272.000 |  3455.000 |  3261.000 |    0.000 |  1000.000 |  1000.000 | 1000.000 |    0.000 | 2000.000 |                          1 |
|    2 |    3 |  90000.000 |    2 |         2 |        2 |   34 |     0 |     0 |     0 |     0 |  ... | 14331.000 | 14948.000 | 15549.000 | 1518.000 |  1500.000 |  1000.000 | 1000.000 | 1000.000 | 5000.000 |                          0 |
|    3 |    4 |  50000.000 |    2 |         2 |        1 |   37 |     0 |     0 |     0 |     0 |  ... | 28314.000 | 28959.000 | 29547.000 | 2000.000 |  2019.000 |  1200.000 | 1100.000 | 1069.000 | 1000.000 |                          0 |
|    4 |    5 |  50000.000 |    1 |         2 |        1 |   57 |    -1 |     0 |    -1 |     0 |  ... | 20940.000 | 19146.000 | 19131.000 | 2000.000 | 36681.000 | 10000.000 | 9000.000 |  689.000 |  679.000 |                          0 |

5 rows × 25 columns

In [4]:

```
id_cols = ['ID']
cat_cols = ['EDUCATION', 'SEX', 'MARRIAGE']
num_cols = [
    'LIMIT_BAL', 'AGE',
    'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
    'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
    'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6'
]
label_col = 'default.payment.next.month'

print('number of categorical columns: ', len(cat_cols))
print('number of numerical columns: ', len(num_cols))
number of categorical columns:  3
number of numerical columns:  20
```

In [5]:

```
test_size = 0.1
val_size = 0.3
random_state = 1234

df_train, df_test = train_test_split(
    df,
    test_size=test_size,
    random_state=random_state,
    stratify=df[label_col])

df_train, df_val = train_test_split(
    df_train,
    test_size=val_size,
    random_state=random_state,
    stratify=df_train[label_col])

print('train shape: ', df_train.shape)
print('validation shape: ', df_val.shape)
print('test shape: ', df_test.shape)

df_train.head()
train shape:  (18900, 25)
validation shape:  (8100, 25)
test shape:  (3000, 25)
```

Out[5]:

|       |    ID |  LIMIT_BAL |  SEX | EDUCATION | MARRIAGE |  AGE | PAY_0 | PAY_2 | PAY_3 | PAY_4 |  ... | BILL_AMT4 | BILL_AMT5 | BILL_AMT6 | PAY_AMT1 | PAY_AMT2 | PAY_AMT3 | PAY_AMT4 | PAY_AMT5 | PAY_AMT6 | default.payment.next.month |
| ----: | ----: | ---------: | ---: | --------: | -------: | ---: | ----: | ----: | ----: | ----: | ---: | --------: | --------: | --------: | -------: | -------: | -------: | -------: | -------: | -------: | -------------------------: |
|  9256 |  9257 |  20000.000 |    2 |         3 |        1 |   23 |     1 |     2 |     2 |    -2 |  ... |     0.000 |     0.000 |     0.000 |  480.000 |    0.000 |    0.000 |    0.000 |    0.000 |    0.000 |                          1 |
| 23220 | 23221 | 150000.000 |    2 |         3 |        2 |   35 |    -1 |     2 |    -1 |     2 |  ... |  1143.000 |   163.000 |  2036.000 |    0.000 | 2264.000 |    0.000 |  163.000 | 2036.000 |    0.000 |                          0 |
| 11074 | 11075 | 260000.000 |    2 |         2 |        1 |   43 |     2 |     2 |     2 |     2 |  ... |  2500.000 |  2500.000 |  2500.000 |    0.000 |    0.000 |    0.000 |    0.000 |    0.000 |    0.000 |                          1 |
|  1583 |  1584 |  50000.000 |    2 |         1 |        2 |   70 |     2 |     2 |     0 |     0 |  ... | 17793.000 | 18224.000 | 18612.000 |    0.000 | 2200.000 |  700.000 |  700.000 |  674.000 |  608.000 |                          0 |
|  8623 |  8624 | 390000.000 |    2 |         2 |        1 |   45 |     1 |    -2 |    -2 |    -2 |  ... |     0.000 |     0.000 |     0.000 |    0.000 |    0.000 |    0.000 |    0.000 |    0.000 | 3971.000 |                          1 |

5 rows × 25 columns

In [6]:

```
# store the category code mapping, so we can encode any new incoming data
# other than our training set
cat_code_dict = {}
for col in cat_cols:
    category_col = df_train[col].astype('category')
    cat_code_dict[col] = {value: idx for idx, value in enumerate(category_col.cat.categories)} 

cat_code_dict
```

Out[6]:

```
{'EDUCATION': {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6},
 'SEX': {1: 0, 2: 1},
 'MARRIAGE': {0: 0, 1: 1, 2: 2, 3: 3}}
```

In [7]:

```
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(df_train[num_cols])
```

Out[7]:

```
StandardScaler()
```

In [8]:

```
def preprocess(df, scaler, cat_code_dict, num_cols, cat_cols, label_col):
    df = df.copy()

    # numeric fields
    df[num_cols] = scaler.transform(df[num_cols])
    df[num_cols] = df[num_cols].astype(np.float32)

    # categorical fields
    for col in cat_cols:
        code_dict = cat_code_dict[col]
        code_fillna_value = len(code_dict)
        df[col] = df[col].map(code_dict).fillna(code_fillna_value).astype(np.int64)

    # label
    df[label_col] = df[label_col].astype(np.float32)
    return df
```

In [9]:

```
df_groups = {
    'train': df_train,
    'val': df_val,
    'test': df_test
}

data_dir = 'onnx_data'
os.makedirs(data_dir, exist_ok=True)

for name, df_group in df_groups.items():
    filename = os.path.join(data_dir, f'{name}.csv')
    df_preprocessed = preprocess(df_group, scaler, cat_code_dict, num_cols, cat_cols, label_col)
    df_preprocessed.to_csv(filename, index=False)

df_preprocessed.dtypes
```

Out[9]:

```
ID                              int64
LIMIT_BAL                     float32
SEX                             int64
EDUCATION                       int64
MARRIAGE                        int64
AGE                           float32
PAY_0                         float32
PAY_2                         float32
PAY_3                         float32
PAY_4                         float32
PAY_5                         float32
PAY_6                         float32
BILL_AMT1                     float32
BILL_AMT2                     float32
BILL_AMT3                     float32
BILL_AMT4                     float32
BILL_AMT5                     float32
BILL_AMT6                     float32
PAY_AMT1                      float32
PAY_AMT2                      float32
PAY_AMT3                      float32
PAY_AMT4                      float32
PAY_AMT5                      float32
PAY_AMT6                      float32
default.payment.next.month    float32
dtype: object
```

## PyTorch Dataset

The next few code chunk involves understanding how to work with [Pytorch's Dataset and DataLoader](https://pytorch.org/docs/stable/data.html). We define a custom Dataset that allows us the load our preprocessed .csv file, and extract the numerical, categorical, and label columns.

In [10]:

```
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
```

In [11]:

```
class TabularDataset(Dataset):

    def __init__(self, path, num_cols, cat_cols, label_col):
        self.path = path
        self.num_cols = num_cols
        self.cat_cols = cat_cols
        self.label_col = label_col
        self.df = read_data(path, num_cols, cat_cols, label_col)

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        num_array = self.df[self.num_cols].iloc[idx].values
        cat_array = self.df[self.cat_cols].iloc[idx].values
        label_array = self.df[self.label_col].iloc[idx]
        return num_array, cat_array, label_array


def read_data(path, num_cols, cat_cols, label_col):
    float_cols = num_cols + [label_col]
    dtype = {col: np.float32 for col in float_cols}
    dtype.update({col: np.int64 for col in cat_cols})
    return pd.read_csv(path, dtype=dtype)
```

In [12]:

```
batch_size = 2

path_train = os.path.join(data_dir, 'train.csv')
dataset = TabularDataset(path_train, num_cols, cat_cols, label_col)
data_loader = DataLoader(dataset, batch_size)

# our data loader now returns batches of numerical/categorical/label tensor
num_tensor, cat_tensor, label_tensor = next(iter(data_loader))

print('numerical value tensor:\n', num_tensor)
print('categorical value tensor:\n', cat_tensor)
print('label tensor:\n', label_tensor)
numerical value tensor:
 tensor([[-1.1365, -1.3578,  0.8985,  1.7693,  1.8037, -1.5156, -1.5167, -1.4744,
         -0.4939, -0.4858, -0.6720, -0.6725, -0.6630, -0.6509, -0.3129, -0.2444,
         -0.3072, -0.2983, -0.3092, -0.2845],
        [-0.1374, -0.0537, -0.8610,  1.7693, -0.6883,  1.9076, -0.6384, -0.6082,
         -0.6658, -0.6794, -0.6392, -0.6544, -0.6602, -0.6161, -0.3424, -0.1522,
         -0.3072, -0.2882, -0.1755, -0.2845]])
categorical value tensor:
 tensor([[3, 1, 1],
        [3, 1, 2]])
label tensor:
 tensor([1., 0.])
```

Note that one serious downside for this particular dataset implementation is it reads the entire data into memory, for large datasets, this might not be feasible. We'll leave out this enhancements for now.

## PyTorch Lightning Module

We'll be leveraging [PyTorch Lightning](https://pytorch-lightning.readthedocs.io/en/latest/) for organizing our neural network model. I personally find it helpful when it comes to standardizing the structure, and avoid manually writing the training loop compared to vanilla PyTorch. This part is optional.

In [13]:

```
class TabularDataModule(pl.LightningDataModule):

    def __init__(self, data_dir, num_cols, cat_cols, label_col, num_workers=2,
                 batch_size_train=128, batch_size_val=64, batch_size_test=512):
        super().__init__()
        self.data_dir = data_dir
        self.num_cols = num_cols
        self.cat_cols = cat_cols
        self.label_col = label_col
        self.num_workers = num_workers
        self.batch_size_train = batch_size_train
        self.batch_size_val = batch_size_val
        self.batch_size_test = batch_size_test

    def setup(self, stage):
        num_cols = self.num_cols
        cat_cols = self.cat_cols
        label_col = self.label_col
        
        path_train = os.path.join(self.data_dir, 'train.csv')
        self.dataset_train = TabularDataset(path_train, num_cols, cat_cols, label_col)

        path_val = os.path.join(self.data_dir, 'val.csv')
        self.dataset_val = TabularDataset(path_val, num_cols, cat_cols, label_col)

        path_test = os.path.join(self.data_dir, 'test.csv')
        self.dataset_test = TabularDataset(path_test, num_cols, cat_cols, label_col)

    def train_dataloader(self):
        return DataLoader(
            self.dataset_train,
            num_workers=self.num_workers,
            batch_size=self.batch_size_train,
            shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset_val,
            num_workers=self.num_workers,
            batch_size=self.batch_size_val,
            shuffle=False
        )

    def test_dataloader(self):
        return DataLoader(
            self.dataset_test,
            num_workers=self.num_workers,
            batch_size=self.batch_size_test,
            shuffle=False
        )
```

One of the highlights of using deep learning with tabular data is to create an embedding layer for each of our categorical features, and concatenate them together with the rest of the other numerical features.

```
embedding(categorical feature 1) ---
                                   |
embedding(categorical feature 2) ---------> rest of the layers
                                   |
numerical features -----------------
```

In [14]:

```
class TabularNet(pl.LightningModule):

    def __init__(self, num_cols, cat_cols, embedding_size_dict, n_classes,
                 embedding_dim_dict=None, learning_rate=0.01):
        super().__init__()
        
        # pytorch lightning black magic, all the arguments can now be
        # accessed through self.hparams.[argument]
        self.save_hyperparameters()

        self.embeddings, total_embedding_dim = self._create_embedding_layers(
            cat_cols, embedding_size_dict, embedding_dim_dict)
        
        # concatenate the numerical variables and the embedding layers
        # then proceed with the rest of the sequential flow
        in_features = len(num_cols) + total_embedding_dim
        self.layers = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, n_classes)
        )

    @staticmethod
    def _create_embedding_layers(cat_cols, embedding_size_dict, embedding_dim_dict):
        """construct the embedding layer, 1 per each categorical variable"""
        total_embedding_dim = 0
        embeddings = {}
        for col in cat_cols:
            embedding_size = embedding_size_dict[col]
            embedding_dim = embedding_dim_dict[col]
            total_embedding_dim += embedding_dim
            embeddings[col] = nn.Embedding(embedding_size, embedding_dim)

        return nn.ModuleDict(embeddings), total_embedding_dim

    def forward(self, num_tensor, cat_tensor):

        # run through all the categorical variables through its
        # own embedding layer and concatenate them together
        cat_outputs = []
        for i, col in enumerate(self.hparams.cat_cols):
            embedding = self.embeddings[col]
            cat_output = embedding(cat_tensor[:, i])
            cat_outputs.append(cat_output)
        
        cat_outputs = torch.cat(cat_outputs, dim=1)
        
        # concatenate the categorical embedding and numerical layer
        all_outputs = torch.cat((num_tensor, cat_outputs), dim=1)
        
        # for binary classification or regression we don't need the additional dimension
        final_outputs = self.layers(all_outputs).squeeze(dim=1)
        return final_outputs

    def training_step(self, batch, batch_idx):
        loss = self.compute_loss(batch, batch_idx)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.compute_loss(batch, batch_idx)
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self.compute_loss(batch, batch_idx)
        self.log('test_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    def compute_loss(self, batch, batch_idx):
        num_tensor, cat_tensor, label_tensor = batch
        output_tensor = self(num_tensor, cat_tensor)
        loss = F.binary_cross_entropy_with_logits(output_tensor, label_tensor)
        return loss
```

The hyperparameters including the number of layers, units per layer and the embedding size below are all chosen in a somewhat arbitrary manner, feel free to perform some hyperparameter tuning or tweak the layer definition for better performance.

For example, when we don't specify the number of embeddings for our categorical variable, [fastai](https://docs.fast.ai/tabular.model#emb_sz_rule) defines a default number based on a rule of thumb corresponding to the number of distinct values for that categorical feature.

> Through trial and error, this general rule takes the lower of two values:
>
> - A dimension space of 600
> - A dimension space equal to 1.6 times the cardinality of the variable to 0.56.

```
def emb_sz_rule(n_cat):
    "Rule of thumb to pick embedding size corresponding to `n_cat`"
    return min(600, round(1.6 * n_cat ** 0.56))
```

In [15]:

```
n_classes = 1

embedding_size_dict = {col: len(code) for col, code in cat_code_dict.items()}
embedding_dim_dict = {col: embedding_size // 2 for col, embedding_size in embedding_size_dict.items()}
embedding_dim_dict
```

Out[15]:

```
{'EDUCATION': 3, 'SEX': 1, 'MARRIAGE': 2}
```

In [16]:

```
tabular_data_module = TabularDataModule(data_dir, num_cols, cat_cols, label_col)

# we can print out the network architecture for inspection
tabular_model = TabularNet(num_cols, cat_cols, embedding_size_dict, n_classes, embedding_dim_dict)
tabular_model
```

Out[16]:

```
TabularNet(
  (embeddings): ModuleDict(
    (EDUCATION): Embedding(7, 3)
    (MARRIAGE): Embedding(4, 2)
    (SEX): Embedding(2, 1)
  )
  (layers): Sequential(
    (0): Linear(in_features=26, out_features=128, bias=True)
    (1): ReLU()
    (2): Linear(in_features=128, out_features=256, bias=True)
    (3): ReLU()
    (4): Linear(in_features=256, out_features=1, bias=True)
  )
)
```

Upon defining the data module, and model module, we can pass it to pytorch lightning's `Trainer` which abstracts the manual training loop process for end users.

In [17]:

```
from pytorch_lightning.callbacks import EarlyStopping

callbacks = [EarlyStopping(monitor='val_loss')]
trainer = pl.Trainer(max_epochs=8, callbacks=callbacks)
trainer.fit(tabular_model, tabular_data_module)
GPU available: False, used: False
TPU available: False, using: 0 TPU cores

  | Name       | Type       | Params
------------------------------------------
0 | embeddings | ModuleDict | 31    
1 | layers     | Sequential | 36 K  
WARNING: Logging before flag parsing goes to stderr.
I1026 20:04:35.695845 4405480896 lightning.py:1288] 
  | Name       | Type       | Params
------------------------------------------
0 | embeddings | ModuleDict | 31    
1 | layers     | Sequential | 36 K  
```

Out[17]:

```
1
```

## Evaluation

We show how we can use the trained model for inference and evaluation on the test set. For the evaluation, we'll use standard binary classification evaluation metrics.

In [18]:

```
tabular_model.eval()
with torch.no_grad():
    model_pred = tabular_model(num_tensor, cat_tensor).cpu().numpy()

model_pred
```

Out[18]:

```
array([-0.02512708, -0.46855184], dtype=float32)
```

In [19]:

```
def predict(tabular_model, tabular_data_module):
    data_loader = tabular_data_module.test_dataloader()
    batch_size = data_loader.batch_size
    n_rows = len(tabular_data_module.dataset_test)
    
    y_true = np.zeros(n_rows, dtype=np.float32)
    y_pred = np.zeros(n_rows, dtype=np.float32)
    with torch.no_grad():
        idx = 0
        for num_batch, cat_batch, label_batch in data_loader:
            y_output = tabular_model(num_batch, cat_batch)

            # we convert the output value to binary classification probability
            # with a sigmoid operation, note that this step is specific to the
            # problem at hand, and might not apply to say a regression problem
            y_prob = torch.sigmoid(y_output).cpu().numpy()

            start_idx = idx
            idx += batch_size
            end_idx = idx
            y_pred[start_idx:end_idx] = y_prob
            y_true[start_idx:end_idx] = label_batch.cpu().numpy()

            if end_idx == n_rows:
                break

    return y_true, y_pred
```

In [20]:

```
y_true, y_pred = predict(tabular_model, tabular_data_module)
y_true, y_pred
```

Out[20]:

```
(array([0., 0., 0., ..., 0., 0., 1.], dtype=float32),
 array([0.22540408, 0.18922299, 0.07959805, ..., 0.11537173, 0.1427396 ,
        0.51938766], dtype=float32))
```

In [21]:

```
import sklearn.metrics as metrics


def compute_score(y_true, y_pred, round_digits=3):
    log_loss = round(metrics.log_loss(y_true, y_pred), round_digits)
    auc = round(metrics.roc_auc_score(y_true, y_pred), round_digits)

    precision, recall, threshold = metrics.precision_recall_curve(y_true, y_pred)
    f1 = 2 * (precision * recall) / (precision + recall)

    mask = ~np.isnan(f1)
    f1 = f1[mask]
    precision = precision[mask]
    recall = recall[mask]

    best_index = np.argmax(f1)
    threshold = round(threshold[best_index], round_digits)
    precision = round(precision[best_index], round_digits)
    recall = round(recall[best_index], round_digits)
    f1 = round(f1[best_index], round_digits)

    return {
        'auc': auc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'threshold': threshold,
        'log_loss': log_loss
    }
```

In [22]:

```
compute_score(y_true, y_pred)
```

Out[22]:

```
{'auc': 0.768,
 'precision': 0.534,
 'recall': 0.529,
 'f1': 0.531,
 'threshold': 0.319,
 'log_loss': 0.439}
```

In [23]:

```
tabular_model_loaded = TabularNet.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

tabular_model_loaded.eval()
with torch.no_grad():
    model_pred_loaded = tabular_model_loaded(num_tensor, cat_tensor).cpu().numpy()

y_true, y_pred = predict(tabular_model_loaded, tabular_data_module)
compute_score(y_true, y_pred)
```

Out[23]:

```
{'auc': 0.766,
 'precision': 0.508,
 'recall': 0.557,
 'f1': 0.532,
 'threshold': 0.262,
 'log_loss': 0.442}
```

## ONNX Runtime

To perform model inferencing in production, we can export our PyTorch model into ONNX format, and run it using [ONNX Runtime](https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html).

We'll first run a sample inference using the original PyTorch model and compare it with the output from ONNX Runtime to verfiy the result matches.

In [24]:

```
tabular_model.eval()
with torch.no_grad():
    torch_pred = tabular_model(num_tensor, cat_tensor).cpu().numpy()

torch_pred
```

Out[24]:

```
array([-0.02512708, -0.46855184], dtype=float32)
```

While exporting our model into ONNX format there are a couple of key parameters that we should specify.

- Names of the input/output. Instead of leaving it blank, and letting the underlying engine auto-generate the values, this makes our ONNX model easier to work with during inferencing stage.
- dynamic_axes. While exporting the model, we need to pass a sample input to inform the underlying engine about the size/shape. While doing so, it's important that we also specify the first dimension (batch size) of our model to be dynamic. This ensures during inferencing stage, the input batch size will not be fixed to the sample input's batch size we provided, but can take on any dynamic value.

In [25]:

```
filepath = 'model.onnx'
args = num_tensor, cat_tensor
input_names = ['num_tensor', 'cat_tensor']
output_names = ['score']
dynamic_axes = {
    'num_tensor': {0 : 'batch_size'},
    'cat_tensor' : {0 : 'batch_size'}
}

torch.onnx.export(
    tabular_model,
    args,
    filepath,
    input_names=input_names,
    output_names=output_names,
    dynamic_axes=dynamic_axes
)
```

In [26]:

```
import onnxruntime

# create the onnx runtime with the exported .onnx model file 
ort_session = onnxruntime.InferenceSession(filepath)

# during inference time, we can be very specific about inputs and outputs
ort_inputs = {
    'num_tensor': num_tensor.detach().numpy(),
    'cat_tensor': cat_tensor.detach().numpy()
}
# call .run on the onnx runtime session with the desired input and output
# the output is a list, but we only need the first output for this example
ort_pred = ort_session.run(['score'], ort_inputs)[0]

np.allclose(torch_pred, ort_pred)
print("Exported model has been tested with ONNXRuntime, and the result looks good!")
ort_pred
Exported model has been tested with ONNXRuntime, and the result looks good!
```

Out[26]:

```
array([-0.02512729, -0.46855184], dtype=float32)
```

Once we converted our model to an onnx format, we can also leverage APIs in different languages to run the inference step. e.g. [onnxruntime's Java API](https://github.com/microsoft/onnxruntime/blob/master/docs/Java_API.md) for performing scoring on a JVM.

## Ending Notes

There are many enhancements that can be introduced such as:

- making the preprocessing step more robust
- implementing a pytorch dataset that scales to larger dataset
- adding complexity to the model architecture
- etc.

But hopefully this gets the basic concepts/workflow across, and make whetted your appetite to try it out for the next modeling challenge. Though do start with a baseline model (e.g. tree-based methods) prior to adding all sorts of complexity.