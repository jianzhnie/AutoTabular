'''
Author: jianzhnie
Date: 2021-11-05 18:08:17
LastEditTime: 2021-11-05 18:25:35
LastEditors: jianzhnie
Description: 

'''
from pl_bolts.datamodules.sklearn_datamodule import SklearnDataModule
from sklearn.datasets import load_diabetes
from linear_regression import LinearRegression
from pytorch_lightning import LightningModule, Trainer, seed_everything


class TabularModel:

    def __init__(self, model=None):
        self.model = model

    def fit(self, X, y) -> None:
        """The fit method which takes in the data and triggers the training

        Args:
            train (pd.DataFrame): Training Dataframe

            valid (Optional[pd.DataFrame], optional): If provided, will use this dataframe as the validation while training.
                Used in Early Stopping and Logging. If left empty, will use 20% of Train data as validation. Defaults to None.

            test (Optional[pd.DataFrame], optional): If provided, will use as the hold-out data,
                which you'll be able to check performance after the model is trained. Defaults to None.

            loss (Optional[torch.nn.Module], optional): Custom Loss functions which are not in standard pytorch library

            metrics (Optional[List[Callable]], optional): Custom metric functions(Callable) which has the
                signature metric_fn(y_hat, y) and works on torch tensor inputs

            optimizer (Optional[torch.optim.Optimizer], optional): Custom optimizers which are a drop in replacements for standard PyToch optimizers.
                This should be the Class and not the initialized object

            optimizer_params (Optional[Dict], optional): The parmeters to initialize the custom optimizer.

            train_sampler (Optional[torch.utils.data.Sampler], optional): Custom PyTorch batch samplers which will be passed to the DataLoaders. Useful for dealing with imbalanced data and other custom batching strategies

            target_transform (Optional[Union[TransformerMixin, Tuple(Callable)]], optional): If provided, applies the transform to the target before modelling
                and inverse the transform during prediction. The parameter can either be a sklearn Transformer which has an inverse_transform method, or
                a tuple of callables (transform_func, inverse_transform_func)

            max_epochs (Optional[int]): Overwrite maximum number of epochs to be run

            min_epochs (Optional[int]): Overwrite minimum number of epochs to be run

            reset: (bool): Flag to reset the model and train again from scratch

            seed: (int): If you have to override the default seed set as part of of ModelConfig
        """
        X, y = load_diabetes(return_X_y=True)  # these are numpy arrays
        loaders = SklearnDataModule(X, y, batch_size=128)

        self.model.train()
        self.trainer = Trainer(max_epochs=10)
        self.trainer.fit(
            self.model,
            train_loader=loaders.train_dataloader(),
            val_loader=loaders.train_dataloader())

    def predct(self):
        pass


if __name__ == '__main__':
    model = LinearRegression(input_dim=10, l1_strength=1, l2_strength=1)