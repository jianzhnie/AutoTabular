from dataclasses import field
from typing import Any, Dict, List

import numpy as np
import torch
from autotabular.algorithms.deeptabular.models.abstract_model import TabularModel
from autotabular.algorithms.deeptabular.models.tabnet import TabNet
from pytorch_tabnet.multiclass_utils import check_output_dim, infer_output_dim
from pytorch_tabnet.utils import PredictDataset, create_explain_matrix, filter_weights
from scipy.special import softmax
from torch.utils.data import DataLoader


class TabNetClassifier(TabularModel):

    def __init__(
        self,
        n_d: int = 8,
        n_a: int = 8,
        n_steps: int = 3,
        gamma: float = 1.3,
        cat_idxs: List[int] = field(default_factory=list),
        cat_dims: List[int] = field(default_factory=list),
        cat_emb_dim: int = 1,
        n_independent: int = 2,
        n_shared: int = 2,
        epsilon: float = 1e-15,
        momentum: float = 0.02,
        lambda_sparse: float = 1e-3,
        seed: int = 0,
        clip_value: int = 1,
        verbose: int = 1,
        optimizer_fn: Any = torch.optim.Adam,
        optimizer_params: Dict = field(default_factory=lambda: dict(lr=2e-2)),
        scheduler_fn: Any = None,
        scheduler_params: Dict = field(default_factory=dict),
        mask_type: str = 'sparsemax',
        input_dim: int = None,
        output_dim: int = None,
        device_name: str = 'auto',
        n_shared_decoder: int = 1,
        n_indep_decoder: int = 1,
        optimizer_fn=torch.optim.Adam, # Any optimizer works here
        optimizer_params=dict(lr=2e-2),
        scheduler_fn=torch.optim.lr_scheduler.OneCycleLR,
        scheduler_params={"is_batch_level":True,
                            "max_lr":5e-2,
                            "steps_per_epoch":int(train.shape[0] / batch_size)+1,
                            "epochs":max_epochs
                            },
        mask_type='entmax', # "sparsemax",


    ):

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma
        self.cat_idxs = cat_idxs
        self.cat_dims = cat_dims
        self.cat_emb_dim = cat_emb_dim
        self.n_independent = n_independent
        self.n_shared = n_shared
        self.epsilon = epsilon
        self.virtual_batch_size = virtual_batch_size
        self.momentum = momentum
        self.mask_type = mask_type

    def __post_init__(self):
        super(TabNetClassifier, self).__post_init__()
        self._task = 'classification'
        self._default_loss = torch.nn.functional.cross_entropy
        self._default_metric = 'accuracy'

    def weight_updater(self, weights):
        """Updates weights dictionary according to target_mapper.

        Parameters
        ----------
        weights : bool or dict
            Given weights for balancing training.

        Returns
        -------
        bool or dict
            Same bool if weights are bool, updated dict otherwise.
        """
        if isinstance(weights, int):
            return weights
        elif isinstance(weights, dict):
            return {
                self.target_mapper[key]: value
                for key, value in weights.items()
            }
        else:
            return weights

    def prepare_target(self, y):
        return np.vectorize(self.target_mapper.get)(y)

    def compute_loss(self, y_pred, y_true):
        return self.loss_fn(y_pred, y_true.long())

    def update_fit_params(
        self,
        X_train,
        y_train,
        eval_set,
        weights,
    ):
        output_dim, train_labels = infer_output_dim(y_train)
        for X, y in eval_set:
            check_output_dim(train_labels, y)
        self.output_dim = output_dim
        self._default_metric = ('auc' if self.output_dim == 2 else 'accuracy')
        self.classes_ = train_labels
        self.target_mapper = {
            class_label: index
            for index, class_label in enumerate(self.classes_)
        }
        self.preds_mapper = {
            str(index): class_label
            for index, class_label in enumerate(self.classes_)
        }
        self.updated_weights = self.weight_updater(weights)

    def stack_batches(self, list_y_true, list_y_score):
        y_true = np.hstack(list_y_true)
        y_score = np.vstack(list_y_score)
        y_score = softmax(y_score, axis=1)
        return y_true, y_score

    def predict_func(self, outputs):
        outputs = np.argmax(outputs, axis=1)
        return np.vectorize(self.preds_mapper.get)(outputs.astype(str))

    def predict_proba(self, X):
        """Make predictions for classification on a batch (valid)

        Parameters
        ----------
        X : a :tensor: `torch.Tensor`
            Input data

        Returns
        -------
        res : np.ndarray
        """
        self.network.eval()

        dataloader = DataLoader(
            PredictDataset(X),
            batch_size=self.batch_size,
            shuffle=False,
        )

        results = []
        for batch_nb, data in enumerate(dataloader):
            data = data.to(self.device).float()

            output, M_loss = self.network(data)
            predictions = torch.nn.Softmax(
                dim=1)(output).cpu().detach().numpy()
            results.append(predictions)
        res = np.vstack(results)
        return res

    def _set_network(self):
        """Setup the network and explain matrix."""
        self.network = TabNet(
            self.input_dim,
            self.output_dim,
            n_d=self.n_d,
            n_a=self.n_a,
            n_steps=self.n_steps,
            gamma=self.gamma,
            cat_idxs=self.cat_idxs,
            cat_dims=self.cat_dims,
            cat_emb_dim=self.cat_emb_dim,
            n_independent=self.n_independent,
            n_shared=self.n_shared,
            epsilon=self.epsilon,
            virtual_batch_size=self.virtual_batch_size,
            momentum=self.momentum,
            mask_type=self.mask_type,
        ).to(self.device)

        self.reducing_matrix = create_explain_matrix(
            self.network.input_dim,
            self.network.cat_emb_dim,
            self.network.cat_idxs,
            self.network.post_embed_dim,
        )


class TabNetRegressor(TabularModel):

    def __post_init__(self):
        super(TabNetRegressor, self).__post_init__()
        self._task = 'regression'
        self._default_loss = torch.nn.functional.mse_loss
        self._default_metric = 'mse'

    def prepare_target(self, y):
        return y

    def compute_loss(self, y_pred, y_true):
        return self.loss_fn(y_pred, y_true)

    def update_fit_params(self, X_train, y_train, eval_set, weights):
        if len(y_train.shape) != 2:
            msg = 'Targets should be 2D : (n_samples, n_regression) ' + \
                  f'but y_train.shape={y_train.shape} given.\n' + \
                  'Use reshape(-1, 1) for single regression.'
            raise ValueError(msg)
        self.output_dim = y_train.shape[1]
        self.preds_mapper = None

        self.updated_weights = weights
        filter_weights(self.updated_weights)

    def predict_func(self, outputs):
        return outputs

    def stack_batches(self, list_y_true, list_y_score):
        y_true = np.vstack(list_y_true)
        y_score = np.vstack(list_y_score)
        return y_true, y_score
