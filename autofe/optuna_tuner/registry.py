"""tasks that can be handled by the package."""
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, roc_auc_score

BINARY_CLASSIFICATION = 'binary_classification'
MULTICLASS_CLASSIFICATION = 'multiclass_classification'
REGRESSION = 'regression'

support_ml_task = [
    BINARY_CLASSIFICATION, MULTICLASS_CLASSIFICATION, REGRESSION
]

default_task_metric = {
    BINARY_CLASSIFICATION: 'auc',
    MULTICLASS_CLASSIFICATION: 'accuracy',
    REGRESSION: 'r2',
}

get_metric_fn = {
    'auc': roc_auc_score,
    'accuracy': accuracy_score,
    'r2': r2_score,
    'mse': mean_squared_error
}

default_optimizer_direction = {
    'auc': 'maximize',
    'accuracy': 'maximize',
    'r2': 'maximize',
    'mse': 'minimize'
}
