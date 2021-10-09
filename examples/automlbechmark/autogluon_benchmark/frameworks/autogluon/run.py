from autogluon.tabular import TabularPredictor


def run(X_train,
        y_train,
        label: str,
        init_args: dict = None,
        fit_args: dict = None):
    if init_args is None:
        init_args = {}
    if fit_args is None:
        fit_args = {}

    X_train[label] = y_train

    predictor = TabularPredictor(
        label=label, **init_args).fit(
            train_data=X_train, **fit_args)

    return predictor
