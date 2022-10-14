"""
This is a boilerplate pipeline 'evaluation'
generated using Kedro 0.18.3
"""
from typing import Dict, Any

from pandas import Series
from sklearn.metrics import f1_score

import mlflow

from spaceship_titanic.core.parameters import Parameters


def evaluate_results(
        y_train_real: Series,
        y_train_predict: Series,
        parameters: Dict[str, Any]
) -> str:
    parameters = Parameters(parameters)
    train_score = f1_score(y_train_real, y_train_predict, pos_label=True)

    mlflow.set_experiment(parameters.mlflow_experiment_name)
    mlflow.log_param('train_score', round(train_score, 2))
    return f'{train_score=:.2%}'
