"""
This is a boilerplate pipeline 'entrenar_modelo'
generated using Kedro 0.18.3
"""
from typing import Dict, Any, Tuple

from pandas import DataFrame, Series
from sklearn.tree import DecisionTreeClassifier

from spaceship_titanic.core.parameters import Parameters
import mlflow


def entrenar_modelo(data: DataFrame, parameters: Dict[str, Any]) -> DecisionTreeClassifier:
    parameters = Parameters(parameters)
    tc = DecisionTreeClassifier(max_depth=8)
    X = data[parameters.X_names]
    y = data[parameters.y_name].astype(bool)
    tc.fit(X, y)
    mlflow.set_experiment(parameters.mlflow_experiment_name)
    mlflow.log_text(parameters.mlflow_experiment_description, 'artifacts/desciption.txt')
    return tc


def train_predict(data: DataFrame, modelo: DecisionTreeClassifier, parameters: Dict[str, Any]) -> Tuple[list, list]:
    parameters = Parameters(parameters)
    y_train_real = list(data[parameters.y_name])
    y_train_predict = modelo.predict(data[parameters.X_names])
    return y_train_real, y_train_predict
