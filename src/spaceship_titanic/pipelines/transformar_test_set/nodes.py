"""
This is a boilerplate pipeline 'transformar_test_set'
generated using Kedro 0.18.3
"""
from typing import Dict, Any

from pandas import DataFrame
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from spaceship_titanic.core.parameters import Parameters


def transformar_test_set(data: DataFrame,
                         transformer_entrenado: Pipeline,
                         parameters: Dict[str, Any]) -> DataFrame:
    parameters = Parameters(parameters)
    data.index = data['PassengerId']
    data = data.reindex(columns=parameters.X_names).copy()
    data['Transported'] = 1
    data = transformer_entrenado.transform(data)
    return data
