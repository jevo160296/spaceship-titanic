"""
This is a boilerplate pipeline 'LimpiarDatos'
generated using Kedro 0.18.3
"""
from functools import reduce
from typing import Dict, Any

from pandas import DataFrame

from spaceship_titanic.core.parameters import Parameters


def eliminar_outliers(data: DataFrame, parameters: Dict[str, Any]):
    parameters = Parameters(parameters)
    columns_classification = parameters.columns_classification
    es_outlier = reduce(lambda a, b: a | b, [data[column] >= 20000 for column in columns_classification['continuous']])
    return data[~es_outlier]


def eliminar_nulos(data: DataFrame, parameters: Dict[str, Any]):
    parameters = Parameters(parameters)
    return data.dropna(subset=parameters.columnas_a_eliminar_nulos)
