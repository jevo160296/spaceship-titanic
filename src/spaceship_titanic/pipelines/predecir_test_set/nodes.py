"""
This is a boilerplate pipeline 'predecir_test_set'
generated using Kedro 0.18.3
"""
from typing import Dict, Any

from pandas import DataFrame, Series
from sklearn.tree import DecisionTreeClassifier
import mlflow

from spaceship_titanic.core.parameters import Parameters


def predecir_test_set(data: DataFrame,
                      modelo: DecisionTreeClassifier,
                      parameters: Dict[str, Any]) -> Series:
    parameters = Parameters(parameters)
    # y_test_real = data[parameters.y_name]
    y_test_predict: Series = Series(data=modelo.predict(data[parameters.X_names]),
                                    index=data['PassengerId'],
                                    name='Transported')
    return y_test_predict


def reportar_tamano_test_orignal_y_procesado(
        test_data_set: DataFrame,
        y_test_predict: Series,
        parameters: Dict[str, Any]
) -> str:
    parameters = Parameters(parameters)
    n_filas_original = test_data_set.shape[0]
    n_filas_procesado = len(y_test_predict)
    reporte = f'Tamaño de entrada: {n_filas_original}\nTamaño de salida: {n_filas_procesado}'

    mlflow.set_experiment(parameters.mlflow_experiment_name)
    mlflow.log_text(reporte, 'artifacts/reporte.txt')
    mlflow.log_param('test_cant_rows_original', n_filas_original)
    mlflow.log_param('test_cant_rows_procesado', n_filas_procesado)

    return reporte
