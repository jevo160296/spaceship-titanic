"""
This is a boilerplate pipeline 'predecir_test_set'
generated using Kedro 0.18.3
"""

from kedro.pipeline import Pipeline, node, pipeline

from spaceship_titanic.pipelines.predecir_test_set.nodes import predecir_test_set, \
    reportar_tamano_test_orignal_y_procesado


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=predecir_test_set,
            inputs=['test_data_procesada', 'modelo_entrenado', 'parameters'],
            outputs='y_test_predict'
        ),
        node(
            func=reportar_tamano_test_orignal_y_procesado,
            inputs=['test_dataset', 'y_test_predict', 'parameters'],
            outputs='reporte_tamano'
        )
    ])
