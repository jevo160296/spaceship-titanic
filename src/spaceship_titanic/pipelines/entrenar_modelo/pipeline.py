"""
This is a boilerplate pipeline 'entrenar_modelo'
generated using Kedro 0.18.3
"""

from kedro.pipeline import Pipeline, node, pipeline

from spaceship_titanic.pipelines.entrenar_modelo.nodes import entrenar_modelo, train_predict


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=entrenar_modelo,
            inputs=['train_data_procesada', 'parameters'],
            outputs='modelo_entrenado'
        ),
        node(
            func=train_predict,
            inputs=['train_data_procesada', 'modelo_entrenado', 'parameters'],
            outputs=['y_train_real', 'y_train_predict']
        )
    ]
    )
