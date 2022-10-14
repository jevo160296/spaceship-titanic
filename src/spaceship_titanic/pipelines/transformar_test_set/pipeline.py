"""
This is a boilerplate pipeline 'transformar_test_set'
generated using Kedro 0.18.3
"""

from kedro.pipeline import Pipeline, node, pipeline

from spaceship_titanic.pipelines.transformar_test_set.nodes import transformar_test_set


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=transformar_test_set,
            inputs=['test_dataset', 'transformer_entrenado', 'parameters'],
            outputs='test_data_procesada'
        )
    ])
