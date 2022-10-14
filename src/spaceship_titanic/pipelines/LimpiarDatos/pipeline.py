"""
This is a boilerplate pipeline 'LimpiarDatos'
generated using Kedro 0.18.3
"""

from kedro.pipeline import Pipeline, node, pipeline
from spaceship_titanic.pipelines.LimpiarDatos.nodes import eliminar_outliers, eliminar_nulos


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=eliminar_outliers,
            inputs=['train_dataset', 'parameters'],
            outputs='train_dataset_sin_outliers'
        ),
        node(
            func=eliminar_nulos,
            inputs=['train_dataset_sin_outliers', 'parameters'],
            outputs='cleaned_data'
        )
    ],
        namespace='LimpiarDatos',
        inputs='train_dataset',
        outputs='cleaned_data'
    )
