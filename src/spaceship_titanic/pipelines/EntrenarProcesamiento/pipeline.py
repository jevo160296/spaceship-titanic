"""
This is a boilerplate pipeline 'EntrenarProcesamiento'
generated using Kedro 0.18.3
"""

from kedro.pipeline import Pipeline, node, pipeline

from spaceship_titanic.pipelines.EntrenarProcesamiento.nodes import entrenar_ordinal_encoder, entrenar_simple_imputer, \
    construir_sklearn_pipeline, entrenar_column_selector


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=entrenar_column_selector,
            inputs=['cleaned_data', 'parameters'],
            outputs=['train_data_column_select', 'column_selector_entrenado']
        ),
        node(
            func=entrenar_ordinal_encoder,
            inputs=['train_data_column_select', 'parameters'],
            outputs=['train_data_ordinal_encoder', 'ordinal_transformer_entrenado']
        ),
        node(
            func=entrenar_simple_imputer,
            inputs=['train_data_ordinal_encoder', 'parameters'],
            outputs=['train_data_imputer', 'simple_imputer_entrenado']
        ),
        node(
            func=construir_sklearn_pipeline,
            inputs=['train_data_imputer', 'ordinal_transformer_entrenado', 'simple_imputer_entrenado',
                    'column_selector_entrenado'],
            outputs=['train_data_procesada', 'transformer_entrenado']
        )
    ],
        namespace='EntrenarTransformers',
        inputs='cleaned_data',
        outputs={'train_data_procesada', 'transformer_entrenado'}
    )
