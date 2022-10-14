"""
This is a boilerplate pipeline 'evaluation'
generated using Kedro 0.18.3
"""

from kedro.pipeline import Pipeline, node, pipeline

from spaceship_titanic.pipelines.evaluation.nodes import evaluate_results


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=evaluate_results,
            inputs=['y_train_real', 'y_train_predict', 'parameters'],
            outputs='train_score',
            name='evaluate_results'
        )
    ])
