"""
This is a boilerplate pipeline 'EntrenarProcesamiento'
generated using Kedro 0.18.3
"""
from typing import Dict, Any, Tuple

from pandas import DataFrame
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, FunctionTransformer

from spaceship_titanic.core.parameters import Parameters


def _column_selector(data: DataFrame, columns: list) -> DataFrame:
    return data[columns]


def entrenar_column_selector(data: DataFrame, parameters: Dict[str, Any]) -> Tuple[DataFrame, FunctionTransformer]:
    parameters = Parameters(parameters)
    column_selector = FunctionTransformer(_column_selector,
                                          kw_args={'columns': parameters.X_names + [parameters.y_name]})
    data_transformed = column_selector.transform(data)
    return data_transformed, column_selector


def entrenar_ordinal_encoder(data: DataFrame, parameters: Dict[str, Any]) -> Tuple[DataFrame, OrdinalEncoder]:
    parameters = Parameters(parameters)
    data = data.copy()
    oe = OrdinalEncoder()
    oe.fit(data[parameters.columnas_ordinal_encoder])
    data[parameters.columnas_ordinal_encoder] = oe.transform(data[parameters.columnas_ordinal_encoder])
    return data, oe


def entrenar_simple_imputer(data: DataFrame, parameters: Dict[str, Any]):
    parameters = Parameters(parameters)
    data = data.copy()
    imputer_categoricas = SimpleImputer(strategy='median')
    columnas_categoricas = parameters.columns_classification['categorical']
    columnas_categoricas = columnas_categoricas.intersection(data.columns)
    imputer_numericas = SimpleImputer(strategy='mean')
    columnas_numericas = parameters. \
        columns_classification['continuous']. \
        union(parameters.columns_classification['discrete'])
    columnas_numericas = columnas_numericas.intersection(data.columns)
    imputer_pipeline = Pipeline(
        [
            ('column_transformer', make_column_transformer(
                (imputer_numericas, list(columnas_numericas)),
                (imputer_categoricas, list(columnas_categoricas)),
                remainder='passthrough',
                verbose_feature_names_out=False
            ))
        ]
    )
    imputer_pipeline.fit(data)
    data_procesada = imputer_pipeline.transform(data)
    return data_procesada, imputer_pipeline


def construir_sklearn_pipeline(
        data: DataFrame,
        ordinal_encoder: OrdinalEncoder,
        simple_imputer: SimpleImputer,
        column_selector: FunctionTransformer
):
    transformer_entrenado = Pipeline(
        [
            ('column_selector', column_selector),
            ('ordinal_encoder', ordinal_encoder),
            ('simple_imputer', simple_imputer)
        ]
    )
    return data, transformer_entrenado
