"""
This is a boilerplate pipeline 'EntrenarProcesamiento'
generated using Kedro 0.18.3
"""
from typing import Dict, Any, Tuple, Union

from pandas import DataFrame
from sklearn.compose import make_column_transformer, ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin

from spaceship_titanic.core.parameters import Parameters


def _column_selector(data: DataFrame, columns: list) -> DataFrame:
    return data[columns]


class SklearnTransformerDataFrame(BaseEstimator, TransformerMixin):
    def __init__(self, transformer):
        self.transformer = transformer

    def fit(self, X, y=None):
        return self.transformer.fit(X)

    def transform(self, X: DataFrame, y=None) -> DataFrame:
        X_t = self.transformer.transform(X)
        columns = self.transformer.get_feature_names_out()
        X_t_df = DataFrame(X_t, columns=columns, index=X.index)
        return X_t_df


def entrenar_column_selector(data: DataFrame, parameters: Dict[str, Any]) -> Tuple[DataFrame, FunctionTransformer]:
    parameters = Parameters(parameters)
    data = data.reindex(columns=parameters.X_names + [parameters.y_name]).copy()
    column_selector = FunctionTransformer(_column_selector,
                                          kw_args={'columns': parameters.X_names + [parameters.y_name]})
    data_transformed = column_selector.transform(data)
    return data_transformed, column_selector


def entrenar_simple_imputer(data: DataFrame, parameters: Dict[str, Any]):
    parameters = Parameters(parameters)
    data = data.reindex(columns=parameters.X_names + [parameters.y_name]).copy()
    imputer_categoricas = SimpleImputer(strategy='most_frequent')
    columnas_categoricas = parameters.columns_classification['categorical']
    columnas_categoricas = columnas_categoricas.intersection(data.columns)
    imputer_numericas = SimpleImputer(strategy='mean')
    columnas_numericas = parameters. \
        columns_classification['continuous']. \
        union(parameters.columns_classification['discrete'])
    columnas_numericas = columnas_numericas.intersection(data.columns)
    columnas_binarias = parameters.columns_classification['binary'].difference([parameters.y_name])
    columnas_binarias = columnas_binarias.intersection(data.columns)
    imputer_binarias = SimpleImputer(strategy='most_frequent')
    imputer_pipeline = Pipeline(
        [
            ('column_transformer', make_column_transformer(
                (imputer_numericas, list(columnas_numericas)),
                (imputer_categoricas, list(columnas_categoricas)),
                (imputer_binarias, list(columnas_binarias)),
                remainder='passthrough',
                verbose_feature_names_out=False
            ))
        ]
    )
    imputer_pipeline_df = SklearnTransformerDataFrame(imputer_pipeline)
    imputer_pipeline_df.fit(data)
    data_procesada = imputer_pipeline_df.transform(data)
    return data_procesada, imputer_pipeline_df


def entrenar_ordinal_encoder(data: DataFrame, parameters: Dict[str, Any]) -> Tuple[DataFrame,
                                                                                   SklearnTransformerDataFrame]:
    parameters = Parameters(parameters)
    data = data.reindex(columns=parameters.X_names + [parameters.y_name]).copy()
    oe = OrdinalEncoder()
    ct = make_column_transformer((oe, parameters.columnas_ordinal_encoder), remainder='passthrough',
                                 verbose_feature_names_out=False)
    ct_df = SklearnTransformerDataFrame(ct)
    ct_df.fit(data)
    data = ct_df.transform(data)
    return data, ct_df


def construir_sklearn_pipeline(
        data: DataFrame,
        ordinal_encoder: SklearnTransformerDataFrame,
        simple_imputer: SklearnTransformerDataFrame,
        column_selector: SklearnTransformerDataFrame
):
    transformer_entrenado = Pipeline(
        [
            ('column_selector', column_selector),
            ('simple_imputer', simple_imputer),
            ('ordinal_encoder', ordinal_encoder)
        ]
    )
    return data, transformer_entrenado
