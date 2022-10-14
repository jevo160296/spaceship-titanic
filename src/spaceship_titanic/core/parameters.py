from typing import Dict, FrozenSet, List


class Parameters(dict):
    @property
    def mlflow_experiment_description(self):
        return self['mlflow_experiment_description']

    @property
    def mlflow_experiment_name(self):
        return self['mlflow_experiment_name']

    @property
    def columns_classification(self) -> Dict[str, FrozenSet]:
        columns_classification_casted = {key: frozenset(value) for key, value in self['columns_classification'].items()}
        return columns_classification_casted

    @property
    def y_name(self) -> str:
        return self['y_name']

    @property
    def columnas_ordinal_encoder(self):
        return self['columnas_ordinal_encoder']

    @property
    def X_names(self) -> List[str]:
        return self['X_names']

    @property
    def columnas_a_eliminar_nulos(self) -> List[str]:
        return self['columnas_a_eliminar_nulos']
