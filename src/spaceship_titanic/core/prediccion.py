from pathlib import Path
from typing import TypedDict, List

from kedro.config import ConfigLoader
from kedro.extras.datasets.pickle import PickleDataSet
from kedro.framework.project import settings
from pandas import DataFrame
from sklearn.pipeline import Pipeline

from src.spaceship_titanic.core.parameters import Catalog, Parameters


class DatosEntrada(TypedDict):
    CryoSleep: List[bool]
    FoodCourt: List[float]
    HomePlanet: List[str]
    Age: List[float]
    ShoppingMall: List[float]
    Spa: List[float]
    VRDeck: List[float]
    RoomService: List[float]
    Destination: List[str]
    VIP: List[bool]
    Transported: List[bool]


def get_config(project_path: Path, ) -> ConfigLoader:
    conf_path = str(project_path / settings.CONF_SOURCE)
    conf_loader = ConfigLoader(conf_source=conf_path, env="local")
    return conf_loader


def predecir(datos_entrada: DatosEntrada, project_path: Path):
    config_loader = get_config(project_path)
    catalog = Catalog(**config_loader.get('catalog*'))
    parameters = Parameters(**config_loader.get('parameters*'))
    transformer_definition = PickleDataSet(filepath=
                                           str(project_path.joinpath(catalog['transformer_entrenado']['filepath'])))
    modelo_definition = PickleDataSet(filepath=
                                      str(project_path.joinpath(catalog['modelo_entrenado']['filepath'])))
    transformer: Pipeline = transformer_definition.load()
    modelo = modelo_definition.load()
    df_entrada = DataFrame(datos_entrada)
    datos_transformados = transformer.transform(df_entrada)
    prediccion = modelo.predict(X=datos_transformados[parameters.X_names])
    return prediccion

