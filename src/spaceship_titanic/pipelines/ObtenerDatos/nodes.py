"""
This is a boilerplate pipeline 'ObtenerDatos'
generated using Kedro 0.18.3
"""
from typing import Tuple

import pandas as pd
from pandas import DataFrame

from spaceship_titanic.core.data_access import Data


def get_data() -> Tuple[DataFrame, DataFrame]:
    return (pd.read_csv(Data.path_raw.joinpath(r'train.csv')),
            pd.read_csv(Data.path_raw.joinpath(f'test.csv')))
