from pathlib import Path

import pandas as pd
from pandas import DataFrame


class Data:
    path_data = Path(r'data').resolve().absolute()
    path_raw = path_data.joinpath(r'01_raw')
    path_raw_train = path_raw.joinpath(r'train.parquet')
    path_intermediate = path_data.joinpath(r'02_intermediate')

    @staticmethod
    def get_csv_raw_train_data():
        return pd.read_csv(Data.path_raw.joinpath(r'train.csv'))

    @staticmethod
    def load_data(path: Path) -> DataFrame:
        return pd.read_parquet(path)

    @staticmethod
    def save_data(data: DataFrame, path: Path) -> Path:
        data.to_parquet(path)
        return path
