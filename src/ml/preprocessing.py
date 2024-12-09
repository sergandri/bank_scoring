from typing import Literal

import numpy as np

from src.tools.data_config import BKI_LOCAL_PATH, PreprocessConfig, \
    TARGET_LOCAL_PATH, TEST_LOCAL_PATH
import pandas as pd

from src.tools.logger import logger


class Preprocessor:
    def __init__(self, df: pd.DataFrame):
        self.df = df


def read_dataframes() -> tuple[pd.DataFrame, pd.DataFrame]:
    df_target = pd.read_csv(TARGET_LOCAL_PATH)
    df_bki = pd.read_csv(BKI_LOCAL_PATH, low_memory=False)
    return df_target, df_bki


def read_test_dataframe() -> pd.DataFrame:
    return pd.read_csv(TEST_LOCAL_PATH,low_memory=False)


def convert_data_types(
    df: pd.DataFrame,
    preprocess_config: PreprocessConfig,
) -> pd.DataFrame:
    logger.info("Converting data types...")
    for column, dtype in preprocess_config.dtype_map.items():
        if dtype.startswith('datetime64'):
            df[column] = pd.to_datetime(df[column], errors='coerce')
        elif dtype == 'category':
            df[column] = df[column].astype('int32')
        else:
            df[column] = df[column].astype(dtype)

    return df.sort_values(by='fund_date', ascending=False)


def fill_missing_values(
    df: pd.DataFrame,
    preprocess_config: PreprocessConfig,
    stage: Literal[1, 2],
) -> pd.DataFrame:
    """
    Заполняет пропуски в датафрейме на основе логики из словаря.
    """
    if stage == 1:
        logger.info("Filling missing values...")
        for column, fill_function in preprocess_config.fillna_logic.items():
            if column in df.columns:
                df[column] = fill_function(df[column])
    elif stage == 2:
        how = 99
        if isinstance(how, int):
            value_to_fill: int = how
        else:
            value_to_fill: int = 0
        """Заполняет пропуски"""
        numeric_cat_columns = df.select_dtypes(
            include=['float64', 'int64', 'int8']
        ).columns
        df[numeric_cat_columns] = df[numeric_cat_columns].fillna(value_to_fill)

        other_columns = df.select_dtypes(
            include=['object']
        ).columns
        df[numeric_cat_columns] = df[numeric_cat_columns].fillna(value_to_fill)
    else:
        raise ValueError("stage must be 1 or 2")
    return df


def drop_features(
    df: pd.DataFrame,
    target_col: str = "target",
    threshold: float = 0.95,
):
    """Очищает датасет, удаляя ненужные признаки."""
    # 1. Удаление признаков с большим количеством пропусков
    missing_percentage = df.isnull().mean()
    cols_to_drop = missing_percentage[
        missing_percentage > threshold].index.tolist()
    logger.info(
        f"deleted {len(cols_to_drop)} features with missing:"
        f" {cols_to_drop}"
    )

    # 2. Ручное удаление
    id_cols = ['client_id']

    # 3. Итог
    all_cols_to_drop = set(cols_to_drop + id_cols)
    if target_col in all_cols_to_drop:
        all_cols_to_drop.remove(target_col)  # Таргет не должен быть удален
    cleaned_df = df.drop(columns=all_cols_to_drop)

    return cleaned_df


def replace_inf_with_nan(df: pd.DataFrame) -> pd.DataFrame:
    """Заменяет значения inf и -inf в датафрейме на NaN."""
    df = df.replace([np.inf, -np.inf], np.nan)
    return df
