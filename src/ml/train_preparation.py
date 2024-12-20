from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from src.tools.data_config import SplitConfig, TARGET_COLUMN
from src.tools.logger import logger


def merge_dates(
    df_features: pd.DataFrame,
    df_dates: pd.DataFrame,
) -> pd.DataFrame:
    df_merged = df_features.merge(df_dates, how='left', on=['application_id'])
    df_merged.set_index('application_id', inplace=True)
    return df_merged


def merge_features_target(
    df_target: pd.DataFrame,
    df_features: pd.DataFrame,
    df_dates: pd.DataFrame,
):
    logger.info("Merging features and target...")
    """Подтягивает таргет, удаляет клиента, устанавливает индекс"""
    df_merged = df_target.merge(df_features, how='left', on=['application_id'])
    df_merged = df_merged.merge(df_dates, how='left', on=['application_id'])
    #df_merged.drop(columns=['client_id'], inplace=True)
    df_merged.set_index('application_id', inplace=True)
    return df_merged

def remove_non_numeric_features(df):
    """Удаляет из DataFrame все нечисловые признаки."""
    logger.info("Removing unconverted features...")
    numeric_df = df.select_dtypes(include=['number'])
    return numeric_df


def t_t_split(
    df: pd.DataFrame,
    split_config: SplitConfig,
    target_col: str = TARGET_COLUMN
) -> tuple:
    if target_col not in df.columns:
        raise ValueError(
            f"No target '{target_col}'"
        )

    train_data, test_data = train_test_split(
        df,
        test_size=split_config.test_size,
        random_state=split_config.random_state,
        stratify=df[target_col]
    )

    # def x_y_split(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    #     return data.drop(columns=[target_column]), data[target_column]
    #
    # x_train, y_train = x_y_split(train_data)
    # x_test, y_test = x_y_split(test_data)

    logger.info(f"Train-Test split completed.")
    logger.info(f"Random State: {split_config.random_state}")
    logger.info(f"Training set size: {len(train_data)} rows")
    logger.info(f"Test set size: {len(test_data)} rows")

    return train_data, test_data
