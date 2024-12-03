from datetime import datetime

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Union
from src.tools.logger import logger


def create_new_features(df: pd.DataFrame) -> pd.DataFrame:
    # Добавление новых признаков
    df['total_delay'] = df[
        ['delay5', 'delay30', 'delay60', 'delay90', 'delay_more']].sum(axis=1)
    df['credit_utilization'] = df['arrear_amt_outstanding'] / df[
        'account_amt_credit_limit'].replace(0, np.nan)
    df['principal_interest_ratio'] = df['arrear_principal_outstanding'] / df[
        'arrear_int_outstanding'].replace(0, np.nan)

    # Заполнение NaN значений 0 для новых признаков
    df['total_delay'] = df['total_delay'].fillna(0)
    df['credit_utilization'] = df['credit_utilization'].fillna(0)
    df['principal_interest_ratio'] = df['principal_interest_ratio'].fillna(0)

    return df


def generate_statistics_features(
    data: pd.DataFrame,
    stats_dict: Dict[str, str | list]
) -> pd.DataFrame:
    """Генерит статистические фичи по словарю агрегаций."""
    logger.info("Генерация статистических фич...")
    stats_df = data.groupby('application_id').agg(stats_dict)
    stats_df.columns = ['_'.join(col).strip() for col in stats_df.columns]
    stats_df.reset_index(inplace=True)
    return stats_df


def generate_categorical_features(
    data: pd.DataFrame,
    categorical_features: List[str],
) -> pd.DataFrame:
    """Генерит категориальные фичи (мода)."""
    logger.info("Генерация категориальных фич...")
    cat_df = (
        data[['application_id'] + categorical_features]
        .groupby('application_id')
        .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else None)
    )
    cat_df.reset_index(inplace=True)
    return cat_df


def generate_last_reporting_features(data: pd.DataFrame) -> pd.DataFrame:
    """Генерит фичи для последней даты reporting_dt. """
    logger.info("Генерация фичей последней даты...")
    last_report_df = (
        data.sort_values(
            by=['application_id', 'trade_opened_dt'], ascending=[True, False]
        )
        .groupby('application_id')
        .first()
    )
    last_report_df.reset_index(inplace=True)
    return last_report_df


def generate_base_features(
    data: pd.DataFrame,
    stats_dict: Dict[str, List[str]],
    categorical_features: List[str]
) -> pd.DataFrame:
    """Запускает генерацию фичей и объединяет """
    data = create_new_features(data)
    stats_features = generate_statistics_features(data, stats_dict)
    cat_features = generate_categorical_features(data, categorical_features)
    last_features = generate_last_reporting_features(data)
    final_df = stats_features.merge(
        cat_features, on='application_id', how='left'
    )
    final_df = final_df.merge(last_features, on='application_id', how='left')
    logger.info("Пропуски final_df: %s", final_df.isna().sum())
    final_df = final_df.dropna(axis='columns', how='any')
    logger.info("Все фичи объединены.")
    logger.info("Размер final_df: %s", final_df.shape)

    return final_df


def diff_dates(data: pd.DataFrame, feature_date: datetime) -> pd.DataFrame:
    """Преобразует даты в числа"""
    feature_date = pd.to_datetime(feature_date)
    date_columns = [col for col in data.columns if
                    pd.api.types.is_datetime64_any_dtype(
                        data[col]
                        ) or 'date' in col.lower()]

    result = data[['application_id']].copy()

    for col in date_columns:
        data[col] = pd.to_datetime(data[col], errors='coerce')
        data[col] = data[col].fillna(feature_date)
        diff_col_name = f"{col}_diff"
        result[diff_col_name] = (feature_date - data[col]).dt.days
    return result.dropna(axis='columns')


def fill_missing_values(df: pd.DataFrame, how: Any):
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

    return df
