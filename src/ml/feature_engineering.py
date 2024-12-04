from datetime import datetime

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Union
from src.tools.logger import logger
from src.tools.data_config import feature_config


def max_priority_value(attr_str: str) -> int:
    return max(
        [feature_config.priority_map[char] for char in attr_str if char in
         feature_config.priority_map]
    )


def count_delays(attr_str: str, delay_char: str) -> int:
    return attr_str.count(delay_char)


def risk_category(row):
    if row['max_delay_level'] >= 5 and row['credit_utilization'] > 0.8:
        return 2
    elif row['max_delay_level'] >= 3:
        return 1
    else:
        return 0


def create_new_features(df: pd.DataFrame) -> pd.DataFrame:
    # Добавление новых признаков
    logger.info("Creating custom features...")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df['bankruptcy_events'] = df['attr_value'].apply(lambda x: x.count('T'))
    df['debt_sold_events'] = df['attr_value'].apply(lambda x: x.count('W'))
    df['max_delay_level'] = df['attr_value'].apply(max_priority_value)
    df['total_delays'] = df['attr_value'].apply(
        lambda x: sum([count_delays(x, char) for char in '123456789A'])
    )
    df['delays_1_5_days'] = df['attr_value'].apply(
        lambda x: count_delays(x, '1')
    )
    df['delays_6_29_days'] = df['attr_value'].apply(
        lambda x: count_delays(x, '2')
    )
    df['delays_30_59_days'] = df['attr_value'].apply(
        lambda x: count_delays(x, '3')
    )
    df['delays_over_240_days'] = df['attr_value'].apply(
        lambda x: count_delays(x, 'A')
    )
    df['on_time_periods'] = df['attr_value'].apply(
        lambda x: count_delays(x, '0')
    )
    df['total_delay'] = df[
        ['delay5', 'delay30', 'delay60', 'delay90', 'delay_more']].sum(axis=1)
    df['credit_utilization'] = df['arrear_amt_outstanding'] / df[
        'account_amt_credit_limit'].replace(0, np.nan)
    df['risk_category'] = df.apply(risk_category, axis=1)
    df['principal_interest_ratio'] = df['arrear_principal_outstanding'] / df[
        'arrear_int_outstanding'].replace(0, np.nan)

    # Заполнение NaN значений 0 для новых признаков
    df['total_delay'] = df['total_delay'].fillna(0)
    df['credit_utilization'] = df['credit_utilization'].fillna(0)
    df['principal_interest_ratio'] = df['principal_interest_ratio'].fillna(0)
    df['credit_utilization_total_delay'] = df['credit_utilization'] * df[
        'total_delay']
    df['max_delay_arrear_outstanding'] = df['max_delay_level'] * df[
        'arrear_amt_outstanding']
    df['arrear_over_total_credit'] = df['arrear_amt_outstanding'] / df[
        'overall_val_credit_total_amt'].replace(0, np.nan)
    df['principal_over_credit_limit'] = (
            df['paymnt_condition_principal_terms_amt'] / df
    ['account_amt_credit_limit'].replace(0, np.nan)
    )
    return df


def generate_statistics_features(
    data: pd.DataFrame,
    stats_dict: Dict[str, str | list]
) -> pd.DataFrame:
    """Генерит статистические фичи по словарю агрегаций."""
    logger.info("Generating stat features...")
    stats_df = data.groupby('application_id').agg(stats_dict)
    stats_df.columns = ['_'.join(col).strip() for col in stats_df.columns]
    stats_df.reset_index(inplace=True)
    return stats_df


def generate_categorical_features(
    data: pd.DataFrame,
    categorical_features: List[str],
) -> pd.DataFrame:
    """Генерит категориальные фичи (мода)."""
    logger.info("Generating cat features...")
    cat_df = (
        data[['application_id'] + categorical_features]
        .groupby('application_id')
        .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else None)
    )
    cat_df.reset_index(inplace=True)
    return cat_df


def generate_last_reporting_features(data: pd.DataFrame) -> pd.DataFrame:
    """Генерит фичи для последней даты reporting_dt. """
    logger.info("Generating last date features...")
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
    logger.info("Merging features...")
    logger.info("Size final_df: %s", final_df.shape)

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
