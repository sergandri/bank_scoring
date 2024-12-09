import enum
import re
from datetime import datetime

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Literal, Union
from src.tools.logger import logger
from src.tools.data_config import feature_config


def count_char(attr_value, char_set):
    """Подсчитывает количество символов из указанного набора в строке."""
    return sum(1 for c in attr_value if c in char_set)


def max_segment_length(char_set, attr_value):
    pattern = f"[{char_set}]+"
    segments = re.findall(pattern, attr_value)
    return max(len(segment) for segment in segments) if segments else 0


def find_first_occurrence(attr_value, char_set):
    """Находит первую позицию символа из указанного набора."""
    for idx, char in enumerate(attr_value):
        if char in char_set:
            return idx + 1
    return -1


def calculate_overdue_ratio(attr_value):
    """Рассчитывает долю просрочек в строке."""
    overdue_chars = "123456789ABC"
    return count_char(attr_value, overdue_chars) / len(attr_value)


def calculate_on_time_ratio(attr_value):
    """Рассчитывает долю состояний 'без просрочки'."""
    return count_char(attr_value, "0") / len(attr_value)


def max_priority_value(attr_str: str) -> int:
    return max(
        [feature_config.priority_map[char] for char in attr_str if char in
         feature_config.priority_map]
    )


def count_delays(attr_str: str, delay_char: str) -> int:
    return attr_str.count(delay_char)


def risk_category(row):
    if row['AV_max_delay_level'] >= 5 and row['credit_utilization'] > 0.8:
        return 2
    elif row['AV_max_delay_level'] >= 3:
        return 1
    else:
        return 0


def count_unique_segments(attr_value: str, char_set: str) -> int:
    pattern = f"[{char_set}]+"
    segments = re.findall(pattern, attr_value)
    return len(set(segments))

def avg_distance_between_overdues(attr_value: str, char_set: str) -> float:
    positions = [i for i, c in enumerate(attr_value) if c in char_set]
    if len(positions) < 2:
        return 0
    return np.mean(np.diff(positions))

def cumulative_risk_score(attr_value: str, priority_map:
dict=feature_config.priority_map) -> float:
    return sum(priority_map.get(char, 0) * (i + 1) for i, char in enumerate(attr_value))



def create_new_features(df: pd.DataFrame) -> pd.DataFrame:
    # Добавление новых признаков
    logger.info("Creating custom features...")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df['AV_overdue_ratio'] = df['attr_value'].apply(calculate_overdue_ratio)
    df['AV_on_time_ratio'] = df['attr_value'].apply(calculate_on_time_ratio)
    df['AV_max_on_time_length'] = df['attr_value'].apply(
        lambda x: max_segment_length("0", x)
    )
    df['AV_max_overdue_length'] = df['attr_value'].apply(
        lambda x: max_segment_length("123456789ABC", x)
    )
    df['AV_first_overdue_position'] = df['attr_value'].apply(
        lambda x: find_first_occurrence(x, "123456789ABC")
    )
    df['AV_bankruptcy_events'] = df['attr_value'].apply(lambda x: x.count('T'))
    df['AV_debt_sold_events'] = df['attr_value'].apply(lambda x: x.count('W'))
    df['AV_max_delay_level'] = df['attr_value'].apply(max_priority_value)
    df['AV_total_delays'] = df['attr_value'].apply(
        lambda x: sum([count_delays(x, char) for char in '123456789A'])
    )
    df['AV_delays_1_5_days'] = df['attr_value'].apply(
        lambda x: count_delays(x, '1')
    )
    df['AV_delays_6_29_days'] = df['attr_value'].apply(
        lambda x: count_delays(x, '2')
    )
    df['AV_delays_30_59_days'] = df['attr_value'].apply(
        lambda x: count_delays(x, '3')
    )
    df['AV_delays_over_240_days'] = df['attr_value'].apply(
        lambda x: count_delays(x, 'A')
    )
    df['AV_on_time_periods'] = df['attr_value'].apply(
        lambda x: count_delays(x, '0')
    )
    df['total_delay'] = df[
        ['delay5', 'delay30', 'delay60', 'delay90', 'delay_more']].sum(axis=1)
    df['credit_utilization'] = (df['arrear_amt_outstanding'] / df[
        'account_amt_credit_limit']).fillna(0)
    df['AV_risk_category'] = df.apply(risk_category, axis=1)
    df['principal_interest_ratio'] = (df['arrear_principal_outstanding'] / df[
        'arrear_int_outstanding']).fillna(0)
    df['AV_credit_utilization_total_delay'] = df['credit_utilization'] * df[
        'total_delay']
    df['AV_max_delay_arrear_outstanding'] = df['AV_max_delay_level'] * df[
        'arrear_amt_outstanding']
    df['AV_arrear_over_total_credit'] = (df['arrear_amt_outstanding'] / df[
        'overall_val_credit_total_amt']).fillna(0)
    df['AV_principal_over_credit_limit'] = (
            df['paymnt_condition_principal_terms_amt'] / df
    ['account_amt_credit_limit']
    ).fillna(0)
    df['AV_unique_segments'] = df['attr_value'].apply(
        lambda x: count_unique_segments(x, "123456789ABC0")
    )
    df['AV_avg_distance_overdues'] = df['attr_value'].apply(
        lambda x: avg_distance_between_overdues(x, "123456789ABC")
    )
    df['AV_cumulative_risk_score'] = df['attr_value'].apply(
        lambda x: cumulative_risk_score(x)
    )
    df['AV_on_time_segment_count'] = df['attr_value'].apply(
        lambda x: count_unique_segments(x, "0")
    )
    df['AV_high_risk_segments'] = df['attr_value'].apply(
        lambda x: count_unique_segments(x, "B")
    )
    df['AV_mixed_risk_segments'] = df['attr_value'].apply(
        lambda x: count_unique_segments(x, "9A")
    )
    # Бинарные признаки для отдельных колонок
    df['D_past_due_principal_missed_binary'] = df[
        'past_due_principal_missed_date'].notnull().astype(int)
    df['D_past_due_int_missed_binary'] = df[
        'past_due_int_missed_date'].notnull().astype(int)
    df['D_loan_indicator_dt_binary'] = df[
        'loan_indicator_dt'].notnull().astype(int)
    df['D_trade_close_dt_binary'] = df[
        'trade_close_dt'].notnull().astype(int)
    # Бинарный признак, объединяющий обе колонки
    df['D_past_due_any_missed_binary'] = (
            df['D_past_due_principal_missed_binary'] | df[
        'D_past_due_int_missed_binary']
    ).astype(int)
    # Признак, объединяющий четыре колонки
    df['D_past_due_sum_missed'] = df['D_trade_close_dt_binary'] + df[
        'D_loan_indicator_dt_binary'] + df['D_past_due_any_missed_binary']

    return df


def generate_statistics_features(
    data: pd.DataFrame,
    stats_dict: Dict[str, str | list]
) -> pd.DataFrame:
    """Генерит статистические фичи по словарю агрегаций."""
    logger.info("Creating aggr features...")
    stats_df = data.groupby('application_id').agg(stats_dict)
    stats_df.columns = ['_'.join(col).strip() for col in stats_df.columns]
    stats_df.reset_index(inplace=True)
    return stats_df


def generate_categorical_features(
    data: pd.DataFrame,
    categorical_features: List[str],
) -> pd.DataFrame:
    """Генерит категориальные фичи (мода)."""
    logger.info("Creating cat features...")
    cat_df = (
        data[['application_id'] + categorical_features]
        .groupby('application_id')
        .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else None)
    )
    cat_df.reset_index(inplace=True)
    return cat_df


def generate_last_reporting_features(data: pd.DataFrame) -> pd.DataFrame:
    """Генерит фичи для последней даты reporting_dt. """
    logger.info("Merging last date features...")
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
    logger.info("Missings final_df: %s", final_df.isna().sum())
    # final_df = final_df.dropna(axis='columns', how='any')
    logger.info("Merging all features...")
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
