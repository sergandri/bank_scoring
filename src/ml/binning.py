from typing import Optional

import pandas as pd
from optbinning import OptimalBinning
from src.tools.data_config import BINNING_FILE, TARGET_COLUMN
import joblib
import numpy as np

from src.tools.logger import logger


# Функция для подготовки и биннинга фичей
def woe_binning(
    train_data: pd.DataFrame,
    features: list,
    target_column: str,
    min_bin_size: float = 0.05,
    max_n_bins: int = 5,
):
    """
    Делает WOE-биннинг для указанных фичей с использованием OptBinning.
    :param train_data: pd.DataFrame, тренировочные данные
    :param features: list, список фич для биннинга
    :param target_column: str, имя колонки таргета
    :param min_bin_size: float, минимальный размер бина
    :param max_n_bins: float, максимальной количество бинов
    :return: dict, словарь с биннерами для каждой переменной
    """
    binning_process_dict = {}
    logger.info('Фит биннинга...')
    for feature in features:
        opt_binning = OptimalBinning(
            name=feature,
            dtype="numerical",
            min_bin_size=min_bin_size,
            max_n_bins=max_n_bins,
        )
        opt_binning.fit(train_data[feature], train_data[target_column])
        binning_process_dict[feature] = opt_binning

    return binning_process_dict


def transform_woe(df: pd.DataFrame, binning_dict: dict):
    """
    Применяет WOE-биннинг к данным.
    :param df: pd.DataFrame, данные для трансформации
    :param binning_dict: dict, словарь с биннерами
    :return: pd.DataFrame, преобразованный датафрейм
    """
    logger.info('Применение биннинга...')
    df_transformed = df.copy()

    for feature, binning in binning_dict.items():
        df_transformed[feature] = binning.transform(df[feature], metric="woe")

    return df_transformed


# Сохранение биннеров для дальнейшего использования
def save_binning_process(
    binning_dict: dict,
    path: str = BINNING_FILE
):
    """
    Сохраняет биннинг процесс в pickle файл.
    :param binning_dict: dict, словарь с биннерами
    :param path: str, путь к файлу для сохранения
    """
    logger.info('Сохранение биннинга...')
    joblib.dump(binning_dict, path)


def load_all_binning_processes(file_path: str = BINNING_FILE):
    """Загружает биннинг"""
    logger.info(f"Загрузка биннинга из {file_path}...")
    binning_dict = joblib.load(file_path)
    return binning_dict


def perform_train_binning(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    target_column: str
):
    # dtypes = [np.float64, np.int64]
    numeric_cols = train_data.select_dtypes(
        include=[np.number]
    ).columns.tolist()
    features = [col for col in train_data.columns if
                col != target_column and train_data[col].dtype in numeric_cols]

    # Вычисляем WOE-биннинг для указанных фич
    binning_dict = woe_binning(train_data, features, target_column)

    # Применяем WOE-трансформацию к тренировочным и тестовым данным
    train_woe = transform_woe(train_data, binning_dict)
    test_woe = transform_woe(test_data, binning_dict)

    # Сохранение биннинга для дальнейшего использования
    save_binning_process(binning_dict, BINNING_FILE)

    return train_woe, test_woe
