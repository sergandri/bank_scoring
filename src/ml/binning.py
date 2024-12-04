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
    logger.info('Fitting binning process...')
    for feature in features:
        try:
            opt_binning = OptimalBinning(
                name=feature,
                dtype="numerical",
                min_bin_size=min_bin_size,
                max_n_bins=max_n_bins,
                outlier_detector='range',
                class_weight='balanced',
            )
            opt_binning.fit(train_data[feature], train_data[target_column])
            binning_process_dict[feature] = opt_binning
        except Exception as e:
            logger.warning(f"Binning feature error '{feature}': {e}")
            continue
    return binning_process_dict


def transform_woe(df: pd.DataFrame, binning_dict: dict):
    """
    Применяет WOE-биннинг к данным.
    :param df: pd.DataFrame, данные для трансформации
    :param binning_dict: dict, словарь с биннерами
    :return: pd.DataFrame, преобразованный датафрейм
    """
    logger.info('Binning transforming...')
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
    logger.info('Saving binning process...')
    joblib.dump(binning_dict, path)


def load_all_binning_processes(file_path: str = BINNING_FILE):
    """Загружает биннинг"""
    logger.info(f"Loading binning from {file_path}...")
    binning_dict = joblib.load(file_path)
    return binning_dict


def perform_train_binning(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    target_column: str
):
    logger.info("Data types train_data:")
    logger.info("%s", train_data.dtypes)

    # Выбираем числовые признаки, исключая целевую переменную
    numeric_cols = train_data.select_dtypes(
        include=[np.number]
    ).columns.tolist()
    features = [col for col in numeric_cols if col != target_column]

    # Фильтруем неподходящие признаки
    suitable_features = []
    for col in features:
        # Проверяем на константность
        unique_values = train_data[col].nunique(dropna=True)
        if unique_values <= 1:
            logger.info(
                f"Feature '{col}' constant - removed "
            )
            continue
        # Проверяем на малое количество уникальных значений
        if unique_values < 3:
            logger.info(
                f"Feature '{col}' has less than 3 values - removed"
            )
            continue
        # Проверяем на пропущенные значения
        missing_ratio = train_data[col].isnull().mean()
        if missing_ratio > 0.5:
            logger.info(
                f"Feature '{col}' has more than 50% missed values - removed"
            )
            continue
        # Проверяем на бесконечные значения
        if np.isinf(train_data[col]).any():
            logger.info(
                f"Feature '{col}' has inf values - removed"
            )
            continue
        suitable_features.append(col)

    logger.info("Suitable for binning: %s", suitable_features)

    if not suitable_features:
        logger.error(
            "No suitable features for binning"
        )
        # Возвращаем исходные данные без изменений
        return train_data, test_data

    # Вычисляем WOE-биннинг для указанных фич
    binning_dict = woe_binning(train_data, suitable_features, target_column)

    # Применяем WOE-трансформацию к тренировочным и тестовым данным
    train_woe = transform_woe(train_data, binning_dict)
    test_woe = transform_woe(test_data, binning_dict)

    # Логирование результатов
    logger.info(
        "Columns train_data after binning: %s", train_woe.columns.tolist()
    )
    logger.info("Size train_data after binning: %s", train_woe.shape)
    logger.info(
        "Columns test_data after binning: %s", test_woe.columns.tolist()
    )
    logger.info("Size test_data after binning: %s", test_woe.shape)

    # Сохранение биннинга для дальнейшего использования
    save_binning_process(binning_dict, BINNING_FILE)

    return train_woe, test_woe
