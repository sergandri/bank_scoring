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
        try:
            opt_binning = OptimalBinning(
                name=feature,
                dtype="numerical",
                min_bin_size=min_bin_size,
                max_n_bins=max_n_bins,
            )
            opt_binning.fit(train_data[feature], train_data[target_column])
            binning_process_dict[feature] = opt_binning
        except Exception as e:
            logger.warning(f"Ошибка при биннинге признака '{feature}': {e}")
            continue
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
    logger.info("Типы данных train_data:")
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
                f"Признак '{col}' является константным и будет исключен из биннинга."
                )
            continue
        # Проверяем на малое количество уникальных значений
        if unique_values < 5:
            logger.info(
                f"Признак '{col}' имеет менее 5 уникальных значений и будет исключен из биннинга."
                )
            continue
        # Проверяем на пропущенные значения
        missing_ratio = train_data[col].isnull().mean()
        if missing_ratio > 0.5:
            logger.info(
                f"Признак '{col}' имеет более 50% пропущенных значений и будет исключен из биннинга."
                )
            continue
        # Проверяем на бесконечные значения
        if np.isinf(train_data[col]).any():
            logger.info(
                f"Признак '{col}' содержит бесконечные значения и будет исключен из биннинга."
                )
            continue
        suitable_features.append(col)

    logger.info("Признаки, подходящие для биннинга: %s", suitable_features)

    if not suitable_features:
        logger.error(
            "Нет признаков, подходящих для биннинга. Проверьте данные."
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
        "Столбцы train_data после биннинга: %s", train_woe.columns.tolist()
    )
    logger.info("Размер train_data после биннинга: %s", train_woe.shape)
    logger.info(
        "Столбцы test_data после биннинга: %s", test_woe.columns.tolist()
    )
    logger.info("Размер test_data после биннинга: %s", test_woe.shape)

    # Сохранение биннинга для дальнейшего использования
    save_binning_process(binning_dict, BINNING_FILE)

    return train_woe, test_woe
