from typing import Optional
import pandas as pd
import numpy as np
from optbinning import OptimalBinning
import joblib

from src.tools.data_config import BINNING_FILE, OUTPUT_PATH
from src.tools.logger import logger


class Binner:
    def __init__(
        self,
        min_bin_size: float = 0.05,
        max_n_bins: int = 6,
        outlier_detector: str = 'range',
        class_weight: str = 'balanced',
        monotonic_trend: str = 'auto_asc_desc'
    ):
        self.min_bin_size = min_bin_size
        self.max_n_bins = max_n_bins
        self.outlier_detector = outlier_detector
        self.class_weight = class_weight
        self.monotonic_trend = monotonic_trend
        self.binning_dict = {}
        self.features = None
        self.train_data = None
        self.train_woe = None
        self.test_woe = None

    def fit_binning(
        self,
        train_data: pd.DataFrame,
        target_col: str,
    ):
        """
        Обучает процесс биннинга на тренировочных данных в
        цикле для каждой фичи.
        """
        self.train_data = train_data
        self.features = [col for col in self.train_data.columns if
                         col != target_col]

        X = self.train_data[self.features]
        y = self.train_data[target_col]

        for feature in self.features:
            try:
                opt_binning = OptimalBinning(
                    name=feature,
                    dtype="numerical",
                    min_bin_size=self.min_bin_size,
                    max_n_bins=self.max_n_bins,
                    outlier_detector=self.outlier_detector,
                    class_weight=self.class_weight,
                    monotonic_trend=self.monotonic_trend
                )
                opt_binning.fit(X[feature], y)
                self.binning_dict[feature] = opt_binning
            except Exception as e:
                print(f"Feature '{feature}' skipped due to error: {e}")
                continue

    def transform(
        self,
        df: pd.DataFrame,
        target_col: Optional[str] = None,
        metric: str = 'woe',
    ):
        """
        Применяет WOE-трансформацию.
        """

        test_woe = df[self.features]
        for feature, binning in self.binning_dict.items():
            if feature in test_woe.columns:
                test_woe[feature] = binning.transform(
                    df[feature],
                    metric=metric,
                )

        if target_col in df.columns:
            test_woe[target_col] = df[target_col]

        return test_woe

    def save(self, path: str = BINNING_FILE):
        """Сохраняет биннинг-процесс в указанный путь."""
        joblib.dump(self.binning_dict, path)
        logger.info(f"Binning process saved to {path}")

    def load(self, path: str = BINNING_FILE):
        """Загружает сохраненный биннинг-процесс."""
        self.binning_dict = joblib.load(path)
        logger.info(f"Binning process loaded from {path}")
        self.features = joblib.load(f"{OUTPUT_PATH}/lr_features.pkl")
