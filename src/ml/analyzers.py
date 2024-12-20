from typing import List

import joblib
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.feature_selection import RFE, RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from statsmodels.stats.outliers_influence import variance_inflation_factor
from src.tools.data_config import FeatureThresholds, OUTPUT_PATH
from src.tools.logger import logger


class FeatureAnalyzer:
    def __init__(
        self,
        train_woe: pd.DataFrame,
        test_woe: pd.DataFrame,
        feature_thresholds: FeatureThresholds,
        target_col: str = "target",
        df_dates: pd.DataFrame = None
    ):
        logger.info(f"Feature analyser initialized")
        self.train_woe = train_woe.drop(columns=[target_col])
        self.test_woe = test_woe.drop(columns=[target_col])
        self.test_target = test_woe[target_col]
        self.train_target = train_woe[target_col]
        self.df_dates = df_dates
        self.analysis_results = {
            "IV": {},
            "Gini Train": {},
            "Gini Test": {},
            "PSI": {},
            "Gini Over Time": {}
        }
        self.selected_features = []
        self.feature_thresholds = feature_thresholds

    @staticmethod
    def calculate_iv(feature: pd.Series, target: pd.Series) -> float:
        """Рассчитывает IV для заданной фичи."""
        df = pd.DataFrame({'feature': feature, 'target': target})
        total_good = len(df[df['target'] == 0])
        total_bad = len(df[df['target'] == 1])

        iv = 0
        for bin_value in df['feature'].unique():
            bin_data = df[df['feature'] == bin_value]
            good = len(bin_data[bin_data['target'] == 0]) / total_good
            bad = len(bin_data[bin_data['target'] == 1]) / total_bad
            woe = np.log((good + 0.0001) / (bad + 0.0001))
            iv += (good - bad) * woe

        return iv

    @staticmethod
    def calculate_gini(
        feature: pd.Series,
        target: pd.Series,
        woe: bool = False
    ) -> float:
        """Считает Gini."""
        if woe:
            feature = - feature # тут надо вытащить ивент рейт и подставить его
        auc = roc_auc_score(target, feature)
        gini = 2 * auc - 1
        return gini

    @staticmethod
    def calculate_psi(
        expected: pd.Series,
        actual: pd.Series,
        bins: int = 10,
    ) -> float:
        """
        Считает PSI между двумя выборками.
        :param expected: pd.Series, якорная выборка (первая неделя)
        :param actual: pd.Series, текущая выборка
        :param bins: int, количество бинов для расчета
        :return: float, ПСИ
        """
        expected_hist, bin_edges = np.histogram(expected, bins=bins)
        actual_hist, _ = np.histogram(actual, bins=bin_edges)

        expected_ratio = expected_hist / len(expected)
        actual_ratio = actual_hist / len(actual)

        psi_value = np.sum(
            (expected_ratio - actual_ratio) * np.log(
                (expected_ratio + 0.0001) / (actual_ratio + 0.0001)
            )
        )
        return psi_value

    def run_analysis(self):
        """
        Выполняет анализ IV, Gini и PSI для всех фичей
        и сохраняет результаты в analysis_results.
        """
        combined_data = pd.concat([self.train_woe, self.test_woe])
        combined_target = pd.concat([self.train_target, self.test_target])

        if self.df_dates is not None:
            combined_data = combined_data.join(
                self.df_dates,
                on=combined_data.index,
            )
            combined_data['reporting_dt'] = pd.to_datetime(
                combined_data['reporting_dt']
            )
            combined_data = combined_data.sort_values(by='reporting_dt')
            unique_weeks = (combined_data['reporting_dt'].dt.to_period('W')
                            .unique())
        else:
            raise ValueError(
                "df_dates must be provided "
                "for Gini Over Time and PSI calculations"
            )

        for feature in self.train_woe.columns:
            iv = self.calculate_iv(self.train_woe[feature], self.train_target)
            self.analysis_results["IV"][feature] = iv

            gini_train = self.calculate_gini(
                self.train_woe[feature],
                self.train_target,
                woe=True,
            )
            self.analysis_results["Gini Train"][feature] = gini_train

            gini_test = self.calculate_gini(
                self.test_woe[feature],
                self.test_target,
                woe=True,
            )
            self.analysis_results["Gini Test"][feature] = gini_test

            gini_over_time = {}
            psi_values = {}

            first_week_data = combined_data[
                combined_data['reporting_dt'].dt
                .to_period('W') == unique_weeks[0]
                ][feature]

            for week in unique_weeks:
                week_data = combined_data[
                    combined_data['reporting_dt'].dt
                    .to_period('W') == week]
                if not week_data.empty:
                    gini_week = self.calculate_gini(
                        week_data[feature],
                        combined_target.loc[week_data.index],
                        woe=True,
                    )
                    gini_over_time[str(week)] = gini_week

                    psi_value = self.calculate_psi(
                        first_week_data,
                        week_data[feature],
                    )
                    psi_values[str(week)] = psi_value

            self.analysis_results["Gini Over Time"][feature] = gini_over_time
            self.analysis_results["PSI"][feature] = psi_values

    def log_results(self):
        """Логирует результаты типа анализа. (очень большой лог получился)"""
        for metric, results in self.analysis_results.items():
            logger.info(f"\n{metric} Results:")
            for feature, value in results.items():
                if isinstance(value, dict):
                    logger.info(f"Feature: {feature}, {metric}: {value}")
                else:
                    logger.info(f"Feature: {feature}, {metric}: {value:.4f}")

    def save_results(self, path: str = 'output_data/factor_analysis.csv'):
        """Сэйвит результаты анализа"""
        results_df = pd.DataFrame(self.analysis_results)
        results_df.to_csv(path, index_label='Feature')

    def filter_features(self):
        """
        Выполняет отбор фичей в модель согласно ограничениям
        из конфигурации
        """

        def get_min(data):
            return min([float(value) for value in data.values()]) if (
                    isinstance(data, dict) and data) else 0

        for feature in self.train_woe.columns:
            iv = self.analysis_results["IV"].get(feature, 0)
            gini_train = self.analysis_results["Gini Train"].get(feature, 0)
            psi = get_min(self.analysis_results["PSI"].get(feature, {}))
            gini_over_time = get_min(
                self.analysis_results["Gini Over Time"].get(feature, {})
            )

            if (iv > self.feature_thresholds.information_value and
                    gini_train > self.feature_thresholds.gini and
                    psi <= self.feature_thresholds.psi and
                    gini_over_time > self.feature_thresholds.gini_over_time):
                self.selected_features.append(feature)
                logger.info(
                    f'feature_analyzer_selected_features '
                    f'{self.selected_features}'
                )


class FeatureCalculator:
    def __init__(self, analyser: FeatureAnalyzer):
        logger.info(f"Feature selector initialized")
        self.analyser = analyser
        self.train_data = analyser.train_woe[analyser.selected_features]
        self.selected_features_corr = []
        self.selected_features_vif = []
        self.selected_features = []
        logger.info(
            f'feature_selector_in_selected_features'
            f'{analyser.train_woe[analyser.selected_features].columns}'
        )

    def calculate_correlation(self, threshold: float = 0.70):
        """Рассчитывает корреляцию между признаками и фильтрует по порогу."""
        corr_matrix = self.train_data.corr().abs()
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )

        to_drop = set()

        for column in upper_triangle.columns:
            if column in to_drop:
                continue
            correlated_features = upper_triangle.index[
                upper_triangle[column] > threshold
                ].tolist()
            for feature in correlated_features:
                # Сравниваем IV и удаляем признак с меньшим значением
                iv_column = self.analyser.analysis_results["IV"].get(column, 0)
                iv_feature = self.analyser.analysis_results["IV"].get(
                    feature, 0
                )
                if iv_column >= iv_feature:
                    to_drop.add(feature)
                else:
                    to_drop.add(column)
                    break

        self.selected_features_corr = [col for col in corr_matrix.columns
                                       if col not in to_drop]

        corr_results_df = corr_matrix.loc[
            self.selected_features_corr,
            self.selected_features_corr,
        ]
        corr_results_df.to_csv(
            f"{OUTPUT_PATH}/correlation_filtered_features.csv",
            index_label='Feature',
        )

    def calculate_vif(self):
        """Рассчитывает и фильтрует фичи с VIF по порогу"""
        selected_data = self.train_data[self.selected_features_corr]
        vif_data = pd.DataFrame()
        vif_data["Feature"] = selected_data.columns
        vif_data["VIF"] = [variance_inflation_factor(selected_data.values, i)
                           for i in range(selected_data.shape[1])]

        self.selected_features_vif = vif_data[
            vif_data["VIF"] < 10]["Feature"].tolist()

        vif_data.to_csv(
            f"{OUTPUT_PATH}/vif_filtered_features.csv",
            index=False
        )

    def calculate_total(self):
        """Сегодня мы с тобой фильтруем"""
        self.selected_features = list(
            set(self.selected_features_corr).intersection(
                set(self.selected_features_vif)
            )
        )

    def plot_gini_by_features(self):
        """Строит графики Gini"""
        features = self.selected_features_vif[:10]
        train_ginis = []
        test_ginis = []

        for i in range(1, len(features) + 1):
            selected_subset = features[:i]
            train_gini = self.analyser.calculate_gini(
                self.analyser.train_woe[selected_subset].sum(axis=1),
                self.analyser.train_target,
                woe=True,
            )
            test_gini = self.analyser.calculate_gini(
                self.analyser.test_woe[selected_subset].sum(axis=1),
                self.analyser.test_target,
                woe=True,
            )
            train_ginis.append(train_gini)
            test_ginis.append(test_gini)

        plt.figure(figsize=(10, 6))
        plt.plot(
            range(
                1,
                len(features) + 1
            ),
            train_ginis,
            marker='o',
            label='Train Gini',
        )
        plt.plot(
            range(
                1,
                len(features) + 1
            ),
            test_ginis,
            marker='o',
            label='Test Gini',
        )
        plt.xlabel('Number of Features')
        plt.ylabel('Gini')
        plt.title('Gini Score by Number of Features')
        plt.legend()


class FeatureSelector:
    def __init__(self, feature_analyser, feature_calculator):
        self.feature_calculator: FeatureCalculator = feature_calculator
        self.feature_analyzer: FeatureAnalyzer = feature_analyser
        self.top_features: List[str] | None = None

    def select_best_features(self):
        """Выбирает лучшее из лучшего"""
        binning_results = pd.DataFrame(self.feature_analyzer.analysis_results)
        binning_results.reset_index(inplace=True)
        binning_results['indicator'] = binning_results['IV'] * 0.7 + \
                                       binning_results['Gini Train'] * 0.15 + \
                                       binning_results['Gini Test'] * 0.15
        feature_filter = self.feature_calculator.selected_features
        top_list = (binning_results.query('index in @feature_filter')
                    .sort_values('indicator', ascending=False).head(10))[
            'index'].tolist()
        top_list.append('target')
        logger.info(f'Top features: {top_list}')
        self.top_features = top_list

    def advanced_feature_selection(self, n_features=10):
        """Отбор ровно 10 наиболее важных признаков"""
        x = self.feature_calculator.train_data[
            self.feature_calculator.selected_features
        ]
        y = self.feature_analyzer.train_target
        logger.info("Начало отбора признаков с использованием RFE.")

        model = LogisticRegression(
            solver='liblinear',
            random_state=666,
            class_weight='balanced'
        )

        rfe = RFE(
            estimator=model,
            n_features_to_select=n_features,
            step=1
        )

        rfe.fit(x, y)

        selected_features = x.columns[rfe.support_].tolist()
        logger.info(f"RFE selected features: {selected_features}")

        self.top_features = selected_features
        joblib.dump(
            self.top_features,
            f"{OUTPUT_PATH}/lr_features.pkl",
        )
        self.top_features.append('target')

    def manual_correction_features(self):
        """Корректирует руками на всякий случай"""
        features_to_remove: list = []
        features_to_add: list = []
        for i in features_to_remove:
            self.top_features.remove(i)

        for i in features_to_add:
            self.top_features.append(i)
