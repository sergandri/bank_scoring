from abc import ABC, abstractmethod
from typing import Any, List, Optional
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix, \
    ConfusionMatrixDisplay, roc_curve
from catboost import CatBoostClassifier, Pool
from sklearn.linear_model import LogisticRegression
import optuna
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from src.tools.data_config import OUTPUT_PATH
from src.tools.logger import logger


class BaseModelBuilder(ABC):
    def __init__(
        self,
        train_data: Optional[pd.DataFrame],
        test_data: pd.DataFrame,
        target_col: str,
        model_name: str,
    ):
        if target_col in train_data.columns:
            self.X = train_data.drop(columns=[target_col])
            self.y = train_data[target_col]
            (self.train_features,
             self.val_features,
             self.train_target,
             self.val_target) = train_test_split(
                self.X,
                self.y,
                test_size=0.2,
                random_state=666,
                stratify=self.y
            )
        else:
            self.X = train_data
            self.y = None
            self.train_features = None
            self.val_features = None
            self.train_features = None
            self.val_target = None

        if target_col in test_data.columns:
            self.X_test = test_data.drop(columns=[target_col])
            self.y_test = test_data[target_col]
        else:
            self.X_test = test_data
            self.y_test = None

        self.model = None
        self.best_params = None
        self.model_name = model_name

    @abstractmethod
    def optimize_model(self):
        """Оптимизация гиперпараметров модели."""
        pass

    @abstractmethod
    def train_final_model(self):
        """Обучение финальной модели."""
        pass

    @abstractmethod
    def save_model(self, path: str):
        """Сохранение модели."""
        pass

    @abstractmethod
    def save_train_predictions(
        self, train_path: str, test_path: str, x_train_path: str,
        x_test_path: str
    ):
        """Сохранение предсказаний."""
        pass

    @abstractmethod
    def load_model(self):
        """Загрузка модели из файла"""
        pass

    @abstractmethod
    def save_real_predictions(self):
        """Прогноз на неразмеченном тесте"""

    def evaluate_model(self):
        """Оценка качества модели на тестовой выборке."""
        predictions_proba = self.model.predict_proba(self.X_test)[:, 1]
        test_auc = roc_auc_score(self.y_test, predictions_proba)
        gini = 2 * test_auc - 1
        logger.info(f"Gini на тестовой выборке: {gini:.4f}")

        y_pred = self.model.predict(self.X_test)
        cm = confusion_matrix(self.y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.title(f"Confusion Matrix - {self.model_name}")
        plt.show()

        fpr, tpr, _ = roc_curve(self.y_test, predictions_proba)
        plt.plot(
            fpr, tpr, label=f'ROC Curve (AUC = {test_auc:.4f})'
        )
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {self.model_name}')
        plt.legend()
        plt.show()
        return gini


class LRModelBuilder(BaseModelBuilder):
    def __init__(self, train_data, test_data, target_col="target"):
        super().__init__(
            train_data,
            test_data,
            target_col,
            model_name="Logistic Regression",
        )

    def optimize_model(self):
        def objective(trial):
            params = {
                'solver': trial.suggest_categorical(
                    'solver', ['lbfgs', 'liblinear']
                ),
                'C': trial.suggest_float('C', 0.01, 10.0, log=True),
                'max_iter': trial.suggest_int('max_iter', 100, 1000),
                'random_state': 42,
                'class_weight': 'balanced',
            }
            model = LogisticRegression(**params)
            model.fit(self.train_features, self.train_target)
            predictions_proba = model.predict_proba(self.val_features)[:, 1]
            val_auc = roc_auc_score(self.val_target, predictions_proba)
            val_gini = 2 * val_auc - 1
            return val_gini

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=50)
        self.best_params = study.best_trial.params
        print("Лучшие параметры:", self.best_params)

    def train_final_model(self):
        self.model = LogisticRegression(**self.best_params)
        self.model.fit(self.X, self.y)

    def save_model(self, path: str = f"{OUTPUT_PATH}/final_model.pkl"):
        joblib.dump(self.model, path)
        logger.info(f"Logistic Regression model saved {path}")

    def save_train_predictions(
        self,
        train_path: str = f"{OUTPUT_PATH}/lr_train_result.csv",
        test_path: str = f"{OUTPUT_PATH}/lr_test_result.csv",
        x_train_path: str = f"{OUTPUT_PATH}/logreg_x_train_result.csv",
        x_test_path: str = f"{OUTPUT_PATH}/logreg_x_test_result.csv"
    ):
        train_predictions = self.model.predict_proba(self.train_features)[:, 1]
        train_results = pd.DataFrame(
            {
                'Actual': self.train_target,
                'Predicted Probability': train_predictions
            }
        )
        train_results.to_csv(train_path, index=False)

        test_predictions = self.model.predict_proba(self.X_test)[:, 1]
        test_results = pd.DataFrame(
            {
                'Actual': self.y_test,
                'Predicted Probability': test_predictions
            }
        )
        test_results.to_csv(test_path, index=False)

    def load_model(self, path: str = f"{OUTPUT_PATH}/final_model.pkl"):
        self.model = joblib.load(path)
        logger.info(f"Model loaded from {path}")

    def save_real_predictions(
        self,
        real_test_path: str = f"{OUTPUT_PATH}/real_lr_test_result.csv",
        real_x_test_path: str = f"{OUTPUT_PATH}/real_lr_x_train_result.csv",
    ):
        self.X_test.to_csv(real_x_test_path, index=False)
        test_predictions = self.model.predict_proba(self.X_test)[:, 1]
        test_results = pd.DataFrame(
            {
                'application_id': self.X_test.index,
                'Predicted Probability': test_predictions
            }
        )
        test_results.to_csv(real_test_path, index=False)


class CBModelBuilder(BaseModelBuilder):
    def __init__(
        self,
        train_data,
        test_data,
        target_col="target",
        cat_features=None,
    ):
        super().__init__(
            train_data,
            test_data,
            target_col,
            model_name="CatBoost",
        )
        self.cat_features = cat_features if cat_features else []

    def optimize_model(self):
        def objective(trial):
            params = {
                'iterations': trial.suggest_int('iterations', 500, 2000),
                'depth': trial.suggest_int('depth', 4, 10),
                'learning_rate': trial.suggest_float(
                    'learning_rate', 0.01, 0.3, log=True
                ),
                'l2_leaf_reg': trial.suggest_float(
                    'l2_leaf_reg', 1e-3, 10.0, log=True
                ),
                'random_state': 777,
                'eval_metric': 'AUC',
                'loss_function': 'Logloss',
                'early_stopping_rounds': 50,
                'verbose': 200,
            }
            train_pool = Pool(
                self.train_features, self.train_target,
                cat_features=self.cat_features
            )
            val_pool = Pool(
                self.val_features, self.val_target,
                cat_features=self.cat_features
            )
            model = CatBoostClassifier(**params)
            model.fit(train_pool, eval_set=val_pool, use_best_model=True)
            predictions_proba = model.predict_proba(self.val_features)[:, 1]
            val_auc = roc_auc_score(self.val_target, predictions_proba)
            val_gini = 2 * val_auc - 1
            return val_gini

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=5)
        self.best_params = study.best_trial.params
        print("Best params:", self.best_params)

    def train_final_model(self):
        self.model = CatBoostClassifier(**self.best_params)
        self.X.drop(columns=['client_id'], inplace=True)
        train_pool = Pool(self.X, self.y, cat_features=self.cat_features)
        self.model.fit(train_pool)

        feature_importances = self.model.get_feature_importance()
        feature_names = self.X.columns
        sorted_idx = np.argsort(feature_importances)[::-1]
        top_35_idx = sorted_idx[:35]
        top_35_features = feature_names[top_35_idx]
        top_35_feature_names = list(top_35_features)

        train_pool = Pool(
            self.X[top_35_feature_names],
            self.y,
            cat_features=self.cat_features,
        )
        self.model = CatBoostClassifier(**self.best_params)
        self.model.fit(train_pool)
        joblib.dump(
            top_35_feature_names,
            f"{OUTPUT_PATH}/top_35_feature_names.pkl",
        )

    def save_model(self, path: str = f"{OUTPUT_PATH}/cb_model.cbm"):
        self.model.save_model(path)
        logger.info(f"CatBoost model saved to {path}")

        # Вывод фич и их важности
        feature_importances = self.model.get_feature_importance()
        feature_names = self.X.columns
        sorted_idx = np.argsort(feature_importances)[::-1]
        logger.info("Feature Importances (Top Features):")
        for idx in sorted_idx:
            logger.info(
                f"Feature: {feature_names[idx]}, Importance: {feature_importances[idx]}"
            )

    def save_train_predictions(
        self,
        train_path=f"{OUTPUT_PATH}/catboost_train_result.csv",
        test_path=f"{OUTPUT_PATH}/catboost_test_result.csv",
        x_train_path=f"{OUTPUT_PATH}/catboost_x_train_result.csv",
        x_test_path=f"{OUTPUT_PATH}/catboost_x_test_result.csv"
    ):
        train_predictions = self.model.predict_proba(self.train_features)[:, 1]
        train_results = pd.DataFrame(
            {
                'Actual': self.train_target,
                'Predicted Probability': train_predictions
            }
        )
        train_results.to_csv(train_path, index=False)

        test_predictions = self.model.predict_proba(self.X_test)[:, 1]
        test_results = pd.DataFrame(
            {
                'Actual': self.y_test,
                'Predicted Probability': test_predictions
            }
        )
        test_results.to_csv(test_path, index=False)

    def plot_feature_importance(self):
        feature_importances = self.model.get_feature_importance()
        feature_names = self.train_features.columns
        sorted_idx = np.argsort(feature_importances)[::-1]
        sorted_importances = feature_importances[sorted_idx]
        sorted_feature_names = [feature_names[i] for i in sorted_idx]

        plt.figure(figsize=(10, 6))
        plt.barh(sorted_feature_names, sorted_importances, color='skyblue')
        plt.xlabel('Feature Importance')
        plt.title('CatBoost Feature Importance')
        plt.gca().invert_yaxis()
        plt.show()

    def plot_roc_curve(self):
        """Строит ROC-кривую для оценки качества модели."""
        test_predictions = self.model.predict_proba(self.X_test)[:, 1]
        fpr, tpr, _ = roc_curve(self.y_test, test_predictions)
        plt.figure(figsize=(10, 6))
        plt.plot(fpr, tpr, label='CatBoost ROC Curve')
        plt.plot([0, 1], [0, 1], linestyle='--', color='red')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.show()

    def load_model(self, path: str = f"{OUTPUT_PATH}/cb_model.cbm"):
        """Загружает модель из указанного файла."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found at {path}")
        self.model = CatBoostClassifier()
        self.model.load_model(path)
        logger.info(f"Model loaded from {path}")

    def save_real_predictions(
        self,
        real_test_path: str = f"{OUTPUT_PATH}/real_cb_test_result.csv",
        real_x_test_path: str = f"{OUTPUT_PATH}/real_cb_x_test_result.csv",
    ):
        """Сохраняет предсказания только на тестовых данных."""
        if self.model is None:
            raise ValueError(
                "Model is not trained or loaded. "
            )
        top_35_feature_names = joblib.load(
            f"{OUTPUT_PATH}/top_35_feature_names.pkl"
        )
        self.X_test[top_35_feature_names].to_csv(real_x_test_path, index=False)
        test_predictions = self.model.predict_proba(
            self.X_test[top_35_feature_names]
        )[:, 1]
        test_results = pd.DataFrame(
            {
                'application_id': self.X_test.index,
                'Predicted Probability': test_predictions
            }
        )
        test_results.to_csv(real_test_path, index=False)
        logger.info(f"Test predictions saved to {real_test_path}")
