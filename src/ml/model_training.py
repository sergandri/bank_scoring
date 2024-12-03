import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, confusion_matrix, \
    ConfusionMatrixDisplay, roc_curve
import optuna
import joblib
import matplotlib.pyplot as plt

from src.tools.logger import logger


class FinalModelBuilder:
    def __init__(
        self,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        target_col: str = "target",
    ):
        logger.info(
            "Столбцы train_data перед разделением: %s",
            train_data.columns.tolist()
        )
        # Разделяем признаки и целевую переменную
        self.X = train_data.drop(columns=[target_col])
        self.y = train_data[target_col]
        self.X_test = test_data.drop(columns=[target_col])
        self.y_test = test_data[target_col]

        # Разделяем данные на обучающую и валидационную выборки
        self.train_features, self.val_features, self.train_target, self.val_target = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )

        self.model = None
        self.best_params = None
        self.random_state = 777

        logger.info(
            "Размер self.train_features: %s", self.train_features.shape
        )
        logger.info("Размер self.train_target: %s", self.train_target.shape)
        logger.info("Количество пропущенных значений в self.train_features:")
        logger.info("train nulls %s", self.train_features.isnull().sum())
        logger.info("Типы данных в self.train_features:")
        logger.info("train types %s", self.train_features.dtypes)

    def optimize_model(self):
        """
        Оптимизация гиперпараметров с использованием Optuna без кросс-валидации.
        """

        def objective(trial):
            if self.train_features.empty or self.train_target.empty:
                raise ValueError(
                    "Трейн пуст - чекнуть предварительную обработку данных."
                )

            solver = trial.suggest_categorical(
                'solver', ['lbfgs', 'liblinear', 'saga']
            )
            C = trial.suggest_float('C', 0.01, 10.0, log=True)
            max_iter = trial.suggest_int('max_iter', 100, 1000)

            model_params = {
                'solver': solver,
                'C': C,
                'max_iter': max_iter,
                'random_state': self.random_state,
                'class_weight': 'balanced'
            }

            model = LogisticRegression(**model_params)
            model.fit(self.train_features, self.train_target)
            predictions_proba = model.predict_proba(self.val_features)
            val_auc = roc_auc_score(self.val_target, predictions_proba[:, 1])
            val_gini = 2 * val_auc - 1
            return val_gini

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=50)
        self.best_params = study.best_trial.params
        print("Лучшие параметры:", self.best_params)

    def train_final_model(self):
        """
        Трейн с оптимизированными гп
        """

        model_params = self.best_params.copy()
        model_params['random_state'] = self.random_state

        self.model = LogisticRegression(**model_params)
        self.model.fit(self.X, self.y)

    def evaluate_model(self):
        """
        Оценка качества модели на тестовой выборке.
        """
        predictions_proba = self.model.predict_proba(self.X_test)
        test_auc = roc_auc_score(self.y_test, predictions_proba[:, 1])
        gini = 2 * test_auc - 1
        print(f"Gini на тестовой выборке: {gini:.4f}")

        y_pred = self.model.predict(self.X_test)
        cm = confusion_matrix(self.y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.title("Confusion Matrix")
        plt.show()

        fpr, tpr, _ = roc_curve(
            np.array(self.y_test),
            np.array(predictions_proba[:, 1])
        )
        plt.plot(
            fpr,
            tpr,
            label='ROC Curve (AUC = {:.4f})'.format(test_auc)
        )
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.show()

        return gini

    def save_model(self, path: str):
        """Сохраняет модель в указанный путь."""
        joblib.dump({'model': self.model}, path)
        print(f"Модель сохранена в {path}")

    def save_predictions(
        self,
        train_path: str = 'train_result.csv',
        test_path: str = 'test_result.csv',
        x_train_path: str = 'x_train_result.csv',
        x_test_path: str = 'x_test_result.csv',
    ):
        """Сохраняет предсказания для обучающей и тестовой выборок."""
        X_full_train = pd.concat(
            [self.train_features, self.val_features], axis=0
        )
        y_full_train = pd.concat([self.train_target, self.val_target], axis=0)
        train_predictions = self.model.predict_proba(X_full_train)[:, 1]

        train_results = pd.DataFrame(
            {
                'Actual': np.array(y_full_train),
                'Predicted Probability': np.array(train_predictions)
            }
        )

        test_predictions = self.model.predict_proba(self.X_test)[:, 1]
        test_results = pd.DataFrame(
            {
                'Actual': np.array(self.y_test),
                'Predicted Probability': np.array(test_predictions)
            }
        )

        train_results.to_csv(train_path, index=False)
        test_results.to_csv(test_path, index=False)
        X_full_train.to_csv(x_train_path, index=False)
        self.X_test.to_csv(x_test_path, index=False)
