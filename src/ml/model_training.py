import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, make_scorer, confusion_matrix, \
    ConfusionMatrixDisplay, roc_curve
import optuna
import joblib
import matplotlib.pyplot as plt


class FinalModelBuilder:
    def __init__(
        self, train_data: pd.DataFrame, test_data: pd.DataFrame,
        target_col: str = "target"
    ):
        self.train_data = train_data.drop(columns=[target_col])
        self.test_data = test_data.drop(columns=[target_col])
        self.train_target = train_data[target_col]
        self.test_target = test_data[target_col]

        self.model = None
        self.best_params = None

    def optimize_model(self):
        """
        Оптимизация гипер-параметров с использованием Optuna.
        """

        def objective(trial):
            param_grid = {
                'C': trial.suggest_float('C', 1e-4, 1e2, log=True),
                'penalty': trial.suggest_categorical('penalty', ['l2']),
                'solver': trial.suggest_categorical(
                    'solver', ['lbfgs', 'liblinear',
                               'saga']
                )
            }
            model = LogisticRegression(**param_grid, max_iter=1000)
            model.fit(self.train_data, self.train_target)
            predictions = model.predict_proba(self.test_data)[:, 1]
            gini = 2 * roc_auc_score(self.test_target, predictions) - 1
            return gini

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=50)
        self.best_params = study.best_params

    def train_final_model(self):
        """
        Тренировка модели с оптимизированными гипер-параметрами.
        """
        self.model = LogisticRegression(**self.best_params, max_iter=1000)
        self.model.fit(self.train_data, self.train_target)

    def evaluate_model(self):
        """
        Оценка качества модели на тестовой выборке.
        """
        predictions = self.model.predict_proba(self.test_data)[:, 1]
        gini = 2 * roc_auc_score(self.test_target, predictions) - 1
        print(f"Gini на тестовой выборке: {gini:.4f}")

        # Расчет и вывод confusion matrix
        y_pred = self.model.predict(self.test_data)
        cm = confusion_matrix(self.test_target, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.title("Confusion Matrix")
        plt.show()

        # Построение графика ROC-кривой
        fpr, tpr, _ = roc_curve(self.test_target, predictions)
        plt.plot(fpr, tpr, label='ROC Curve')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.show()

        return gini

    def save_model(self, path: str):
        """
        Сохраняет модель в указанный путь.
        :param path: str, путь для сохранения файла
        """
        joblib.dump(self.model, path)
        print(f"Модель сохранена в {path}")

    def save_predictions(
        self, train_path: str = 'train_result.csv',
        test_path: str = 'test_result.csv',
    ):
        """
        Сохраняет предсказания для тренировочной и тестовой выборок.
        :param train_path: str, путь для сохранения предсказаний на тренировочной выборке
        :param test_path: str, путь для сохранения предсказаний на тестовой выборке
        """
        train_predictions = self.model.predict_proba(self.train_data)[:, 1]
        test_predictions = self.model.predict_proba(self.test_data)[:, 1]

        train_results = pd.DataFrame({
            'Actual': self.train_target,
            'Predicted Probability': train_predictions
        })
        test_results = pd.DataFrame({
            'Actual': self.test_target,
            'Predicted Probability': test_predictions
        })

        train_results.to_csv(train_path, index=False)
        test_results.to_csv(test_path, index=False)
