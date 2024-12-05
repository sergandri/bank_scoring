import os

import numpy as np

from datetime import datetime
from typing import Dict, List

TARGET_LOCAL_PATH = 'eda/df_target_30k.csv'
BKI_LOCAL_PATH = 'eda/df_BKI_30k.csv'

FEATURE_DATE: datetime = datetime(2024, 11, 30)
TARGET_COLUMN: str = 'target'
BINNING_PATH = "binning_files"
BINNING_FILE = os.path.join(BINNING_PATH, "all_binnings.pkl")


class FeatureThresholds:
    information_value: float = 0.01
    gini: float = 0.03
    gini_over_time: float = 0.0
    psi: float = 0.1
    corr: 0.7
    vif: 10


class SplitConfig:
    test_size: float = 0.2
    random_state: int = 777


class PreprocessConfig:
    dtype_map: Dict[str, np.dtype] = {
        'application_id': 'int64',
        'client_id': 'int64',
        'equifax_id': 'int64',
        'reporting_dt': 'datetime64[ns]',
        'account_uid': 'int64',
        'fund_date': 'datetime64[ns]',
        'trade_owner_indic': 'float64',
        'trade_opened_dt': 'datetime64[ns]',
        'trade_trade_type_code': 'float64',
        'trade_loan_kind_code': 'float64',
        'trade_acct_type1': 'float64',
        'trade_is_consumer_loan': 'float64',
        'trade_has_card': 'float64',
        'trade_is_novation': 'float64',
        'trade_is_money_source': 'float64',
        'trade_close_dt': 'datetime64[ns]',
        'account_amt_credit_limit': 'float64',
        'account_amt_currency_code': 'str',
        'account_amt_ensured_amt': 'float64',
        'coborrower_has_solidary': 'float64',
        'coborrower_solidary_num': 'float64',
        'paymnt_condition_principal_terms_amt': 'float64',
        'paymnt_condition_principal_terms_amt_dt': 'datetime64[ns]',
        'paymnt_condition_interest_terms_amt': 'float64',
        'paymnt_condition_interest_terms_amt_dt': 'datetime64[ns]',
        'paymnt_condition_terms_frequency': 'float64',
        'paymnt_condition_min_paymt': 'float64',
        'paymnt_condition_grace_start_dt': 'datetime64[ns]',
        'paymnt_condition_grace_end_dt': 'datetime64[ns]',
        'paymnt_condition_interest_payment_due_date': 'datetime64[ns]',
        'overall_val_credit_total_amt': 'float64',
        'overall_val_credit_total_monetary_amt': 'float64',
        'overall_val_credit_total_amt_date': 'datetime64[ns]',
        'month_aver_paymt_aver_paymt_amt': 'float64',
        'month_aver_paymt_calc_date': 'datetime64[ns]',
        'has_collaterals': 'float64',
        'has_guarantees': 'float64',
        'has_indie_guarantees': 'float64',
        'collat_insured_insur_sign': 'float64',
        'collat_insured_insur_limit': 'float64',
        'collat_insured_currency_code': 'str',
        'collat_insured_has_franchise': 'float64',
        'collat_insured_insur_start_dt': 'datetime64[ns]',
        'collat_insured_insur_end_dt': 'datetime64[ns]',
        'collat_insured_insur_fact_end_dt': 'datetime64[ns]',
        'collat_insured_insur_end_reason': 'float64',
        'colat_repaid': 'float64',
        'collat_repay_code': 'float64',
        'collat_repay_dt': 'datetime64[ns]',
        'collat_repay_amt': 'float64',
        'loan_indicator': 'float64',
        'loan_indicator_dt': 'datetime64[ns]',
        'legal_items_has_legal_dispute': 'float64',
        'legal_items_court_act_dt': 'datetime64[ns]',
        'legal_items_court_act_effect_code': 'float64',
        'hold_code': 'float64',
        'hold_dt': 'datetime64[ns]',
        'file_since_dt': 'datetime64[ns]',
        'last_updated_dt': 'datetime64[ns]',
        'last_uploaded_dt': 'datetime64[ns]',
        'arrear_sign': 'float64',
        'arrear_start_amt_outstanding': 'float64',
        'arrear_last_payment_due_code': 'float64',
        'arrear_amt_outstanding': 'float64',
        'arrear_principal_outstanding': 'float64',
        'arrear_int_outstanding': 'float64',
        'arrear_other_amt_outstanding': 'float64',
        'arrear_calc_date': 'datetime64[ns]',
        'arrear_unconfirm_grace': 'float64',
        'due_arrear_start_dt': 'datetime64[ns]',
        'due_arrear_last_payment_due_code': 'float64',
        'due_arrear_amt_outstanding': 'float64',
        'due_arrear_principal_outstanding': 'float64',
        'due_arrear_int_outstanding': 'float64',
        'due_arrear_other_amtoutstanding': 'float64',
        'due_arrear_calc_date': 'datetime64[ns]',
        'past_due_dt': 'datetime64[ns]',
        'past_due_last_payment_due_code': 'float64',
        'past_due_amt_past_due': 'float64',
        'past_due_principal_amt_past_due': 'float64',
        'past_due_int_amt_past_due': 'float64',
        'past_due_other_amt_past_due': 'float64',
        'past_due_calc_date': 'datetime64[ns]',
        'past_due_principal_missed_date': 'datetime64[ns]',
        'past_due_int_missed_date': 'datetime64[ns]',
        'delay5': 'int64',
        'delay30': 'int64',
        'delay60': 'int64',
        'delay90': 'int64',
        'delay_more': 'int64',
        'cred_max_overdue': 'float64',
        'attr_value': 'object'
    }


class FeatureConfig:
    categorical_features: List[str] = [
        'trade_owner_indic',
        'trade_trade_type_code',
        'trade_loan_kind_code',
        'attr_value',
        'arrear_last_payment_due_code',
    ]
    agg_dict: Dict[str, str | list] = {
        # Простая статистика по счетам
        'account_uid': ['count'],  # Количество
        'fund_date': ['min', 'max', lambda x: (x.max() - x.min()).days],
        'trade_opened_dt': ['min', 'max', lambda x: (x.max() - x.min()).days],
        'trade_close_dt': ['min', 'max', lambda x: (x.max() - x.min()).days],
        # Диапазон закрытия сделок
        'collat_insured_has_franchise': ['mean', 'sum', 'max', 'first'],
        'collat_insured_insur_limit': ['mean', 'sum', 'max', 'first'],
        # insured
        'account_amt_credit_limit': ['mean', 'max', 'first'],
        # Лимиты по кредитам
        'account_amt_ensured_amt': ['mean', 'max', 'first'],
        # Сумма обеспечений
        'paymnt_condition_principal_terms_amt': ['mean', 'max', 'sum',
                                                 'first'],
        # Основной долг
        'paymnt_condition_interest_terms_amt': ['sum', 'mean', 'max', 'first'],
        # Процентная задолженность
        'month_aver_paymt_aver_paymt_amt': ['sum', 'mean', 'max', 'first'],
        # Среднемесячные платежи
        'trade_is_consumer_loan': ['first'],
        # Количество потребительских кредитов
        'trade_owner_indic': ['first'],
        'trade_trade_type_code': ['first'],
        'trade_loan_kind_code': ['first'],
        'trade_has_card': ['sum', 'max', 'first'],
        # Количество кредитов с картами
        'has_collaterals': ['sum', 'max', 'first'],
        # Количество кредитов с залогами
        'has_guarantees': ['sum', 'max', 'first'],
        # Количество кредитов с поручительством

        # Задолженности и просрочки
        'arrear_sign': ['sum', 'max', 'first'],
        # Количество кредитов с задолженностью
        'arrear_amt_outstanding': ['sum', 'max', 'first'],
        # Суммарная задолженность
        'arrear_principal_outstanding': ['sum', 'mean', 'max', 'first'],
        # Основная задолженность
        'arrear_int_outstanding': ['sum', 'mean', 'max', 'first'],
        # Процентная задолженность
        'arrear_other_amt_outstanding': ['sum', 'mean', 'max', 'first'],
        # Иные
        'credit_utilization': ['sum', 'mean', 'max', 'first'],
        'principal_interest_ratio': ['sum', 'mean', 'max', 'first'],

        # История просрочек
        'delay5': ['sum', 'mean', 'first'],  # Количество просрочек до 6 дней
        'delay30': ['sum', 'mean', 'first'],
        # Количество просрочек от 6 до 30 дней
        'delay60': ['sum', 'mean', 'first'],
        # Количество просрочек от 31 до 60 дней
        'delay90': ['sum', 'mean', 'first'],
        # Количество просрочек от 61 до 90 дней
        'delay_more': ['sum', 'mean', 'first'],
        # Количество просрочек более 90 дней
        'total_delay': ['sum', 'mean', 'first'],
        # Финальные показатели
        'cred_max_overdue': ['max', 'mean', 'sum', 'first'],
        # Максимальная сумма просроченной задолженности
        'attr_value': ['nunique'],
        # Количество уникальных значений + самый частый статус
        'max_delay_level': ['max', 'mean', 'sum', 'first'],
        'total_delays': ['max', 'mean', 'sum', 'first'],
        'delays_1_5_days': ['max', 'mean', 'sum', 'first'],
        'delays_6_29_days': ['max', 'mean', 'sum', 'first'],
        'delays_30_59_days': ['max', 'mean', 'sum', 'first'],
        'delays_over_240_days': ['max', 'mean', 'sum', 'first'],
        'on_time_periods': ['max', 'mean', 'sum', 'first'],
        # Новые признаки из платежной строки
        'credit_utilization_total_delay': ['max', 'mean', 'sum', 'first'],
        'max_delay_arrear_outstanding': ['max', 'mean', 'sum', 'first'],
        'arrear_over_total_credit': ['max', 'mean', 'sum', 'first'],
        'principal_over_credit_limit': ['max', 'mean', 'sum', 'first'],
        'bankruptcy_events': ['max', 'mean', 'sum', 'first'],
        'debt_sold_events': ['max', 'mean', 'sum', 'first'],
        'risk_category': ['max', 'mean', 'sum', 'first'],
    }
    priority_map: Dict[str, int] = {
        '-': 0,  # Нет данных
        '0': 1,  # Без просрочки
        '1': 2,  # Просрочка 1-5 дней
        '2': 3,  # Просрочка 6-29 дней
        '3': 4,  # Просрочка 30-59 дней
        '4': 5,  # Просрочка 60-89 дней
        '5': 6,  # Просрочка 90-119 дней
        '6': 7,  # Просрочка 120-149 дней
        '7': 8,  # Просрочка 150-179 дней
        '8': 9,  # Просрочка 180-209 дней
        '9': 10,  # Просрочка 210-239 дней
        'A': 11,  # Просрочка ≥ 240 дней
        'B': 12,  # Безнадежный долг
        'C': 0,  # Договор закрыт
        'S': 0,  # Договор продан
        'R': 0,  # Договор рефинансирован
        'W': 13,  # Договор продан коллекторам
        'U': 0,  # Договор расторгнут
        'T': 14,  # Субъект КИ признан банкротом
        'I': 0,  # Прекращена передача информации
    }


feature_config = FeatureConfig()
preprocess_config = PreprocessConfig()
split_config = SplitConfig()
feature_thresholds = FeatureThresholds()
