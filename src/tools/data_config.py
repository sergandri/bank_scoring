import os

import numpy as np

from datetime import datetime
from typing import Dict, List

TARGET_LOCAL_PATH = 'input_data/df_target_30k.csv'
BKI_LOCAL_PATH = 'input_data/df_BKI_30k.csv'
TEST_LOCAL_PATH = 'input_data/df_test_notarget_10k.csv'

FEATURE_DATE: datetime = datetime(2024, 12, 9)
TARGET_COLUMN: str = 'target'
OUTPUT_PATH: str = "output_data"
BINNING_FILE = os.path.join(OUTPUT_PATH, "all_binnings.pkl")


class FeatureThresholds:
    information_value: float = 0.01
    gini: float = 0.03
    gini_over_time: float = 0.0
    psi: float = 0.1
    corr: 0.7
    vif: 10


class SplitConfig:
    test_size: float = 0.2
    random_state: int = 666


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
    fillna_logic: dict = {
        'application_id': lambda x: x,  # пропусков быть не должно
        'client_id': lambda x: x,  # пропусков быть не должно
        'equifax_id': lambda x: x,  # пропусков быть не должно
        'reporting_dt': lambda x: x,
        # пропусков быть не должно
        'account_uid': lambda x: x,  # пропусков быть не должно
        'fund_date': lambda x: x,  # пропусков быть не должно
        'trade_owner_indic': lambda x: x.fillna(99),
        # Неопределенный тип участия
        'trade_opened_dt': lambda x: x.fillna('1970-01-01'),
        # Заполняем фиктивной датой
        'trade_trade_type_code': lambda x: x.fillna(99),  # Иной тип сделки
        'trade_loan_kind_code': lambda x: x.fillna(99),  # Иной вид займа
        'trade_acct_type1': lambda x: x.fillna(99),
        # Неопределенная цель займа
        'trade_is_consumer_loan': lambda x: x.fillna(0),
        # Предполагаем отсутствие потребительского кредита
        'trade_has_card': lambda x: x.fillna(0),
        # Предполагаем отсутствие карты
        'trade_is_novation': lambda x: x.fillna(0),
        # Предполагаем отсутствие новации
        'trade_is_money_source': lambda x: x.fillna(0),
        # Предполагаем отсутствие денежного обязательства
        'trade_close_dt': lambda x: x,
        # Оставляем пропуск
        'account_amt_credit_limit': lambda x: x.fillna(0),
        # Заполняем нулевым значением
        'account_amt_currency_code': lambda x: x.fillna('unknown'),
        # Заполняем фиктивным кодом валюты
        'account_amt_ensured_amt': lambda x: x.fillna(0),
        # Нулевая обеспеченная сумма
        'coborrower_has_solidary': lambda x: x.fillna(0),
        # Предполагаем отсутствие солидарных должников
        'coborrower_solidary_num': lambda x: x.fillna(0),
        # Нулевое количество солидарных должников
        'paymnt_condition_principal_terms_amt': lambda x: x.fillna(0),
        # Нулевая сумма платежа
        'paymnt_condition_principal_terms_amt_dt': lambda x: x.fillna(
            '1970-01-01'
        ),  # Фиктивная дата платежа
        'paymnt_condition_interest_terms_amt': lambda x: x.fillna(0),
        # Нулевая сумма процентов
        'paymnt_condition_interest_terms_amt_dt': lambda x: x.fillna(
            '1970-01-01'
        ),  # Фиктивная дата процентов
        'paymnt_condition_terms_frequency': lambda x: x.fillna(99),
        # Иная частота платежей
        'paymnt_condition_min_paymt': lambda x: x.fillna(0),
        # Нулевой минимальный платеж
        'paymnt_condition_grace_start_dt': lambda x: x.fillna('1970-01-01'),
        # Фиктивная дата начала льготного периода
        'paymnt_condition_grace_end_dt': lambda x: x.fillna('1970-01-01'),
        # Фиктивная дата окончания льготного периода
        'paymnt_condition_interest_payment_due_date': lambda x: x,
        # Фиктивная дата уплаты процентов
        'overall_val_credit_total_amt': lambda x: x.fillna(0),
        # Заполняем 0
        'overall_val_credit_total_monetary_amt': lambda x: x.fillna(0),
        # Заполняем 0
        'overall_val_credit_total_amt_date': lambda x: x,
        'month_aver_paymt_aver_paymt_amt': lambda x: x.fillna(x.median()),
        # Заполняем медианным значением
        'month_aver_paymt_calc_date': lambda x: x.fillna('1970-01-01'),
        # Фиктивная дата расчета
        'has_collaterals': lambda x: x.fillna(0),  # Отсутствие залога
        'has_guarantees': lambda x: x.fillna(0),  # Отсутствие поручительства
        'has_indie_guarantees': lambda x: x.fillna(0),
        # Отсутствие независимой гарантии
        'collat_insured_insur_sign': lambda x: x.fillna(0),
        # Отсутствие страхования
        'collat_insured_insur_limit': lambda x: x.fillna(0),
        # Нулевой лимит страховых выплат
        'collat_insured_currency_code': lambda x: x.fillna('unknown'),
        # Фиктивная валюта
        'collat_insured_has_franchise': lambda x: x.fillna(0),
        # Отсутствие франшизы
        'collat_insured_insur_start_dt': lambda x: x.fillna('1970-01-01'),
        # Фиктивная дата начала страхования
        'collat_insured_insur_end_dt': lambda x: x,
        # Фиктивная дата окончания страхования
        'collat_insured_insur_fact_end_dt': lambda x: x,
        # Фиктивная дата фактического окончания
        'collat_insured_insur_end_reason': lambda x: x.fillna(99),
        # Иная причина окончания страхования
        'colat_repaid': lambda x: x.fillna(0),
        # Отсутствие погашения за счет обеспечения
        'loan_indicator': lambda x: x,
        # Иное основание прекращения обязательства
        'loan_indicator_dt': lambda x: x,
        # оставляем пропуск
        'legal_items_has_legal_dispute': lambda x: x.fillna(0),
        # Отсутствие судебного акта
        'legal_items_court_act_dt': lambda x: x.fillna('1970-01-01'),
        # Фиктивная дата судебного акта
        'legal_items_court_act_effect_code': lambda x: x.fillna(0),
        # Акт не вступил в законную силу
        'attr_value': lambda x: x.fillna('-'),  # Отсутствие данных
        'past_due_principal_missed_date': lambda x: x,
        'past_due_int_missed_date': lambda x: x,
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
        # как буду агрегировать
        'account_uid': ['count'],
        'fund_date': ['min', 'max', lambda x: (x.max() - x.min()).days],
        'trade_opened_dt': ['min', 'max', lambda x: (x.max() - x.min()).days],
        'trade_close_dt': ['min', 'max', lambda x: (x.max() - x.min()).days],
        'collat_insured_has_franchise': ['mean', 'sum', 'max', 'first'],
        'account_amt_credit_limit': ['mean', 'max', 'first'],
        'account_amt_ensured_amt': ['mean', 'max', 'first'],
        'paymnt_condition_principal_terms_amt': ['mean', 'max', 'sum',
                                                 'first'],
        'paymnt_condition_interest_terms_amt': ['sum', 'mean', 'max', 'first'],
        'month_aver_paymt_aver_paymt_amt': ['sum', 'mean', 'max', 'first'],
        'trade_is_consumer_loan': ['first'],
        'trade_owner_indic': ['first'],
        'trade_trade_type_code': ['first'],
        'trade_loan_kind_code': ['first'],
        'trade_has_card': ['sum', 'max', 'first'],
        'has_collaterals': ['sum', 'max', 'first'],
        'has_guarantees': ['sum', 'max', 'first'],
        'arrear_sign': ['sum', 'max', 'first'],
        'arrear_amt_outstanding': ['sum', 'max', 'first'],
        'arrear_principal_outstanding': ['sum', 'mean', 'max', 'first'],
        'arrear_int_outstanding': ['sum', 'mean', 'max', 'first'],
        'arrear_other_amt_outstanding': ['sum', 'mean', 'max', 'first'],
        'credit_utilization': ['sum', 'mean', 'max', 'first'],
        'principal_interest_ratio': ['sum', 'mean', 'max', 'first'],
        'delay5': ['sum', 'mean', 'first'],
        'delay30': ['sum', 'mean', 'first'],
        'delay60': ['sum', 'mean', 'first'],
        'delay90': ['sum', 'mean', 'first'],
        'delay_more': ['sum', 'mean', 'first'],
        'total_delay': ['sum', 'mean', 'first'],
        'cred_max_overdue': ['max', 'mean', 'sum', 'first'],
        'attr_value': ['nunique'],
        'AV_max_delay_level': ['max', 'mean', 'sum', 'first'],
        'AV_total_delays': ['max', 'mean', 'sum', 'first'],
        'AV_delays_1_5_days': ['max', 'mean', 'sum', 'first'],
        'AV_delays_6_29_days': ['max', 'mean', 'sum', 'first'],
        'AV_delays_30_59_days': ['max', 'mean', 'sum', 'first'],
        'AV_delays_over_240_days': ['max', 'mean', 'sum', 'first'],
        'AV_on_time_periods': ['max', 'mean', 'sum', 'first'],
        'AV_credit_utilization_total_delay': ['max', 'mean', 'sum', 'first'],
        'AV_max_delay_arrear_outstanding': ['max', 'mean', 'sum', 'first'],
        'AV_arrear_over_total_credit': ['max', 'mean', 'sum', 'first'],
        'AV_principal_over_credit_limit': ['max', 'mean', 'sum', 'first'],
        'AV_bankruptcy_events': ['max', 'mean', 'sum', 'first'],
        'AV_debt_sold_events': ['max', 'mean', 'sum', 'first'],
        'AV_risk_category': ['max', 'mean', 'first'],
        'AV_overdue_ratio': ['mean', 'sum', 'max', 'first'],
        'AV_on_time_ratio': ['mean', 'max', 'first'],
        'AV_max_on_time_length': ['mean', 'max', 'first'],
        'AV_max_overdue_length': ['mean', 'max', 'first'],
        'AV_first_overdue_position': ['min', 'mean', 'first'],
        'AV_unique_segments': ['mean', 'max', 'sum'],
        'AV_avg_distance_overdues': ['mean', 'max', 'first'],
        'AV_cumulative_risk_score': ['mean', 'max', 'first'],
        'AV_on_time_segment_count': ['mean', 'max', 'sum'],
        'AV_high_risk_segments': ['mean', 'sum', 'first'],
        'AV_mixed_risk_segments': ['mean', 'sum', 'first'],
        'D_past_due_principal_missed_binary': ['mean', 'sum', 'max', 'first'],
        'D_past_due_int_missed_binary': ['mean', 'sum', 'max', 'first'],
        'D_past_due_any_missed_binary': ['mean', 'sum', 'max', 'first'],
        'D_loan_indicator_dt_binary': ['mean', 'sum', 'max', 'first'],
        'D_trade_close_dt_binary': ['mean', 'sum', 'max', 'first'],
        'D_past_due_sum_missed': ['mean', 'sum', 'max', 'first'],
    }
    priority_map: Dict[str, int] = {
        '-': 0,  # Нет данных
        '0': 1,  # Без просрочки
        '1': 2,  # 1-5 дней
        '2': 3,  # 6-29 дней
        '3': 4,  # 0-59 дней
        '4': 5,  # 60-89 дней
        '5': 6,  # 90-119 дней
        '6': 7,  # 120-149 дней
        '7': 8,  # 150-179 дней
        '8': 9,  # 180-209 дней
        '9': 10,  # 210-239 дней
        'A': 11,  # ≥ 240 дней
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
