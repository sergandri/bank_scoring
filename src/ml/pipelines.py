import pandas as pd

from src.ml.preprocessing import (
    read_dataframes, convert_data_types,
    drop_features
)
from src.ml.feature_engineering import generate_base_features
from src.tools.data_config import feature_config, preprocess_config
from src.ml.feature_engineering import diff_dates, fill_missing_values
from src.tools.data_config import FEATURE_DATE
from src.ml.train_preparation import (
    merge_features_target,
    remove_non_numeric_features
)
from src.ml.train_preparation import t_t_split
from src.tools.data_config import TARGET_COLUMN, split_config
from src.ml.binning import perform_train_binning
from src.ml.train_preparation import t_t_split
from src.tools.data_config import TARGET_COLUMN, split_config
from src.ml.binning import perform_train_binning
from src.tools.analyzers import (
    FeatureAnalyzer,
    FeatureCalculator,
    FeatureSelector,
)
from src.tools.data_config import feature_thresholds
from src.ml.model_training import FinalModelBuilder
from src.tools.logger import logger


def full_preprocessing():
    df_target, df_bki = read_dataframes()

    df_aggr = (df_bki.pipe(convert_data_types, preprocess_config)
    .pipe(drop_features)
    .pipe(
        generate_base_features,
        feature_config.agg_dict,
        feature_config.categorical_features,
    )
    )
    df_f = (df_aggr.pipe(
        merge_features_target,
        df_target,
        diff_dates(df_aggr, feature_date=FEATURE_DATE)
    )
            .pipe(remove_non_numeric_features)
            .pipe(fill_missing_values, how=0)
            )
    return df_f


def training(df_aggr_w_target_n: pd.DataFrame):
    df_target, df_bki = read_dataframes()
    train_data, test_data = t_t_split(
        df=df_aggr_w_target_n,
        split_config=split_config,
        target_column=TARGET_COLUMN
    )
    train_woe, test_woe = perform_train_binning(
        train_data, test_data, target_column=TARGET_COLUMN
    )
    df_report_dates = (
        df_bki[['application_id', 'reporting_dt']].copy(deep=True)
        .drop_duplicates())
    df_report_dates.set_index('application_id', inplace=True)
    feature_analyzer = FeatureAnalyzer(
        train_woe=train_woe, test_woe=test_woe,
        target_col=TARGET_COLUMN, df_dates=df_report_dates,
        feature_thresholds=feature_thresholds
    )
    feature_analyzer.run_analysis()
    feature_analyzer.save_results('factor_analysis.csv')
    feature_analyzer.log_results()
    feature_analyzer.filter_features()

    feature_calculator = FeatureCalculator(feature_analyzer)
    feature_calculator.calculate_correlation()
    feature_calculator.calculate_vif()
    feature_calculator.calculate_total()
    feature_calculator.plot_gini_by_features()

    feature_selector = FeatureSelector(feature_analyzer, feature_calculator)
    #feature_selector.select_best_features()
    feature_selector.advanced_feature_selection()
    feature_selector.manual_correction_features()

    final_model_builder = FinalModelBuilder(
        train_data=train_woe[feature_selector.top_features],
        test_data=test_woe[feature_selector.top_features],
        target_col=TARGET_COLUMN
    )
    final_model_builder.optimize_model()
    final_model_builder.train_final_model()
    final_model_builder.evaluate_model()
    final_model_builder.save_predictions()
    final_model_builder.save_model("final_model.pkl")
