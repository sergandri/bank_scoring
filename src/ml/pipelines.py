import pandas as pd

from src.ml.preprocessing import (
    fill_missing_values,
    read_dataframes,
    convert_data_types,
    drop_features,
    read_test_dataframe,
    replace_inf_with_nan,
)
from src.ml.feature_engineering import generate_base_features, diff_dates
from src.tools.data_config import (
    feature_config,
    preprocess_config,
    FEATURE_DATE,
    TARGET_COLUMN,
    split_config,
    feature_thresholds,
)
from src.ml.train_preparation import (
    merge_dates, merge_features_target,
    remove_non_numeric_features,
    t_t_split,
)
from src.ml.binning import Binner
from src.ml.analyzers import (
    FeatureAnalyzer,
    FeatureCalculator,
    FeatureSelector,
)
from src.ml.model_training import LRModelBuilder, CBModelBuilder
from src.tools.logger import logger


def train_full_preprocessing():
    df_target, df_bki = read_dataframes()

    df_aggr = (df_bki
               .pipe(convert_data_types, preprocess_config)
               .pipe(fill_missing_values, preprocess_config, stage=1)
               .pipe(
        generate_base_features,
        feature_config.agg_dict,
        feature_config.categorical_features,
    )

               .pipe(drop_features)
               .pipe(replace_inf_with_nan))
    df_aggr = (df_aggr
               .pipe(
        merge_features_target,
        df_target,
        diff_dates(df_aggr, feature_date=FEATURE_DATE)
    )
               .pipe(remove_non_numeric_features)
               .pipe(fill_missing_values, preprocess_config, stage=2))
    return df_aggr


def test_full_preprocessing():
    df_test = read_test_dataframe()
    df_aggr = (df_test
               .pipe(convert_data_types, preprocess_config)
               .pipe(fill_missing_values, preprocess_config, stage=1)
               .pipe(
        generate_base_features,
        feature_config.agg_dict,
        feature_config.categorical_features,
    )

               .pipe(drop_features)
               .pipe(replace_inf_with_nan))
    df_aggr = (df_aggr.pipe(
        merge_dates,
        diff_dates(df_aggr, feature_date=FEATURE_DATE)
    )
               .pipe(remove_non_numeric_features)
               .pipe(fill_missing_values, preprocess_config, stage=2)

               )
    return df_aggr


def training(df_aggr_w_target_n: pd.DataFrame):
    df_target, df_bki = read_dataframes()
    train_data, test_data = t_t_split(
        df=df_aggr_w_target_n,
        split_config=split_config,
        target_col=TARGET_COLUMN
    )

    binner = Binner()
    binner.fit_binning(train_data, target_col=TARGET_COLUMN)
    binner.save()
    train_woe = binner.transform(train_data, target_col=TARGET_COLUMN)
    test_woe = binner.transform(test_data, target_col=TARGET_COLUMN)

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
    feature_analyzer.save_results()
    feature_analyzer.log_results()
    feature_analyzer.filter_features()

    feature_calculator = FeatureCalculator(feature_analyzer)
    feature_calculator.calculate_correlation()
    feature_calculator.calculate_vif()
    feature_calculator.calculate_total()
    feature_calculator.plot_gini_by_features()

    feature_selector = FeatureSelector(feature_analyzer, feature_calculator)
    # feature_selector.select_best_features()
    feature_selector.advanced_feature_selection()
    feature_selector.manual_correction_features()

    lr_model_builder = LRModelBuilder(
        train_data=train_woe[feature_selector.top_features],
        test_data=test_woe[feature_selector.top_features],
        target_col=TARGET_COLUMN,
    )
    lr_model_builder.optimize_model()
    lr_model_builder.train_final_model()
    lr_model_builder.evaluate_model()
    lr_model_builder.save_train_predictions()
    lr_model_builder.save_model()

    cb_model_builder = CBModelBuilder(
        train_data=train_data,
        test_data=test_data,
        target_col=TARGET_COLUMN,
    )
    cb_model_builder.optimize_model()
    cb_model_builder.train_final_model()
    cb_model_builder.evaluate_model()
    cb_model_builder.save_train_predictions()
    cb_model_builder.save_model()


def predicting(df_aggr: pd.DataFrame):
    binner = Binner()
    binner.load()
    test_woe = binner.transform(df_aggr, target_col=TARGET_COLUMN)
    lr_model_builder = LRModelBuilder(
        train_data=test_woe,
        test_data=test_woe,
        target_col=TARGET_COLUMN,
    )
    lr_model_builder.load_model()
    lr_model_builder.save_real_predictions()
    cb_model_builder = CBModelBuilder(
        train_data=df_aggr,
        test_data=df_aggr,
        target_col=TARGET_COLUMN,
    )
    cb_model_builder.load_model()
    cb_model_builder.save_real_predictions()


