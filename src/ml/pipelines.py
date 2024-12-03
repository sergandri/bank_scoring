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
from src.tools.analysis import FeatureAnalyser
from src.tools.data_config import feature_thresholds
from src.tools.analysis import FeatureSelector
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
    ))
    df_dates = diff_dates(df_aggr, feature_date=FEATURE_DATE)
    df_aggr_w_target_n = (df_aggr.pipe(
        merge_features_target,
        df_target, df_dates
    )
                          .pipe(remove_non_numeric_features)
                          .pipe(fill_missing_values, how=0))

    return df_aggr_w_target_n


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
    feature_analyser = FeatureAnalyser(
        train_woe=train_woe, test_woe=test_woe,
        target_col=TARGET_COLUMN, df_dates=df_report_dates,
        feature_thresholds=feature_thresholds
    )
    feature_analyser.run_analysis()
    feature_analyser.save_results('factor_analysis.csv')
    feature_analyser.log_results()
    feature_analyser.filter_features()

    feature_selector = FeatureSelector(feature_analyser)
    feature_selector.calculate_correlation()
    feature_selector.calculate_vif()
    feature_selector.calculate_total()
    feature_selector.plot_gini_by_features()

    binning_results = pd.read_csv('factor_analysis.csv')
    feature_filter = feature_selector.selected_features
    binning_results['indicator'] = binning_results['IV'] * 0.5 + \
                                   binning_results['Gini Train'] * 0.25 + \
                                   binning_results['Gini Test'] * 0.25
    top_list = (binning_results.query('Feature in @feature_filter')
                .sort_values('indicator', ascending=False).head(10))[
        'Feature'].tolist()
    top_list.append('target')
    logger.info(f'toplist: {top_list}')

    final_model_builder = FinalModelBuilder(
        train_data=train_woe[top_list],
        test_data=test_woe[top_list],
        target_col=TARGET_COLUMN
    )
    final_model_builder.optimize_model()
    final_model_builder.train_final_model()
    final_model_builder.evaluate_model()
    final_model_builder.save_predictions()
    final_model_builder.save_model("final_model.pkl")

