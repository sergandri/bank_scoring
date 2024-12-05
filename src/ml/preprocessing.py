from src.tools.data_config import BKI_LOCAL_PATH, PreprocessConfig, \
    TARGET_LOCAL_PATH
import pandas as pd

from src.tools.logger import logger


def read_dataframes() -> tuple[pd.DataFrame, pd.DataFrame]:
    df_target = pd.read_csv(TARGET_LOCAL_PATH)
    df_bki = pd.read_csv(BKI_LOCAL_PATH)
    return df_target, df_bki


def convert_data_types(
    df: pd.DataFrame,
    preprocess_config: PreprocessConfig,
) -> pd.DataFrame:
    logger.info("Converting data types...")
    for column, dtype in preprocess_config.dtype_map.items():
        if dtype.startswith('datetime64'):
            df[column] = pd.to_datetime(df[column], errors='coerce')
        elif dtype == 'category':
            df[column] = df[column].astype('int32')
        else:
            df[column] = df[column].astype(dtype)

    return df.sort_values(by='fund_date', ascending=False)


def drop_features(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Deleting unused features...")
    df = df.drop(
        columns=[
            'account_amt_currency_code',
            'collat_insured_currency_code',
            'client_id',
        ]
    )
    return df
