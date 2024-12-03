from src.tools.data_config import BKI_LOCAL_PATH, PreprocessConfig, \
    TARGET_LOCAL_PATH
import pandas as pd


def read_dataframes() -> tuple[pd.DataFrame, pd.DataFrame]:
    df_target = pd.read_csv(TARGET_LOCAL_PATH)
    df_bki = pd.read_csv(BKI_LOCAL_PATH)
    return df_target, df_bki


def convert_data_types(
    df: pd.DataFrame,
    preprocess_config: PreprocessConfig,
) -> pd.DataFrame:
    for column, dtype in preprocess_config.dtype_map.items():
        if dtype.startswith('datetime64'):
            df[column] = pd.to_datetime(df[column], errors='coerce')
        elif dtype == 'category':
            df[column] = df[column].astype('int32')
        else:
            df[column] = df[column].astype(dtype)

    return df


def drop_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(
        columns=[
            'account_amt_currency_code',
            'collat_insured_currency_code',
            'client_id',
        ]
    )
    return df
