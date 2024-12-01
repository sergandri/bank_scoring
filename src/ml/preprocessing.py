from src.tools.data_config import PreprocessConfig
import pandas as pd


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
    df = df.drop(columns=[
        'account_amt_currency_code',
        'collat_insured_currency_code',
        'client_id',
    ])
    return df
