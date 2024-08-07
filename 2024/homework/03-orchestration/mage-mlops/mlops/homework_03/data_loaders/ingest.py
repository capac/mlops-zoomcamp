import requests
from io import BytesIO
from typing import List
import numpy as np

import pandas as pd

if 'data_loader' not in globals():
    from mage_ai.data_preparation.decorators import data_loader
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


@data_loader
def ingest_files(**kwargs) -> pd.DataFrame:
    dfs: List[pd.DataFrame] = []
    """
    Template code for loading data from any source.

    Returns:
        Anything (e.g. data frame, dictionary, array, int, str, etc.)
    """
    # Specify your data loading logic here
    for year, months in [(2023, (3, 4))]:
        for i in range(*months):
            response = requests.get(
                'https://d37ci6vzurychx.cloudfront.net/trip-data'
                f'/yellow_tripdata_{year}-{i:02d}.parquet'
            )
            if response.status_code != 200:
                raise Exception(response.text)
            df = pd.read_parquet(BytesIO(response.content))
            df['tpep_pickup_datetime_cleaned'] = df['tpep_pickup_datetime'].astype(np.int64) // 10**9
            dfs.append(df)

    return pd.concat(dfs)


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'