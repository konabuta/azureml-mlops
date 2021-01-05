import os
import numpy as np
import pandas as pd

from azureml.core import Dataset, Workspace
from azureml.core.authentication import AzureCliAuthentication

cli_auth = AzureCliAuthentication()

ws = Workspace.from_config(auth=cli_auth)
expected_columns = 11


def test_check_bad_schema():
    df = Dataset.get_by_name(ws, 'diabetesData').to_pandas_dataframe()
    actual_columns = df.shape[1]
    assert actual_columns == expected_columns
