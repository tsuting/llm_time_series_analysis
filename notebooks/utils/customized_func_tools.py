"""This module contains the customized functions and tools for the customized function calling."""

import pandas as pd
from typing import Dict, Union
from darts.utils.statistics import check_seasonality
from darts.timeseries import TimeSeries
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm
import numpy as np


TIME_COL = "time_col"
TARGET_COL = "target_col"
# tools for used in customized_func
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_time_col_and_target_col",
            "description": "Read the CSV file then extract the time column and target column from the dataframe.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "the file path of the dataset to read",
                    },
                },
                "required": ["file_path"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_descriptive_statistics",
            "description": "Retrieve descriptive statistics, such as the mean and standard deviation, for the specified column from the start index to the end index. If the start_index is not provided, the calculation will begin from the start of the dataframe. Similarly, if the end_index is not provided, it will extend to the end of the dataframe. start_index and end_index are used to slice the dataframe by pandas dataframe e.g., df.iloc[start_index:end_index].",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "the file path of the dataset to read",
                    },
                    "statistic_name": {
                        "type": "string",
                        "enum": [
                            "count",
                            "mean",
                            "std",
                            "min",
                            "25%",
                            "50%",
                            "75%",
                            "max",
                            "sum",
                        ],
                        "description": "the name of the statistics to get",
                    },
                    "col_name": {
                        "type": "string",
                        "enum": ["time_col", "target_col"],
                        "description": "the column name to get the statistics.",
                    },
                    "start_index": {
                        "type": ["integer", "null"],
                        "description": "the start index to get the statistics. Default is None which means 0. It's using df.iloc[start_index:end_index] so it can be negative.",
                    },
                    "end_index": {
                        "type": ["integer", "null"],
                        "description": "the end index to get the statistics. Default is None which means the length of the dataframe. It's using df.iloc[start_index:end_index] so it can be negative.",
                    },
                    "selected_day_in_a_week": {
                        "type": ["integer", "null"],
                        "description": "selected day to calculate the statistics. 0 is Monday, 1 is Tuesday, 2 is Wednesday, 3 is Thursday, 4 is Friday, 5 is Saturday, 6 is Sunday. Default is None which means everyday.",
                        "enum": [0, 1, 2, 3, 4, 5, 6],
                    },
                },
                "required": ["file_path", "statistic_name", "col_name"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_number_of_outliers",
            "description": "Get the number of outliers",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "the file path of the dataset",
                    },
                },
                "required": ["file_path"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_frequency",
            "description": "Get the frequency of the column",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "the file path of the dataset",
                    },
                },
                "required": ["file_path"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_number_of_missing_datetime",
            "description": "Get the number of missing datetime.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "the file path of the dataset",
                    },
                },
                "required": ["file_path"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_number_of_null_values",
            "description": "Get the number of null values.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "the file path of the dataset",
                    },
                    "col_name": {
                        "type": "string",
                        "enum": [TIME_COL, TARGET_COL],
                        "description": "the column name to get the the number of null values.",
                    },
                },
                "required": ["file_path", "col_name"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_seasonality",
            "description": "Check if the column has seasonality. It will return True or False.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "the file path of the dataset",
                    },
                    "type": {
                        "type": "string",
                        "enum": ["has_seasonality", "seasonality_period"],
                        "description": "whether to check if the column has seasonality or return the seasonality period",
                    },
                },
                "required": ["file_path", "type"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_trend_by_pearson_correlation",
            "description": "Check if the column has trend by extracting the trend with seasonality period, then calculating the pearson correlation between the trend and the index. It will return the pearson correlation value.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "the file path of the dataset",
                    },
                    "seasonality_period": {
                        "type": "integer",
                        "description": "the seasonality period",
                    },
                },
                "required": ["file_path", "seasonality_period"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_moving_average",
            "description": "Calculating the moving average with a certain window_size.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "the file path of the dataset",
                    },
                    "window_size": {
                        "type": "integer",
                        "description": "the window size for the moving average",
                    },
                },
                "required": ["file_path", "window_size"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_forecasting",
            "description": "forecast the data point with the given model name and given time. Support only naive and average.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "the file path of the dataset",
                    },
                    "model_name": {
                        "type": "string",
                        "enum": ["naive", "average"],
                        "description": "the forecasting model name",
                    },
                    "forecast_time": {
                        "type": "string",
                        "description": "all the data less equal than (<=) this time will be used as a training set to do forecasting.",
                    },
                },
                "required": ["file_path", "model_name", "forecast_time"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_extreme_acf_pacf_lag",
            "description": "Find the lag with the minimum or maximum value of autocorrelation or partial autocorrelation between start_lag and end_lag",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "the file path of the dataset",
                    },
                    "start_lag": {
                        "type": "integer",
                        "description": "the start lag to calculate correlation",
                    },
                    "end_lag": {
                        "type": "integer",
                        "description": "the end lag to calculate correlation",
                    },
                    "type_of_correlation": {
                        "type": "string",
                        "enum": ["acf", "pacf"],
                        "description": "the type of correlation to calculate",
                    },
                    "return_max_or_min": {
                        "type": "string",
                        "enum": ["max", "min"],
                        "description": "whether to return the lag with maximum or minimum correlation value",
                    },
                    "return_absolute": {
                        "type": "boolean",
                        "enum": [True, False],
                        "description": "Whether or not to return the absolute value of the correlation",
                    },
                },
                "required": [
                    "file_path",
                    "start_lag",
                    "end_lag",
                    "type_of_correlation",
                    "return_max_or_min",
                ],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_weekend_pearson_correlation",
            "description": "Call this function whenever you need to calculate the Pearson correlation between the target column and weekends.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "the file path to read",
                    },
                },
                "required": ["file_path"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "retrieve_a_row_by_index",
            "description": "Call this function whenever you need to retrieve a row from the dataset by index.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "the file path to read",
                    },
                    "index": {
                        "type": "integer",
                        "description": "the index of the row to retrieve. It could be negative since it's using df.iloc[index].",
                    },
                },
                "required": ["file_path", "index"],
                "additionalProperties": False,
            },
        },
    },
]


# tools
def read_csv_file_then_convert_types_and_rename_columns(
    file_path: str, fill_na: bool = True
) -> pd.DataFrame:
    """
    Read a csv file from the file path, convert the types of the columns and rename the columns.

    Steps:
    1. Read the file from the file path.
    2. Find the time column and the target column.
    3. Convert time column to datetime and target column to numeric.
    4. Rename the columns to TIME_COL and TARGET_COL.

    Args:
        file_path (str): The path to the csv file.

    Returns:
        pd.DataFrame: The dataframe with the columns converted and renamed.
    """
    # read the csv file
    df = pd.read_csv(file_path)

    # get time and target col
    result = get_time_col_and_target_col(df=df)

    # convert types
    df[result[TIME_COL]] = pd.to_datetime(df[result[TIME_COL]])
    df[result[TARGET_COL]] = pd.to_numeric(df[result[TARGET_COL]])

    if fill_na:
        df[result[TARGET_COL]] = df[result[TARGET_COL]].ffill()

    df = df.sort_values(by=result[TIME_COL]).reset_index(drop=True)

    # rename columns
    return df.rename(columns={v: k for k, v in result.items()})


def get_descriptive_statistics(
    file_path: Union[str, None],
    statistic_name: str,
    col_name: str,
    start_index: Union[int, None] = None,
    end_index: Union[int, None] = None,
    df: pd.DataFrame = pd.DataFrame(),
    selected_day_in_a_week: Union[int, None] = None,
) -> float:
    """Get descriptive statistics between start_index and end_index for the column from the file_path e.g., mean, std, min, max, etc.
    When the start_index is None, it will start from the beginning of the dataframe.
    When the end_index is None, it will end at the end of the dataframe.

    Args:
        file_path (Union[str, None]): the path of dataset
        statistic_name (str): statistic name to get. Note, count will be the len of the dataframe
        col_name (str): column name to get the statistics.
        start_index (Union[int, None], optional): start index to get the statistics. Defaults to None.
        end_index (Union[int, None], optional): end index to get the statistics. Defaults to None.
        df (pd.DataFrame, optional): read dataframe. Defaults to pd.DataFrame().
        selected_day_in_a_week (Union[int, None], optional): selected day to calculate the statistics. 0 is Monday, 1 is Tuesday, 2 is Wednesday, 3 is Thursday, 4 is Friday, 5 is Saturday, 6 is Sunday. Default is None which means everyday.

    Returns:
        float: statistic value
    """
    if whether_or_not_to_read_from_file(file_path=file_path, df=df):
        df = read_csv_file_then_convert_types_and_rename_columns(file_path=file_path)
    if start_index is None:
        start_index = 0
    if end_index is None:
        end_index = df.shape[0]

    df = df.iloc[start_index:end_index]

    if selected_day_in_a_week is not None:
        df = df[df[TIME_COL].dt.dayofweek == selected_day_in_a_week]

    if statistic_name == "sum":
        result = df[col_name].sum()
    elif statistic_name in ["mean", "std", "min", "max", "25%", "50%", "75%", "count"]:
        result = df.describe().loc[statistic_name, col_name]
    else:
        raise ValueError(
            "statistic_name should be one of count, mean, std, min, 25%, 50%, 75%, max, sum"
        )

    return result


def get_time_col_and_target_col(
    file_path: str = None, df: pd.DataFrame = pd.DataFrame()
) -> Dict[str, str]:
    """Get the time column and target column from the dataframe
    If file_path is given, it will read from the file_path. If df is given, it will use the df.

    Args:
        file_path (str, optional): the file path of the dataset Defaults to None.
        df (pd.DataFrame, optional): read dataframe. Defaults to pd.DataFrame().

    Raises:
        ValueError: converting to numeric failed
        ValueError: target column not found

    Returns:
        Dict[str, str]: dictionary with time column and target column
            e.g., {"time_col": "ds", "target_col": "y"}
    """
    if file_path and df.empty:
        df = pd.read_csv(file_path, dtype=str)
    elif not file_path and not df.empty:
        pass
    else:
        raise ValueError("Either file_path or df should be provided.")

    # find target col
    target_col = None
    for col in df.columns.tolist():
        try:
            df[col] = pd.to_numeric(df[col], downcast="float")
        except ValueError:
            continue
        target_col = col
        break

    if not target_col:
        raise ValueError("No target column found")

    # find time col
    time_col = df.columns.tolist()
    time_col.remove(target_col)
    assert len(time_col) == 1, f"Expected 1 datetime column, got {len(time_col)}"
    time_col = time_col[0]

    return {TARGET_COL: target_col, TIME_COL: time_col}


def get_number_of_outliers(file_path: str, col_name: str = TARGET_COL) -> int:
    """Get the number of outliers for the column from the file_path

    Args:
        file_path (str): the path of dataset
        col_name (str): column name to get the number of outliers. Defaults to TARGET_COL.
    Returns:
        int: the number of outliers
    """
    df = read_csv_file_then_convert_types_and_rename_columns(file_path=file_path)

    # Calculate quantiles and IQR
    Q1 = df[col_name].quantile(0.25)
    Q3 = df[col_name].quantile(0.75)
    IQR = Q3 - Q1

    # Calculate upper and lower bounds
    upper_bound = Q3 + 1.5 * IQR
    lower_bound = Q1 - 1.5 * IQR

    # Find and print outliers
    outliers = df[(df[col_name] > upper_bound) | (df[col_name] < lower_bound)]

    return len(outliers)


def get_frequency(
    col_name: str = TIME_COL,
    file_path: Union[str, None] = None,
    df: pd.DataFrame = pd.DataFrame(),
) -> str:
    """Get frequency of the column from the file_path

    If file_path is given, it will read from the file_path. If df is given, it will use the df.

    Args:
        col_name (str): column name to get the frequency. Defaults to TIME_COL.
        file_path (Union[str, None], optional): the file path of the dataset Defaults to None.
        df (pd.DataFrame, optional): read dataframe. Defaults to pd.DataFrame().

    Raises:
        ValueError: Either file_path or df should be provided.

    Returns:
        str: inferred frequency from pd.infer_freq
    """
    # read the csv file only if file_path is provided and df is not provided

    if whether_or_not_to_read_from_file(file_path=file_path, df=df):
        df = read_csv_file_then_convert_types_and_rename_columns(file_path=file_path)

    return pd.infer_freq(df[col_name])


def whether_or_not_to_read_from_file(file_path: str, df: pd.DataFrame) -> bool:
    """Check if the file_path or df is provided

    Args:
        file_path (str): the file path to read
        df (pd.DataFrame): pandas dataframe

    Raises:
        ValueError: both are provided or neither of them are provided

    Returns:
        bool: True if file_path is provided and df is empty, False otherwise
    """
    if file_path and df.empty:
        return True
    elif not file_path and not df.empty:
        return False
    else:
        raise ValueError("Either file_path or df should be provided.")


def get_number_of_missing_datetime(file_path: str, col_name: str = TIME_COL) -> int:
    """Get the number of missing datetime values in the column from the file_path

    Args:
        file_path (str): the file path of the dataset
        col_name (str): column name to get the missing datetime. Defaults to TIME_COL.

    Returns:
        int: the number of missing datetime
    """
    df = read_csv_file_then_convert_types_and_rename_columns(file_path=file_path)

    # get the range
    datetime_range = pd.date_range(
        start=df[col_name].min(),
        end=df[col_name].max(),
        freq=get_frequency(col_name=col_name, df=df),
    )

    return len(set(datetime_range.tolist()) - set(df[col_name].tolist()))


def get_number_of_null_values(file_path: str, col_name: str) -> int:
    """Get the number of null values in the column from the file_path

    Args:
        file_path (str): the file path of the dataset
        col_name (str): column name to get the the number of null values.

    Returns:
        int: the number of null values
    """
    df = read_csv_file_then_convert_types_and_rename_columns(
        file_path=file_path, fill_na=False
    )

    return df[col_name].isnull().sum()


def get_seasonality(
    file_path: str,
    type: str,
    col_name: str = TARGET_COL,
    df: pd.DataFrame = pd.DataFrame(),
) -> Union[bool, int]:
    """Check if the column has seasonality or return the seasonality period

    Args:
        file_path (str): the file path of the dataset
        col_name (str): column name to check the seasonality. Defaults to TARGET_COL.
        type (str): whether to check if the column has seasonality or return the seasonality period
        df (pd.DataFrame, optional): read dataframe. Defaults to pd.DataFrame().

    Returns:
        Union[bool, int]: True if the column has seasonality, False otherwise
    """

    if whether_or_not_to_read_from_file(file_path=file_path, df=df):
        df = read_csv_file_then_convert_types_and_rename_columns(file_path=file_path)

    has_seasonality, seasonality_period = check_seasonality(
        TimeSeries.from_dataframe(df, value_cols=col_name), max_lag=len(df)
    )

    if type == "has_seasonality":
        return has_seasonality
    elif type == "seasonality_period":
        return seasonality_period
    else:
        raise ValueError("type should be either has_seasonality or seasonality_period")


def get_trend_by_pearson_correlation(
    file_path: str,
    seasonality_period: int,
    col_name: str = TARGET_COL,
) -> float:
    """Return the pearson correlation value between index and the trend.
        1. decompose the time series to get trend
        2. calculate the pearson correlation between the trend and the index

    Args:
        file_path (str): the file path of the dataset
        col_name (str): column name to check the trend. Defaults to TARGET_COL.
        seasonality_period (int): the seasonality period

    Returns:

    """
    df = read_csv_file_then_convert_types_and_rename_columns(file_path=file_path)

    # decompose the time series to get trend
    trend = seasonal_decompose(
        df[col_name], model="additive", period=seasonality_period
    ).trend

    # get the trend
    return pd.Series(trend).corr(pd.Series(range(len(trend))), method="pearson")


def get_moving_average(
    file_path: str,
    window_size: int,
    col_name: str = TARGET_COL,
) -> pd.Series:
    """Calculate the moving average

    Args:
        file_path (str): the file path of the dataset
        col_name (str): column name to calculate the moving average. Defaults to TARGET_COL.
        window_size (int): the window size for the moving average.

    Returns:
        pd.Series: the moving average series
    """
    df = read_csv_file_then_convert_types_and_rename_columns(file_path=file_path)
    return df[col_name].rolling(window=window_size).mean()


def calculate_weekend_pearson_correlation(
    file_path: str,
    col_name: str = TARGET_COL,
    time_col: str = TIME_COL,
) -> float:
    """Calculate the Pearson correlation between the target column and weekends.
    Args:
        file_path (str): the file path of the dataset
        time_col (str, Optional): time column name. Defaults to TIME_COL.
        col_name (str): target column name. Defaults to TARGET_COL.

    Returns:
        float: pearson correlation value
    """

    df = read_csv_file_then_convert_types_and_rename_columns(file_path=file_path)

    condition = df[time_col].apply(lambda x: x.weekday() in [5, 6])

    return df[col_name].corr(condition.astype(int), method="pearson")


def get_extreme_acf_pacf_lag(
    file_path: str,
    start_lag: int,
    end_lag: int,
    type_of_correlation: str,
    return_max_or_min: str,
    return_absolute: bool,
    col_name: str = TARGET_COL,
) -> int:
    """Return the max or min correlation value among start_lag and end_lag.
        1. calculate the autocorrelation function or Partial autocorrelation
        2. return the absolute max or min value among start_lag and end_lag

    Args:
        file_path (str): the file path of the dataset
        col_name (str): column name to calculate the correlation. Defaults to TARGET_COL.
        start_lag (int): the start lag.
        end_lag (int): the end lag.
        type_of_correlation (str): the type of correlation to calculate. Support only acf and pacf
        return_max_or_min (str): whether to return the max or min correlation value
        return_absolute (bool): whether to return the absolute value of the correlation

    Returns:
        int: index with max or min correlation value among start_lag and end_lag
    """
    df = read_csv_file_then_convert_types_and_rename_columns(file_path=file_path)

    if type_of_correlation == "acf":
        corr, _ = sm.tsa.acf(df[col_name], alpha=0.05, nlags=end_lag)
    elif type_of_correlation == "pacf":
        corr, _ = sm.tsa.pacf(df[col_name], alpha=0.05, nlags=end_lag)
    else:
        raise ValueError('type_of_correlation should be either "acf" or "pacf"')

    if return_absolute:
        corr = np.abs(corr)

    if return_max_or_min == "max":
        return np.argmax(corr[start_lag : end_lag + 1]) + start_lag
    elif return_max_or_min == "min":
        return np.argmin(corr[start_lag : end_lag + 1]) + start_lag
    else:
        raise ValueError('return_max_or_min should be either "max" or "min"')


def get_forecasting(
    file_path: str,
    model_name: str,
    forecast_time: str,
    col_name: str = TARGET_COL,
) -> float:
    """Forecast the next data point with the given model name. Support only naive and average.

    Args:
        file_path (str): file path to read
        model_name (str): the model name to use for forecasting. Support only naive and prophet
        col_name (str): the col name for forecasting. Defaults to TARGET_COL.
        forecast_time (str): the datetime to forecast
    Returns:
        float: the forecasting value
    """
    df = read_csv_file_then_convert_types_and_rename_columns(file_path=file_path)

    df_train = df[df[TIME_COL] <= pd.to_datetime(forecast_time)]

    if model_name == "naive":
        # find the last point before the forecast start date
        return df_train[col_name].iloc[-1].round(2)
    elif model_name == "average":
        return get_descriptive_statistics(
            df=df_train, file_path=None, statistic_name="mean", col_name=col_name
        ).round(2)
    else:
        raise ValueError("model_name should be either average or naive")


def retrieve_a_row_by_index(
    file_path: str,
    index: int,
) -> dict:
    """Return the row of the certain index in the dataframe

    Args:
        file_path (str): the file path of the dataset
        index (int): the index to get the time

    Returns:
        dict: the dict of the row
    """
    df = read_csv_file_then_convert_types_and_rename_columns(file_path=file_path)
    return df.iloc[index].to_dict()
