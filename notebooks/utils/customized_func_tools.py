"""This module contains the customized functions and tools for the customized function calling."""

import pandas as pd
from typing import Dict, Union, List, Any
from darts.utils.statistics import check_seasonality
from darts.timeseries import TimeSeries
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm
import numpy as np
import holidays
from darts.models import Prophet

TIME_COL = "time_col"
TARGET_COL = "target_col"
# tools for used in customized_func
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_time_col_and_target_col",
            "description": "read the csv file based on the file path, then get the time column and target column from the dataframe",
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
            "name": "get_descriptive_statistics",
            "description": """Get descriptive statistics e.g., mean, std for the column from the start index to the end index."""
            """When the start_index is None, it will start from the beginning of the dataframe. When the end_index is None, it is the length of the dataframe."""
            """The 1st data point is at index 0. For example, first 5 data points are start_index = 0, end_index=5. last 3 data points are start_index = -3, end_index=None.""",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "the file path to read",
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
                        "description": "the start index to get the statistics. Default is None which means 0. It's using .iloc[start_index:end_index]",
                    },
                    "end_index": {
                        "type": ["integer", "null"],
                        "description": "the end index to get the statistics. Default is None which means the end of the dataframe. It's using .iloc[start_index:end_index]",
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
            "description": "Get the number of outliers for the column from the file_path",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "the file path to read",
                    },
                    "col_name": {
                        "type": "string",
                        "enum": ["time_col", "target_col"],
                        "description": "the column name to get the outliers.",
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
            "name": "get_frequency",
            "description": "Get the frequency of the column from the file_path",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "the file path to read",
                    },
                    "col_name": {
                        "type": "string",
                        "enum": ["time_col", "target_col"],
                        "description": "the column name to get the frequency.",
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
            "name": "get_number_of_missing_datetime",
            "description": "Get the number of missing datetime values in the column from the file_path",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "the file path to read",
                    },
                    "col_name": {
                        "type": "string",
                        "enum": ["time_col", "target_col"],
                        "description": "the column name to get the frequency.",
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
            "name": "get_number_of_null_values",
            "description": "Get the number of null values in the column from the file_path",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "the file path to read",
                    },
                    "col_name": {
                        "type": "string",
                        "enum": ["time_col", "target_col"],
                        "description": "the column name to get the the number of null values.",
                    },
                    "fill_na": {
                        "type": "boolean",
                        "description": "whether to fill the null values by the forward fill",
                    },
                },
                "required": ["file_path", "col_name", "fill_na"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_seasonality",
            "description": "Check if the column has seasonality or return the seasonality period. If the type is 'has_seasonality', it will return True if the column has seasonality, False otherwise. If the type is 'seasonality_period', it will return the seasonality period.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "the file path to read",
                    },
                    "col_name": {
                        "type": "string",
                        "enum": ["time_col", "target_col"],
                        "description": "the column name to get and calculate the seasonality.",
                    },
                    "type": {
                        "type": "string",
                        "enum": ["has_seasonality", "seasonality_period"],
                        "description": "whether to check if the column has seasonality or return the seasonality period",
                    },
                },
                "required": ["file_path", "col_name", "type"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_the_trend",
            "description": "Check if the column has trend by extracting the trend with seasonality period, then calculating the pearson correlation between the trend and the index. It will return the pearson correlation value.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "the file path to read",
                    },
                    "col_name": {
                        "type": "string",
                        "enum": ["time_col", "target_col"],
                        "description": "the column name to calculate trend.",
                    },
                    "seasonality_period": {
                        "type": "integer",
                        "description": "the seasonality period",
                    },
                },
                "required": ["file_path", "col_name", "seasonality_period"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_moving_average",
            "description": "Calculating the moving average with a certain window_size for the column at a certain index",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "the file path to read",
                    },
                    "col_name": {
                        "type": "string",
                        "enum": ["time_col", "target_col"],
                        "description": "the column name to calculate moving average.",
                    },
                    "window_size": {
                        "type": "integer",
                        "description": "the window size for the moving average",
                    },
                    "index": {
                        "type": "integer",
                        "description": "the index to check the moving average.",
                    },
                },
                "required": ["file_path", "col_name", "window_size", "index"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "forecast_last_n_data",
            "description": "forecast the last n points data. It will return a list of forecasted values.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "the file path to read",
                    },
                    "model_name": {
                        "type": "string",
                        "enum": ["prophet", "naive"],
                        "description": "the forecasting model name",
                    },
                    "forecast_horizon": {
                        "type": "integer",
                        "description": "the number of data points to forecast",
                    },
                },
                "required": ["file_path", "model_name", "forecast_horizon"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_index_by_calc_acf_or_pacf",
            "description": "return the index with max or min correlation value among start_lag and end_lag. It will calculate the autocorrelation function (ACF) or Partial autocorrelation from the column, then return the max or min value among start_lag and end_lag.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "the file path to read",
                    },
                    "col_name": {
                        "type": "string",
                        "enum": ["time_col", "target_col"],
                        "description": "the column name to get the the number of null values.",
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
                        "description": "whether to return the index with max or min correlation value",
                    },
                },
                "required": [
                    "file_path",
                    "col_name",
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
            "name": "get_pearson_correlation_between_the_target_col_and_category",
            "description": "Get Pearson correlation between the target column and the category. The category can be weekend, weekday, and public holiday.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "the file path to read",
                    },
                    "day_category": {
                        "type": "array",
                        "description": "the list of day category to calculate the correlation. Options are ['weekend', 'weekday', 'public_holiday']",
                        "items": {
                            "type": "string",
                            "enum": ["weekend", "weekday", "public_holiday"],
                        },
                    },
                    "country_name": {
                        "type": "string",
                        "description": "the country name to get the public holidays. Default is 'US'",
                        "enum": ["US", "AU"],
                    },
                },
                "required": ["file_path", "day_category", "country_name"],
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
    df[result[TARGET_COL]] = pd.to_numeric(df[result[TARGET_COL]], downcast="float")

    if fill_na:
        df[result[TARGET_COL]] = df[result[TARGET_COL]].ffill()

    df = df.sort_values(by=result[TIME_COL]).reset_index(drop=True)

    # rename columns
    return df.rename(columns={v: k for k, v in result.items()})


def get_descriptive_statistics(
    file_path: str,
    statistic_name: str,
    col_name: str,
    start_index: Union[int, None] = None,
    end_index: Union[int, None] = None,
) -> float:
    """Get descriptive statistics between start_index and end_index for the column from the file_path e.g., mean, std, min, max, etc.
    When the start_index is None, it will start from the beginning of the dataframe.
    When the end_index is None, it will end at the end of the dataframe.

    Args:
        file_path (str): file path to read
        statistic_name (str): statistic name to get. Note, count will be the len of the dataframe
        col_name (str): column name to get the statistics.
        start_index (Union[int, None], optional): start index to get the statistics. Defaults to None.
        end_index (Union[int, None], optional): end index to get the statistics. Defaults to None.

    Returns:
        float: statistic value
    """
    df = read_csv_file_then_convert_types_and_rename_columns(file_path=file_path)

    if start_index is None:
        start_index = 0
    if end_index is None:
        end_index = df.shape[0]

    df = df.iloc[start_index:end_index]

    if statistic_name == "count":
        return df.shape[0]

    result = df.describe().loc[statistic_name, col_name]
    if isinstance(result, (int, float)):
        return round(result, 2)

    return result


def get_time_col_and_target_col(
    file_path: str = None, df: pd.DataFrame = pd.DataFrame()
) -> Dict[str, str]:
    """Get the time column and target column from the dataframe
    If file_path is given, it will read from the file_path. If df is given, it will use the df.

    Args:
        file_path (str, optional): file path to read. Defaults to None.
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


def get_number_of_outliers(file_path: str, col_name: str) -> int:
    """Get the number of outliers for the column from the file_path

    Args:
        file_path (str): file path to read.
        col_name (str): column name to get the outliers.

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
    col_name: str, file_path: Union[str, None] = None, df: pd.DataFrame = pd.DataFrame()
) -> str:
    """Get frequency of the column from the file_path

    If file_path is given, it will read from the file_path. If df is given, it will use the df.

    Args:
        col_name (str): column name to get the frequency.
        file_path (Union[str, None], optional): file path to read. Defaults to None.
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


def get_number_of_missing_datetime(file_path: str, col_name: str) -> int:
    """Get the number of missing datetime values in the column from the file_path

    Args:
        file_path (str): file path to read.
        col_name (str): column name to get the missing datetime.

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


def get_number_of_null_values(file_path: str, col_name: str, fill_na: bool) -> int:
    """Get the number of null values in the column from the file_path

    Args:
        file_path (str): file path to read.
        col_name (str): column name to get the the number of null values.
        fill_na (bool): whether to fill the null values by the forward fill

    Returns:
        int: the number of null values
    """
    df = read_csv_file_then_convert_types_and_rename_columns(
        file_path=file_path, fill_na=fill_na
    )

    return df[col_name].isnull().sum()


def get_seasonality(
    file_path: str, col_name: str, type: str, df: pd.DataFrame = pd.DataFrame()
) -> bool:
    """Check if the column has seasonality or return the seasonality period

    Args:
        file_path (str): file path to read.
        col_name (str): column name to check the seasonality.
        type (str): whether to check if the column has seasonality or return the seasonality period
        df (pd.DataFrame, optional): read dataframe. Defaults to pd.DataFrame().

    Returns:
        bool: True if the column has seasonality, False otherwise
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


def get_the_trend(file_path: str, col_name: str, seasonality_period: int) -> float:
    """Return the pearson correlation value between index and the column

    Args:
        file_path (str): file path to read.
        col_name (str): column name to check the trend.
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
    file_path: str, col_name: str, window_size: int, index: int
) -> pd.Series:
    """Calculate the moving average for the column at index

    Args:
        file_path (str): file path to read.
        col_name (str): column name to calculate the moving average.
        window_size (int): the window size for the moving average.
        index (int): the index to check the moving average. starting from 0

    Returns:
        pd.Series: the moving average series
    """
    df = read_csv_file_then_convert_types_and_rename_columns(file_path=file_path)
    return df[col_name].rolling(window=window_size, center=True).mean().iloc[index]


def get_holiday_lib(country_name: str) -> Any:
    """Return the public holidays library based on the country name

    Args:
        country_name (str): the country name to get the public holidays.

    Returns:
        Any: the public holidays library
    """

    if country_name == "US":
        holidays_lib = holidays.US()
    elif country_name == "AU":
        holidays_lib = holidays.AU()
    else:
        raise ValueError("Country name is not supported")

    return holidays_lib


def get_pearson_correlation_between_the_target_col_and_category(
    file_path: str,
    day_category: List[str],
    time_col: str = TIME_COL,
    target_col: str = TARGET_COL,
    country_name: str = "US",
) -> float:
    """Calculate the pearson correlation between target_col and day category

    Args:
        file_path (str): file path to read.
        day_category (List[str], optional): the list of day category to calculate the correlation. Options are ["weekend", "weekday", "public_holiday"].
        country_name (str): the country name to get the public holidays. Default is "US"
        time_col (str, optional): time column name. Defaults to TIME_COL.
        target_col (str, optional): target column name. Defaults to TARGET_COL.

    Returns:
        float: pearson correlation value
    """

    df = read_csv_file_then_convert_types_and_rename_columns(file_path=file_path)
    condition = pd.Series([False] * len(df))

    if "weekend" in day_category:
        condition = condition | df[time_col].apply(lambda x: x.weekday() in [5, 6])
    if "weekday" in day_category:
        condition = condition | df[time_col].apply(
            lambda x: x.weekday() in list(range(5))
        )
    if "public_holiday" in day_category:
        holiday_lib = get_holiday_lib(country_name=country_name)
        condition = condition | df[time_col].apply(lambda x: x in holiday_lib)

    return df[target_col].corr(condition.astype(int), method="pearson")


def get_index_by_calc_acf_or_pacf(
    file_path: str,
    col_name: str,
    start_lag: int,
    end_lag: int,
    type_of_correlation: str,
    return_max_or_min: str,
) -> int:
    """Return the max or min correlation value among start_lag and end_lag.
        1. calculate the autocorrelation function (ACF) or Partial autocorrelation from the column
        2. return the max or min value among start_lag and end_lag

    Args:
        file_path (str): file path to read.
        col_name (str): column name to calculate the ACF.
        start_lag (int): the start lag.
        end_lag (int): the end lag.
        type_of_correlation (str): acf or pacf
        return_max_or_min (str): max or min

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

    if return_max_or_min == "max":
        return np.argmax(corr[start_lag : end_lag + 1]) + start_lag
    elif return_max_or_min == "min":
        return np.argmin(corr[start_lag : end_lag + 1]) + start_lag
    else:
        raise ValueError('return_max_or_min should be either "max" or "min"')


def forecast_last_n_data(
    file_path: str,
    model_name: str,
    forecast_horizon: int,
    time_col: str = TIME_COL,
    target_col: str = TARGET_COL,
) -> List[float]:
    """Forecast the time series data

    Args:
        file_path (str): file path to read
        model_name (str): the model name to use for forecasting. Support only naive and prophet
        forecast_horizon (int): the forecast horizon
        time_col (str, optional): the time column name. Defaults to TIME_COL.
        target_col (str, optional): the target column name. Defaults to TARGET_COL.
    Returns:
        List[float]: the forecasted values
    """
    df = read_csv_file_then_convert_types_and_rename_columns(file_path=file_path)

    forecast_start = pd.to_datetime(df[TIME_COL].iloc[-forecast_horizon])

    # split into train and test
    df_darts_train, _ = TimeSeries.from_dataframe(
        df.set_index(time_col), value_cols=target_col
    ).split_before(forecast_start)

    if model_name == "prophet":
        model = Prophet()
        model.fit(df_darts_train)
        prediction = (
            model.predict(forecast_horizon).pd_dataframe().round(2)[target_col].tolist()
        )
    elif model_name == "naive":
        # find the last point before the forecast start date
        prediction = [
            df[df[time_col] < forecast_start][target_col].iloc[-1]
        ] * forecast_horizon
    else:
        raise ValueError("model_name should be either prophet or naive")

    return prediction
