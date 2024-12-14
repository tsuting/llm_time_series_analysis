"""This module contains the customized functions and tools for the customized function calling."""

import pandas as pd
from typing import Dict, Union

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
            "description": "Get descriptive statistics e.g., mean, std, min, max, etc. for the column from the file_path",
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
                },
                "required": ["file_path", "col_name"],
                "additionalProperties": False,
            },
        },
    },
]


# tools
def read_csv_file_then_convert_types_and_rename_columns(file_path: str) -> pd.DataFrame:
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

    # rename columns
    return df.rename(columns={v: k for k, v in result.items()})


def get_descriptive_statistics(
    file_path: str, statistic_name: str, col_name: str
) -> float:
    """Get descriptive statistics e.g., mean, std, min, max, etc. for the column from the file_path

    Args:
        file_path (str): file path to read
        statistic_name (str): statistic name to get. Note, count will be the len of the dataframe
        col_name (str): column name to get the statistics.

    Returns:
        float: statistic value
    """
    df = read_csv_file_then_convert_types_and_rename_columns(file_path=file_path)

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
    if file_path and df.empty:
        df = read_csv_file_then_convert_types_and_rename_columns(file_path=file_path)
    elif not file_path and not df.empty:
        pass
    else:
        raise ValueError("Either file_path or df should be provided.")

    return pd.infer_freq(df[col_name])


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


def get_number_of_null_values(file_path: str, col_name: str) -> int:
    """Get the number of null values in the column from the file_path

    Args:
        file_path (str): file path to read.
        col_name (str): column name to get the the number of null values.

    Returns:
        int: the number of null values
    """
    df = read_csv_file_then_convert_types_and_rename_columns(file_path=file_path)

    return df[col_name].isnull().sum()
