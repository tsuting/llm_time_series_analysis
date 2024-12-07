import pandas as pd
from typing import Any


def convert_types(element: any) -> Any:
    """ "
    Convert element to float or datetime if possible
    """
    if element is None:
        return element

    try:
        return float(element)
    except ValueError:
        try:
            return pd.to_datetime(element)
        except:
            return element
