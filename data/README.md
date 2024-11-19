# Introduction

All the dataset are univariate and numeric time series data.

| name                 | difficulty | data points | frequency | miss values |  from |  to |  description                                                                                                                                                      | Download from     |
|----------------------|------------|-------------|-----------|-------------|-|-|------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------|
| `air_passengers.csv` | easy       | 144         | monthly     | no          | 1949-01-01| 1960-12-01| also known as the Box & Jenkins airline data, is widely available on various platforms with clear trend and seasonality.                                         | [here](https://github.com/facebook/prophet/tree/main/examples) |
| `melbourne_temp.csv` | medium     | 3652        | daily     | yes, 1984-12-31 and 1988-12-31 | 1981-01-01 | 1990-12-31| Daily temperature in Melbourne between 1981 and 1990. Clear seasonality.                                                                                          | [here](https://unit8co.github.io/darts/_modules/darts/datasets.html#TemperatureDataset) |
| `nyc_taxi.csv`       | hard       | 10320       | hourly    | no          | 2014-07-01 00:00:00 | 2015-01-31 23:30:00|Taxi Passengers in New York, from 2014-07 to 2015-01. The data consists of aggregated total number of taxi passengers into 30 minute buckets. | [here](https://unit8co.github.io/darts/_modules/darts/datasets.html#TaxiNewYorkDataset) |

Please see [`data_exploration.ipynb`](../notebooks/data_exporation.ipynb) for more details.

## Training and validation and testing

Maybe reserve the last 10 data points as the test set as a starting point?
