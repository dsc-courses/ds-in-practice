# The Basics of Tabular Data

---

## Content Summary

The basics of tabular data consist of understanding:
* the structure of a table and how it represents a real-world
  phenomenon,
* the basic operations that can be performed on a table and how they
  reflect the real-world phenomenon it represents,
* the computational foundations for tabular data structures in Pandas.

## Datasets

The primary dataset in this chapter consists of player statistics from
the US Women's National Team in Soccer between 1991 and 2019. The data
is taken from [Football
Reference](https://fbref.com/en/comps/106/Womens-World-Cup-Stats).

## Summary of Library References

In the lists below, assume that the usual imports have been executed:
```
import pandas as pd
import numpy as np
```


### Creating Tabular Structures:

|Function or Method Name|Description|
|---|---|
|[`pd.Series`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.html)|Series constructor|
|[`pd.DataFrame`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html#pandas.DataFrame)|DataFrame constructor|
|[`pd.read_csv`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html)|Reading CSV from file|


### Series/DataFrame attributes and methods:

|Function or Method Name|Description|
|---|---|
|[`shape`](http://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.shape.html)|Number of rows/columns|
|[`head`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.head.html)|Returns first few lines|
|[`tail`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.tail.html)|Returns the last few lines|
|[`nunique`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.nunique.html)|Returns the number of unique values|
|[`dtypes`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.dtypes.html)|Returns the type of the column(s)|
|[`astype`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.astype.html)|Returns column(s) coerced to a given type|
|[`sort_values`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.sort_values.html)|sorts Series/DataFrame according to its values|
|[`drop_duplicates`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.drop_duplicates.html)|drops duplicates indices/columns|
|[`Series.apply`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.apply.html) and [`DataFrame.apply`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.apply.html)|apply a function to the entries of a Series / slices of a DataFrame|
|[`Series.agg`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.agg.html) and [`DataFrame.agg`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.agg.html)|apply a collection of functions to a Series/DataFrame|


### Methods for computing descriptive statistics on Series/DataFrames:

|Function or Method Name|Description|
|---|---|
|[`describe`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.describe.html)|Returns descriptive statistice of column(s)|
|[`count`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.count.html)|Returns the number of non-null entries|
|[`sum`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.sum.html)|Returns the sum|
|[`median`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.median.html)|Returns the median|
|[`mean`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.mean.html)|Returns the median|
|[`std`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.std.html)|Returns the *sample* standard deviation|
|[`var`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.var.html)|Returns the *sample* variance|
