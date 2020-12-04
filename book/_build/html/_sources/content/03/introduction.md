# Querying and Describing Data

---

## Content Summary

This chapter covers techniques for exploring, understanding, and describing the data
contained in tables:
* selecting subsets of rows and columns of a table using conditions,
* classifying different kinds of measurements contained in the columns
  of a table,
* describing/summarizing the measurements of a population using
  techniques appropriate for the kind of data being described.

## Datasets

The primary dataset in this chapter consists of restaurant health
inspection data from the San Francisco Health Department.

## Summary of Library References

In the lists below, assume that the usual imports have been executed:
```
import pandas as pd
import numpy as np
import seaborn as sns
```

Selecting data:

|Function or Method Name|Description|
|---|---|
|[`[]`]()|Column Selection|
|[`loc`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.loc.html)|Selects sub-tables by index/boolean array|
|[`iloc`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.iloc.html)|Selects sub-tables by positional index|

Computing distributions:

|Function or Method Name|Description|
|---|---|
|[`value_counts`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.value_counts.html)|Returns the counts of values of a column|
|[`sort_index`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.sort_index.html)|Returns a table with rows sorted by index|
|[`np.histogram`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.histogram.html)|Returns bins and counts in each bin|

Plotting:

|Function or Method Name|Description|
|---|---|
|[`plot`](https://pandas.pydata.org/pandas-docs/stable/reference/frame.html#plotting)|plots column(s) in a DataFrame/Series|
|[`sns.distplot`](https://seaborn.pydata.org/generated/seaborn.distplot.html)|plots a rug-plot/kde/histogram of data|
|[`sns.boxplot`](https://seaborn.pydata.org/generated/seaborn.boxplot.html)|plots box-plot of data|
|[`sns.countplot`](https://seaborn.pydata.org/generated/seaborn.countplot.html)|plots a categorical histogram of data|
