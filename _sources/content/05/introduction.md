# Aggregation and Extension of Data

--- 

## Content Summary

This chapter covers more drastic data manipulation and data
transformation techniques to improve usefulness of a dataset. These
techniques include:
* Grouping data and applying transformations across those groups,
* Manipulating the granulary to a coarser view of the data, while
  understanding the information lost from applying such a
  transformation,
* Adding new observations to an existing dataset, paying special
  attention to potential differences in the process that generated the
  datasets,
* Adding new attributes to existing observations, paying special
  attention to how an imperfect correndspondence may bias the original
  dataset.
* Assesing the differences between populations of a dataset using
  statistical inference (permutation tests).


## Datasets

The two datasets used in this chapter consist of:
* All attempts to climb Mt. Rainier, in Washington State.
* The population and average income of California counties and cities.

## Summary of Library References

In the lists below, assume that the usual imports have been executed:
```
import pandas as pd
import numpy as np
import seaborn as sns
```

Aggregation methods:

|Function or Method Name|Description|
|---|---|
|[`groupby`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.groupby.html)|Split-Apply-Combine processing on tables|
|[`agg`](https://pandas.pydata.org/pandas-docs/stable/reference/groupby.html)|Apply collections of functions to groups|
|[`transform`](https://pandas.pydata.org/pandas-docs/stable/reference/groupby.html)|Apply transformations to groups|
|[`apply`](https://pandas.pydata.org/pandas-docs/stable/reference/groupby.html)|Apply general functions to groups|
|[`filter`](https://pandas.pydata.org/pandas-docs/stable/reference/groupby.html)|Filter out groups based on conditions|


Reshaping methods:

|Function or Method Name|Description|
|---|---|
|[`pivot_table`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.pivot_table.html)|Reshape (pivot) the entries of a DataFrame|


Appending and joining methods:

|Function or Method Name|Description|
|---|---|
|[`pd.concat`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.concat.html)|Concatentate a list of dataframes by rows/columns|
|[`merge`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.merge.html)|Join two DataFrames by common columns|


Datetime:

|Function or Method Name|Description|
|---|---|
|[`pd.to_datetime`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.to_datetime.html)|convert strings to datetime objects|
|[`dt` namespace](https://pandas.pydata.org/pandas-docs/stable/reference/series.html#accessors)|datetime related properties and methods|

Plotting:

|Function or Method Name|Description|
|---|---|
|[`pd.plotting.scatter_matrix`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.plotting.scatter_matrix.html)|plot a scatter-matrix|
|[`sns.scatterplot`](https://seaborn.pydata.org/generated/seaborn.scatterplot.html)|scatter-plot with easy customization|
|[`sns.catplot`](https://seaborn.pydata.org/generated/seaborn.catplot.html)|(strip/box)-plotting by categories|


