# Understanding Assumptions and Data Cleaning

---

## Content Summary

This chapter focuses on assessing a dataset in terms of one's
understanding of the likely process that generated the data. This
understanding is translated into code that 'cleans' the data into a
useable format that faithfully represents the process that generated
it. 

The required topics to understand and clean data are:
* Techniques for modifying tables, to both build summaries of a given
  dataset, as well as to incrementally clean a dataset,
* Use *data provenance* to understand common ways in which a dataset
  needs cleaning, whether it be issues with data types, systematically
  incorrect values, or unfaithful data.
* Hypothesis testing helps one understand a dataset as just one sample
  of an underlying data generating process. This helps ones assess the
  quality of the dataset and how it aligns with what is known about the
  process that generated it.

## Datasets

The primary dataset in this chapter is the [College Scorecard
Dataset](https://collegescorecard.ed.gov/data/). This dataset is
compiled and published by the US government to provide the public with
information about each Title IV college in the United States.

## Summary of Library References


|Function or Method Name|Description|
|---|---|
|[`assign`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.assign.html)|Add a new column to a DataFrame|
|[`str` namespace](https://pandas.pydata.org/pandas-docs/stable/user_guide/text.html)|string methods for Series values|
|[`replace`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.replace.html)|Replace a given value|
|[`sample`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.sample.html)|Sample observations of a dataset|
