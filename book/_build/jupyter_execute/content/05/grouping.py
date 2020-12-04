#!/usr/bin/env python
# coding: utf-8

# In[35]:


import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 7)

climbs = (
    pd.read_csv('data/climbs.csv')
    .assign(Route=lambda x:x['Route'].replace('Fuhrers Finger', "Fuhrer's Finger"))
)


# # Data Granularity:
# ---

# ## From questions to code
# 
# While determining what measurements to make to understand a process is clearly important, the difficulty of understanding *what* or *who* to measure shouldn't be overlooked. One needs to be clear about both the individuals being observed, along with the attributes being measured.
# 
# **Example:** Suppose a Data Scientist, working at video streaming platform, studies users' viewing behavior. The data she uses might come in two possible forms:
# 1. Each row represents an event comprising of a user streaming a particular video; attributes include the user streaming the video, the name of the video, the length of the streaming event, whether the video was watched to completion, etc.
# 1. Each row represents a user; attributes include the average and total time spent streaming, how often a video was watched to completion, etc.
# 
# These datasets have different *data designs*; each is better suited to answer certain questions than the other. However, the user-level data is fundamentally an aggregate view of the event-level data: it gives a *coarser* view of the data. Below are a few differences between the two data designs.
# 
# |---|Event Level (fine)|User Level (coarse)|
# |---|---|---|
# |Answering questions about: |cost of running servers|net revenue per user|
# |Cost implications of data collection: |large/expensive to store|smaller/cheaper to store|
# |Privacy implications of data collection: |possibly detailed personal data|more easily anonymized|
# |Flexibility of use: |usable for a broad range of questions|drops more information about underlying events|
# |...|...|...|
# 
# Suppose, for example, the Data Scientist notices when looking at the user-level data that 5% of paying users accounts for 95% of all video watched on the platform. A knee-jerk response might be to discourage these users from such heavy use, as this small number of paying subscribers account for most of the costs of running the platform. However, a look at the event-level data might reveal that these users all have something in common: they watch the new live-streaming channels that the company is promoting, which consists of inordinately long events. In fact, the root of the problem lies in the service being promoted, not the users.
# 
# Data can be represented at different levels of granularity and it's useful to move from a fine grained picture to coarser summaries while understanding what information is lost. An astute Data Scientist treats coarse grained data with skepticism, without losing the ability to effectively use it.

# ## Group-wise operations: split-apply-combine
# 
# Moving to coarser views of a dataset involves *grouping individuals into subpopulations* and aggregating the measurements of the subpopulation into a table containing the aggregate summaries. This process follows the logic of [split-apply-combine](https://www.jstatsoft.org/article/view/v040i01/v40i01.pdf) nicely summarized in the context of data analysis by Hadley Wickham. This data processing strategy is useful across a very broad class of computational task, allowing one to:
# * cleanly describe processing logic into easy-to-understand, discrete steps, and
# * scale a solution, using parallel processing, to large datasets.
# 
# The components of split-apply-combine are as follows:
# 1. **Split** the data into groups based on some criteria (often encoded in a column).
# 2. **Apply** function(s) to each group independently.
# 3. **Combine** the results into a data structure (e.g. a Series or DataFrame)
# 
# For example, given a table with group labels A, B, C in a column `Col1`, the sum of values in each group is calculated by splitting the table in a table for each group, summing the values in each groups' table separately, and combining the resulting sums in a new, smaller, table (see the figure below).
# 
# ![split-apply-combine](imgs/split-apply-combine.png)
# 
# This abstraction allows one to develop transformations on a single table without worrying about the particulars of how it's applied across groups and returned as an aggregated result.

# ### Split: the `groupby` DataFrame method
# 
# In the context of Pandas DataFrames, the split-apply-combine paradigm is implemented via the [DataFrameGroupBy](https://pandas.pydata.org/pandas-docs/stable/user_guide/groupby.html) class.
# 
# `DataFrame.groupby(key)` is a DataFrame method that represents the "split" portion of split-apply-combine. The `DataFrameGroupBy` object it returns is a dictionary like object, keyed on a (list of) column name(s) `key`, with values given by the indices of each group.
# 
# **Example:** `df` defined below is a DataFrame with group labels contained in `Col1` and integer values in `Col2` and `Col3`. Several useful `GroupBy` properties and methods are illustrated below:

# In[9]:


df = pd.DataFrame({
    'Col1': 'A B C C B B A'.split(), 
    'Col2': [1, 2, 3, 4, 2, 5, 3],
    'Col3': [4, 3, 1, 2, 4, 5, 3]
})
df


# The `GroupBy` object does not have a descriptive string representation:

# In[10]:


G = df.groupby('Col1')
G


# However, the `groups` property returns a dictionary keyed by group label, with the indices of `df` corresponding to each group label:

# In[11]:


G.groups


# The `G.get_group(label)` method returns the DataFrame `df.loc[df[key] == label]`:

# In[12]:


G.get_group('A')


# If the aggregation only involves one of the columns, selecting that column saves space. The resulting object is a `SeriesGroupBy` object, which is similar to a `DataFrameGroupBy` object:

# In[13]:


G['Col2']


# In[14]:


G['Col2'].get_group('A')


# **Example:** In place of a column name, `groupby` also accepts a function to split data into groups:
# * the domain of the function is the index of the DataFrame, 
# * the groups are labeled by the image of the function,
# * each group consists of indices that map to the same value under the function.
# 
# For example, the following function creates a `GroupBy` object with three groups: Small, Medium, and Large.

# In[15]:


def grouper(x):
    if x <= 2:
        return 'Small'
    elif 3 < x <= 4:
        return 'Medium'
    else:
        return 'Large'


# In[16]:


(
    df
    .set_index('Col2')
    .groupby(grouper)
    .groups
)


# ### Apply-Combine: `GroupBy` methods
# 
# Recall that the 'apply-combine' steps consist of applying a transformation to the groups created in the 'split' step, followed by combining the separate results into a single data structure. Pandas `GroupBy` methods couple these two steps: while each method implements its own apply logic, all of them combine the transformed groups into a natural output data structure.
# 
# Many DataFrame methods also naturally extend to `GroupBy` objects. The Pandas [documentation](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.groupby.html) contains an exhaustive list.

# **Example:** Calculating the mean of the groups of the of the simple DataFrame `df` defined in the previous example:

# In[17]:


df


# In[18]:


G = df.groupby('Col1')
G.mean()


# **Example:** The table `climbs` contains a list of almost 4000 attempts at climbing to the summit of Mt. Rainier in Washington State, during the years 2014-2015. Each row represents a party of climbers attempting to climbing a specified *route* up the mountain, on a given *date*.

# In[19]:


climbs


# Most parties are composed of multiple climbers, as evidenced by the 'Attempted' column. The total number of people who attempted each route over the two year period is:

# In[20]:


climbs.groupby('Route')['Attempted'].sum()


# Notice the resulting Series is sorted by the grouping *key*; there are 24 routes sorted alphabetically.
# 
# To calculate the routes with the most climbers on a given day, group the climbs by both date and route name:

# In[21]:


(
    climbs
    .groupby(['Date', 'Route'])['Attempted']
    .sum()
    .sort_values(ascending=False)
    .reset_index()
)


# *Remark:* Grouping on multiple columns results in a multi-indexed Series/DataFrame; using `reset_index` on the output simplifies further processing.

# Similarly to the case of DataFrames, the `GroupBy` method **`agg`** applies columnar Series methods to the groups of a `GroupBy` object. Compare how each of the following outputs relates to the analogous DataFrame method.
# * Passing a list of functions applies each function to each column of each group
# * Passing a dictionary of functions applies the functions to the specified column in each group.

# **Example:** Using `agg` quickly creates a compact table of descriptive statistics of the attempts up the routes on Mt. Rainier:

# In[22]:


climbs.groupby('Route').agg(['count', 'sum', 'mean', np.median, 'max'])


# The columns of the output are multi-indexed, as each function is applied to both the 'Attempted' and 'Succeeded' columns of the DataFrames of each route. For example, the DataFrame lists 'Disappointment Cleaver' was:
# * attempted by 2720 climbing parties,
# * attempted by a total of 15227 climbers,
# * a total of 8246 climbers successfully made it to the summit by the route,
# * the average party size attempting the route was 5.59 climbers.

# Notice that the Date column was dropped from the output; Pandas drops non-numeric columns when any of the aggregation methods requires numeric data. Passing a dictionary of aggregations into `agg` allows for a more tailored set of aggregations:

# In[23]:


climbs.groupby('Route').agg({'Attempted':['mean', 'max'], 'Date': ['max']})


# ### Generic Apply-Combine:  `apply` method
# 
# The `apply` method is the most general `GroupBy` method; it applies any function defined on the 'grouped' dataframes and attempts to combined the result into meaningful output. While very flexible, the `apply` method is slower, less space efficient, and more prone to error than using standard `GroupBy` methods; it should be avoided when possible.
# 
# Use this method when either:
# * the transformation depends on complicated group-wise conditions on multiple columns.
# * the output dimensions of the resulting Series/DataFrame are nonstandard.
# 
# **Example:** For the dataframe `df`, generate a dataframe indexed by label `Col1`, with five of random integers chosen within the range of values of `Col2` within the group. For example, within the `A` group `Col2` contains 1, 3; the output row with index `A` should therefore contain five randomly chosen integers between 1 and 3.

# In[24]:


df


# Define a function that takes in a group-dataframe (i.e. retrievable by `df.groupby('Col1').get_group(label)`), calculates the minimum and maximum values of Col2 within that group, and returns the random integers as a Series:

# In[25]:


def generate_random(gp):
    n_rand = 5
    m, M = gp['Col2'].min(), gp['Col2'].max()
    r = np.random.randint(m, M, size=n_rand)
    return pd.Series(r, index=['rand_%d' % (x + 1) for x in range(n_rand)])


# The `apply` method combines these Series of random integers into a DataFrame indexed by label:

# In[26]:


G.apply(generate_random)


# Another common use case for apply is when the output DataFrame has the same index as the input DataFrame. This often occurs when the elements of a table need to be scaled by a groupwise condition.
# 
# **Example (transform):** Scale the DataFrame linearly, so that an entry is represented as the proportion of the range for each group. For example, in 'Col3', the 'B' group has values 3, 4, 5. The range of the 'B' group is (5 - 3) = 2. The value 4 is halfway is in the middle of the range and thus is scaled to 0.5.

# In[27]:


df


# In[28]:


def scale_by_range(gp):
    rng = gp.max() - gp.min()
    from_min = gp - gp.min()
    return from_min / rng


# The function `scale_by_range` returns a DataFrames of the same index as the input. Pandas thus aligns them by common columns, returning a dataframe of the same shape as `df`:

# In[29]:


G.apply(scale_by_range)


# Pandas provides a method specifically for this pattern:  `transform`. It's appropriate when passing a function to `apply` that returns a Series/DataFrame with the same index as the input. This method is both faster and less error prone than using `apply` in this case.

# In[30]:


G.transform(scale_by_range)


# **Example:** Among the 24 routes up Mt. Rainier, only a few of them see more than 15 attempts in 2014-15. The `apply` method can be used to filter out all attempts on routes that aren't very popular (fewer than 15 attempts in 2014-15). The split-apply-combine logic:
# 1. splits the table into separate routes on the mountain (split),
# 1. checks the size of each table (apply):
#     - leaving the table untouched if it contains at least 15 rows,
#     - otherwise, omitting the rows (returning an empty DataFrame)
# 1. Aggregating the (possibly empty) DataFrames into a filtered subtable of the original `climbs`.

# In[31]:


def filter_few(df, min_size=15):
    '''return df if it has at least `min_size` rows,
    otherwise return an empty dataframe with the same columns'''
    
    if df.shape[0] >= min_size:
        return df
    else:
        return pd.DataFrame(columns=df.columns)


# In[32]:


popular_climbs = (
    climbs
    .groupby('Route')
    .apply(filter_few)
    .reset_index(level=0, drop=True) # drop group key: Route
)

popular_climbs.sort_index()


# The resulting table is a filtered subtable of the original table. To verify all routes contained in `popular_climbs` have at least 15 attempts, we can calculate the distribution of routes:

# In[33]:


popular_climbs['Route'].value_counts()


# **Example:** This pattern is common enough that Pandas provides the `filter` method to optimize this computation. The `filter` method takes in a boolean 'filtering function' as input. The `filter` method keeps a grouped DataFrame exactly when the filter function returns `True`. Repeating the prior example, to filter out all but the popular climbing routes:

# In[34]:


climbs.groupby('Route').filter(lambda df:df.shape[0] >= 15)


# *Remark:* The same filtering is possible by a two step process that avoids a `groupby`, via:
# 1. use the empirical distribution of routes to create an array of 'popular routes'
# 1. filter the original `climbs` DataFrame using a boolean array created with the `isin` DataFrame method.
# 
# Try this! Of course, calculating an empirical categorical distribution on a Series, using `value_counts()`, is logically equivalent to splitting by Route and taking the size of each group; this approach is more memory efficient, though slower.

# In[ ]:





# In[ ]:




