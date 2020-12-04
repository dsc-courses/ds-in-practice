#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# # Selecting Data
# ---
# 
# A given table usually represents a set of measurements for a given population. Oftentimes, one's interest lies in either a specific subpopulation of interest (i.e. rows) or a specific subset of measurements (i.e. columns).
# 
# For example, in the table below, one might want to 
# * restrict attention to those individuals in their thirties,
# * concern oneself with the name and age of the individuals.
# 
# Selecting rows and columns based on this criteria results in a smaller table:
# 
# ![selecting a subtable](imgs/selecting-subtables.png)
# 
# The [previous chapter](02/tabular-data.html#selecting-rows-with-loc-and-columns-with-) covered selection of single rows and columns, using `.loc[]` and `[]` respectively. This section covers general approaches to selecting a subtables of a given table. These approaches include:
# * selecting explicit subsets of observations (rows) and/or attributes (columns),
# * selecting subsets of observations and attributes based on conditions using boolean arrays.
# 

# ## Selecting explicit subsets of rows and columns
# 
# ### Row selection
# 
# The dataframe `currencies` contains different currency names, their exchange rate against the US Dollar, and the continent of the given country, indexed by the name of the country manufacturing the currency.

# In[2]:


currencies = pd.DataFrame({
    'currency': ['usd', 'yen', 'euro', 'peso', 'renminbi'],
    'exchange': [1, 105.94, 0.897, 19.64, 7.02],
    'continent': ['North America', 'Asia', 'Europe', 'North America', 'Asia']
}, index=['USA', 'JPN', 'EU', 'MEX', 'CHN'])

currencies


# Recall that `currencies.loc['MEX']` selects the attributes corresponding to `MEX`; this row is a *one dimensional* Series object. 

# In[3]:


currencies.loc['MEX']


# Selecting a subset of rows results in a potentially *smaller* table. However, such a table is still a *two dimensional* dataframe. To select a subset rows, simply pass a list to `loc`. For example, to create a table from `currencies` consisting of the rows indexed by `JAPAN` and `CHN`:

# In[4]:


currencies.loc[['JPN', 'CHN']]


# **Selecting explicit subsets of rows using `loc`**: Given a dataframe `df` and a subset `idx_list` of the index `df.index`, the dataframe `df.loc[idx_list]` consists of the rows of `df` with index given by `idx_list`.

# **Example:** If the index list consists of one a single index, the resulting object is still a two dimensional dataframe, consisting of a single row.

# In[5]:


idx_list = ['USA']
currencies.loc[idx_list]


# ### Column Selection
# 
# Recall that `currencies['exchange']` selects exchange rate column for each country in the table; this column is a one dimensional object.

# In[6]:


currencies['exchange']


# Similar to row selection, passing a list of column names produces a dataframe with the columns given in the list. For example, to select only the 'currency' and 'exchange' columns:

# In[7]:


currencies[['currency', 'exchange']]


# **Selecting explicit subsets of columns using `[]`**: Given a dataframe `df` and a subset `cols` of the columns `df.columns`, the dataframe `df[cols]` consists of the rows of `df` with columns given by the columns in `cols`.
# 
# **Example:** This method of column selection offers a convenient way of explicitly reordering the columns of a table: pass the full column list in the desired order.

# In[8]:


cols = ['continent', 'currency', 'exchange']
currencies[cols]


# ## Selection of subtables via conditions
# 
# Most often, selection of observations and attributes occur via applying some relevant criteria. For example, in a table of survey responses, one may only want to consider:
# * responses from respondents of a certain age (row selection) 
# * answers to questions that have a 100% response rate (column selection).
# 
# Such selections occur in two steps:
# 1. create a boolean index using a vector comparison that captures the selection logic,
# 2. pass the boolean index to the Pandas row/column selector. 

# ### Boolean indexing
# 
# Boolean indexes are boolean arrays that represent whether or not a condition is met for a given position in an index. Such arrays are created using logical operators on array objects. For example, a boolean index for those countries whose exchange rate is create than one is given by:

# In[9]:


currencies['exchange'] > 1


# A boolean index that reads `True` if a country is either in Asia, or has an exchange rate less than 1, is given by:

# In[10]:


(currencies['continent'] == 'Asia') | (currencies['exchange'] < 1)


# ### Selecting rows using boolean indexes
# 
# Rows of a Pandas dataframe can be selected by passing a boolean index to `loc`. For example, one can create a dataframe consisting of the rows containing JPN and MEX using a boolean array with `True` only in the second and fourth positions:

# In[11]:


bool_arr = [False, True, False, True, False]
currencies.loc[bool_arr]


# The subtable consisting of countries either in Asia or with an exchange rate less than 1 is obtained via:

# In[12]:


asia_or_exch_less_1 = (currencies['continent'] == 'Asia') | (currencies['exchange'] < 1)
currencies.loc[asia_or_exch_less_1]


# **Selecting rows using boolean indexes:** Suppose `df` is a dataframe and `bool_arr` is a boolean array of the same length of `df`. Then `df.loc[bool_arr]` is a dataframe whose rows are the rows of `df` for which the corresponding position in `bool_arr` is `True`.

# ### Selecting rows using functions
# 
# Rows of a Pandas DataFrame can also be selected by passing a *function* to loc:
# * the function takes in a DataFrame and returns a boolean array;
# * this boolean array (applied to the DataFrame at hand) is then passed to loc to select the rows as outlined above.
# 
# **Example:** To select the rows in `currencies` whose currency begins with the letter 'o':

# In[15]:


def ends_in_o(df):
    '''returns a boolean array representing
    whether each row in the currency 
    column of `df` ends in  the letter o.'''
    return df['currency'].str.endswith('o')

currencies.loc[ends_in_o]


# **selecting rows using function:** Suppose `df` is a DataFrame and `f` is a function that takes in a DataFrame and returns a boolean array. Then `df.loc[f]` returns the same DataFrame as `df.loc[f(df)]`.
# 
# This technique is useful when using method chaining, as the function generalizes the selection logic without referencing a specific DataFrame.
# 
# **Example:** Without using `sort_values`/`drop_duplicates`, and using method-chaining, return a DataFrame containing the country (or countries) with the largest exchange rate among the currencies that end with the letter 'o'.
# 
# This requires two steps:
# 1. select the countries whose currency begins with the letter 'o',
# 1. select the countries in the DataFrame from step 1 whose exchange rate is the largest.
# 

# In[19]:


def equal_to_max(df):
    '''returns a boolean array denoting if the exchange
    rate of a row of `df` is equal to the max exchange rate.'''
    return df['currency'] == df['currency'].max()


# In[20]:


(
    currencies
    .loc[ends_in_o]
    .loc[equal_to_max]
)


# *Remark:* This cannot be done by passing a boolean index directly, as the logic in step 2 refers to the output of step 1, which doesn't have a name! While this example is contrived, this constraint commonly appears when adding and modifying columns of a DataFrame using method chaining.

# ## Selecting subtables using `loc`
# 
# The `loc` selector allows simultaneous selection of rows and columns via matrix notation. That is, given a dataframe `df`, a list of indexes `idx`, and column labels `cols`, the expression `df.loc[idx, cols]` evaluates to the dataframe with rows corresponding to the index `idx` and columns corresponding to the columns in `cols`.
# 
# For example, the currency and exchange rate for MEX and JPN is given by:

# In[13]:


countries = ['MEX', 'JPN']
attributes = ['currency', 'exchange']
currencies.loc[countries, attributes]


# Similarly `loc` can also take *pairs* of boolean arrays corresponding rows and columns. Below selects the first and last row and the middle column using boolean arrays:

# In[14]:


currencies.loc[
    [True, False, False, False, True], 
    [False, True, False]
]


# The slicing operator `:` selects all rows and/or columns when passed into `loc`:

# In[15]:


currencies.loc[:, attributes]


# ### Beware: strange behaviors with row and column selection
# * What if your boolean arrays have incorrect length?
# * What if your indexes/columns are of boolean type?
# * `[]` has the same problems!
# * Duplicate column names / duplicate selection?

# In[ ]:




