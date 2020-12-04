#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')

import pandas as pd
import numpy as np

pd.set_option('display.max_rows', 7)

import warnings
warnings.filterwarnings('ignore')


# # Modifying DataFrames
# 
# Methods for modifying existing Pandas dataframes fall into two categories:
# * chaining together modified copies of an existing dataframe, and
# * modifying the data contained in an existing dataframe in-place.
# 

# ## Mutating DataFrames using `assign`
# 
# The `assign` method returns a copy of a dataframe, augmented by the column(s) specified in the method. `assign` fits into the functional 'method chaining' paradigm of table manipulation: easy to understand, free of side-effects, and easily generalized to distributed data processing libraries. However, when used in Pandas, `assign` returns a copy of the dataframe, making it a poor choice for manipulating a large dataset on a single processor.
# 
# Given an input dataframe `df` and `N` Series `series1, ..., seriesN`:
# ```
# df.assign(newcol1=series1, ..., newcolN=seriesN)
# ```
# returns a copy of `df` with the `N` additional columns labeled as `newcol1, ..., newcolN`. The new columns are aligned by their indexes.

# **Example:** Given dataframe of random integers, with columns:
# * `rand_big` containing integers 0-9 inclusive,
# * `rand_small` containing integers 0-2 inclusive

# In[2]:


rand_df = pd.DataFrame({
    'rand_big': np.random.randint(10,size=5), 
    'rand_small': np.random.randint(3, size=5)
})
rand_df


# To append boolean columns to `rand_df` that specifies if a given entry of `rand_big` is divisible by 3 and if a given entry of `rand_small` is even:

# In[3]:


rand_df.assign(
    big_is_div3=(rand_df['rand_big'] % 3 == 0), 
    small_is_even=(rand_df['rand_small'] % 2 == 0)
)


# Alternatively, the names of the columns can be passed in the keys of a keyword dictionary. This is useful when the new column names contain spaces or special characters:

# In[4]:


new_cols = {
    'big_is_div3': rand_df['rand_big'] % 3 == 0,
    'small_is_even': rand_df['rand_small'] % 2 == 0
}

rand_df.assign(**new_cols)


# When building a new dataframe from derived columns as above, a common pattern is to assign new columns to an empty dataframe:

# In[5]:


pd.DataFrame().assign(**new_cols)


# ## Modifying an existing dataframe
# 
# A Pandas dataframe is a columnar data store where each column is a Series keyed by its column name. Just as columns (and subtables) selected using `[]` and `loc`, they can be also be set/modified by reassigning the data associated to a given key. While this approach is the most common way to modify a dataframe, it usually isn't the best way to modify them; using `assign` is the better choice when possible.
# 
# Given an input dataframe `df` and `N` Series `series1, ..., seriesN`:
# ```
# df['newcol1'] = series1 
# ... 
# df['newcolN'] = seriesN
# ```
# 
# modifies `df` by adding `N` additional columns labeled as `newcol1, ..., newcolN`. The new columns are aligned by their indexes.
# 
# 
# *Remark:* Modifying a dataframe via column/index reassignment is the most time/space efficient procedure on a single processor. However it has many disadvantages worth understanding. Reassignment
# * encourages procedural code that is hard to maintain and extend (unlike method chaining),
# * often has side-effects on existing dataframes that cause hard-to-find bugs. This is especially true when working in Jupyter notebooks, as re-running previous code on a modified dataframe may lead to unintended output.
# * obfuscates the data processing logic and makes it harder to translate the code into a distributed environment.

# **Example:** To append boolean columns to `rand_df` that specifies if a given entry of `rand_big` is divisible by 3 and if a given entry of `rand_small` is even:

# In[6]:


rand_df['big_is_div3'] = rand_df['rand_big'] % 3 == 0
rand_df['small_is_even'] = rand_df['rand_small'] % 2 == 0

rand_df


# ### Modifying subtables with `loc`
# 
# One can also (re-)assign columns using `loc`. Given an input dataframe `df` and `N` Series `series1, ..., seriesN`:
# ```
# df.loc[:, 'newcol1'] = series1 
# ... 
# df.loc[:, 'newcolN'] = seriesN
# ```
# 
# modifies `df` by adding `N` additional columns labeled as `newcol1, ..., newcolN`. The new columns are aligned by their indexes.
# 
# **Example:** To append boolean columns to `rand_df` that specifies if a given entry of `rand_big` is divisible by 3 and if a given entry of `rand_small` is even:

# In[7]:


rand_df.loc[:, 'big_is_div3'] = rand_df['rand_big'] % 3 == 0
rand_df.loc[:, 'small_is_even'] = rand_df['rand_small'] % 2 == 0

rand_df


# *Remark:* Note, this code actually *reassigns* the two new columns, as we modifying the original dataframe in the previous example!

# Modify tables with `loc` actually allows to modify subsets of rows as well. Generally, just as one can selecting general subtables using `loc` and boolean arrays, one can also reassign those selected values. For example, to add a column `small_number_type` that labels each value of `rand_small` as either `Even` or `Odd`:

# In[8]:


# copy rand_small data to a new column
rand_df.loc[:, 'small_number_type'] = rand_df['rand_small']

# boolean arrays for selecting even/odd entries
evens_bool = rand_df['rand_small'] % 2 == 0
odds_bool = rand_df['rand_small'] % 2 == 1

# reassign even/odd entries to the values 'Even'/'Odd'
rand_df.loc[evens_bool, 'small_number_type'] = 'Even'
rand_df.loc[odds_bool, 'small_number_type'] = 'Odd'

rand_df


# ### Warning: chained assignment
# 
# Modifying a dataframe through reassignment sometimes leads to unexpected results, especially when part of a long data cleaning process. The root of these ambiguities come from whether certain Pandas operations return copy or a reference to the underlying object being modified. This behavior is context dependent (e.g. it may depend on the data-types present in the dataframe.
# 
# Unintentionally setting new values on a copy dataframe, instead of a reference, leads to unexpected results. This most commonly happens when performing *chained assignment* -- reassigning values to a dataframe created from multiple indexing operations.
# 
# Pandas will return a `SettingWithCopyWarning` when attempting to reassign the values of a dataframe that is already a copy of a dataframe. Such code is ambiguous: is the reassignment intended for solely the copy or the original dataframe as well? Regardless, such code is unclear and `assign` should be used if possible.

# **Example:** The different chained assignments of the table `numbers` illustrate the dangers of chained assignment.

# In[20]:


numbers = pd.DataFrame([['even', 2, 4, 6], ['odd', 1, 3, 5]], columns=['type', 'A', 'B', 'C'])
numbers


# The first `.loc` returns a copy; the reassignment is performed on a copy that is not saved to a variable. This results in the value remaining unchanged.

# In[21]:


numbers_modify1 = numbers.copy()

numbers_modify1.loc[1].loc['B'] = 1
numbers_modify1


# Removing the chained assignment returns the expected result: the 3 is changed to a 1.

# In[23]:


numbers_modify3 = numbers.copy()

numbers_modify3.loc[1, 'B'] = 1
numbers_modify3


# For performance reasons, Pandas often returns a reference instead of a copy when the underlying DataFrame has homogeneous data-type. This leads to different results from the previous illustration! After dropping the `object` type column in `numbers`, chained assignment manages to give to the desired result.

# In[27]:


without_type = numbers.drop('type', axis=1) # drop returns a copy!
without_type.loc[1].loc['B'] = 1
without_type


# **Example:** Chained assignment usually appears when reassignment to on a subtable created through more complicated processing. Selecting a subtable creates a copy; modifying that subtable doesn't affect the original table.

# In[50]:


firstrow = numbers.loc[numbers['C'] == 6]
firstrow


# In[61]:


firstrow.loc[0, 'A'] = 100
firstrow


# In[63]:


numbers


# In[ ]:




