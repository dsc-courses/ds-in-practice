#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

pd.set_option('display.max_rows', 8)


# # Methods and Descriptive Statistics

# ### Informational Methods
# 
# Pandas methods that help the user 'peek at the data' in different ways (e.g. look at a few rows at a time, count the number of non-null entries, count the number of distinct entries). These methods are particularly useful when the data is too large to look at in its entirety.

# |Method Name|Description|
# |---|---|
# |`head`|return the first `n` entries of a Series|
# |`tail`|return the last `n` entries of a Series|
# |`count`|Count the number of non-null entries of a Series|
# |`nunique`|Returns number of unique values of a Series|

# *Example:* The `DataFrame` named `uswnt` contains information on all soccer players on the US Women's national team from 1991 through 2019.

# In[2]:


uswnt = pd.read_csv('data/world_cups.csv')


# In[3]:


# number of rows / columns
uswnt.shape


# In[4]:


# first 7 entries; players from the 90s
uswnt.head(7)


# In[5]:


# last 2 entries; players from 2019
uswnt.tail(2)


# A look at the Player column:

# In[6]:


# Look at players column; `head` also a Series method.
players = uswnt['Player']
players.head()


# In[7]:


# no duplicates
players.shape


# In[8]:


players.count()


# In[9]:


players.nunique()


# In[10]:


# Top 5: Most goals in a single world-cup tournament; note the index.
uswnt.sort_values(by='Gls', ascending=False).head()


# In[11]:


# Top goal scorer per world cup for USWNT
(
    uswnt
    .sort_values(by='Gls', ascending=False)
    .drop_duplicates(subset=['Year'])
    .sort_values(by='Year')
)


# ### Array arithmetic
# * `Series` can use array arithmetic just like Numpy
# * Warning: arrays indices are lined up before operation! (More on this later)
# 
# *Example:* Compute (1) the minutes played per appearance and (2) each players year of birth.

# In[14]:


minutes = uswnt['Min']
apps = uswnt['Apps']

(minutes / apps)


# In[15]:


year = uswnt['Year']
ages = uswnt['Age']

(year - ages)


# ### Descriptive methods
# 
# As noted in the previous section, `Series` and `DataFrame` objects are Numpy arrays with named labels. As such, 
# * Numpy functions and methods are directly applicable to Pandas objects (particularly `Series`), and
# * many Pandas methods are inherited from Numpy, often with tweaks to default arguments that are convenient for data analysis.

# *Example:* Applying Numpy functions to a `Series` (e.g. a column of a `DataFrame`) results in applying the function to the data in the underlying Numpy array.

# In[16]:


# mean age of the players
np.sum(ages) / ages.shape[0]


# In[17]:


# The mean
np.mean(ages)


# In[18]:


np.median(ages)


# *Example:* Pandas supplies these Numpy function as `Series` methods as well.

# In[19]:


ages.mean()


# In[20]:


ages.median()


# In[21]:


ages.describe()


# *Example:* The *variance* is an example of a method that differs between Numpy and Pandas.
# * In Numpy, `np.var` computes the population variance.
# * In Pandas, the `var` method computes the sample variance.
# 

# In[22]:


(((ages - ages.mean())**2).sum() / ages.shape[0])**(1/2)


# In[23]:


np.std(ages)


# In[24]:


(((ages - ages.mean())**2).sum() / (ages.shape[0] - 1))**(1/2)


# In[25]:


ages.std()


# ## `DataFrame` Methods and the `axis` keyword
# 
# * DataFrames share *many* of the same methods with Series.
#     - The dataFrame method applies the Series method to every row/column.
# * Some of these methods take the `axis` keyword argument:
#     - `axis=0`: the method is applied to series with index given by **rows**.
#     - `axis=1`: the method is applied to series with index given by **columns**.
# * Default value: `axis=0` (apply method to each column).

# In[26]:


uswnt.mean() 


# In[27]:


uswnt.max()


# In[28]:


uswnt.head()


# In[29]:


uswnt.head().sum(axis=1)


# In[30]:


uswnt.describe()


# ### The `apply` method
# 
# The `apply` method is both a Series and a DataFrame method for applying custom functions across data.
# 
# * `ser.apply(func)` applies `func` to the values contained in the Series `ser`,
# * `df.apply(func)` applies `func` to the *columns* of the DataFrame `df`,
# * `df.apply(func, axis=1)` applies `func` to the *rows* of the DataFrame `df`.
# 
# *Remark:* Notice that, when applied to a DataFrame, `func` should be a function that takes in a *Series*.

# **Example:** To create a boolean column that describes if a given player's first name ends in the letter `e`, create a custom function to pass to `apply`:

# In[68]:


def firstname_endswith_e(player):
    '''returns True if the first name ends in the letter e'''
    fn, _ = player.split(maxsplit=1)
    return fn[-1] == 'e'


# In[69]:


uswnt['Player'].apply(firstname_endswith_e)


# ### The `agg` method
# 
# The `agg` method simultaneously applies multiple Series methods to the columns of a DataFrame. Given a DataFrame `df`, 
# * `df.agg(func)` returns a Series obtained by applying the function to the columns of a `df`,
# * `df.agg([f1,...,fN])` returns a DataFrame obtained by applying each function to each column of `df`,
# * `df.agg({col1:f1,...,colN:fN})` returns a Series obtained by applying each function to column specified by its corresponding key.
# * Analogously, `agg` can also be passed a dictionary, keyed by column name, of lists of functions.
# 
# *Remark 1:* `agg` accepts function/method *names* as well, represented as strings.
# 
# *Remark 2:* `agg` has an `axis` keyword argument that applies functions row-wise instead of column-wise.

# **Example:** `uswnt.agg('max')` computes the maximum value for each column. This value is also computable using the method directly -- that is, `uswnt.max()`.

# In[53]:


uswnt.agg('max')


# **Example:** Passing a list of functions into `agg` results in a DataFrame whose rows contain the results of applying each function to columns of the original DataFrame. If a function throws an exception upon application to a column, the value in the resulting DataFrame is `NaN`.

# In[54]:


uswnt.agg(['mean', np.median, 'max'])


# **Example:** Similarly, passing a dictionary of functions keyed by column name applies the function only to the specified columns.

# In[55]:


uswnt.agg({'Player': 'max', 'Pos': 'min', 'Age': 'mean', 'Ast': 'min'})


# In[58]:


uswnt.agg({'Player': ['min', 'max'], 'Age': ['mean', np.median, 'max']})


# In[ ]:




