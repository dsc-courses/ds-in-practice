#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
pd.set_option('display.max_rows', 7)


# # Pandas Data Types and Performance Considerations
# ---
# 
# While the rest of the chapter emphasized learning to work with tabular data in Pandas, this section focuses on details of how Pandas implements these data structures. Understanding these details is an necessary if using Pandas as the primary library for tabular manipulation.
# 
# While the Pandas library performs many computations very fast, the Pandas library is *not* developed to be optimized for speed. Instead, *Pandas optimizes developer ease*. Writing code in Pandas should be easy, fast, and require little overhead. Pandas embraces the Donald Knuth truism that:
# 
# > The real problem is that programmers have spent far too much time worrying about efficiency in the wrong places and at the wrong times; premature optimization is the root of all evil (or at least most of it) in programming.
# 
# Pandas emphasizes faster-to-code yet slower-to-run design patterns under the assumptions that:
# * The data scientist iteratively performs analyses; to start there's no clear computational goal.
# * What may be slower to run on a single laptop, may distribute well to a large cluster of servers using a distributed framework like Spark or Dask.
# * Datasets are often small and easily handled with (somewhat) inefficient code (and a sample of the data may help start the analysis).
# 
# However, it's equally important to know the drawbacks and strengths of the library, as it often becomes necessary to push Pandas to the limits of what it can handle. In particular, 
# * Resource constraints (e.g. a small laptop) shouldn't make a complete analysis of a large dataset impossible.
# * Moving from Pandas to a more performant library involves significant developer time. The larger the dataset one can process with Pandas, the faster one can do the work.
# 
# ## Pandas is built upon Numpy
# 
# The C-optimized libraries powering Pandas is thanks to the Numpy library (which stands for '*Num*erical *py*thon'). The Pandas DataFrame is a columnar collection of Numpy arrays and thus many built-in DataFrame methods are fast Numpy methods applied across the column(s) of a DataFrame. 
# 
# Pandas is written to take advantage of Numpy performance, which leverages *vectorized code*. As such, it relies on the python interpreter knowing:
# 1. the plan of execution ahead of time.
# 1. the type of possible inputs and outputs that methods will use.
# 1. the size of possible inputs and outputs.
# 
# **Tip:** These needs translate to a few practical habits that lead to faster and more efficient Pandas code:
# 1. Never loop over the rows of a DataFrame (operations over columns are vectorized by Numpy array calculations).
# 1. Use built-in DataFrame methods on columns, over custom python functions, whenever possible (these functions are C-optimized Numpy methods).
# 1. Explicitly type the data if memory is an issue (more on that later!).
# 
# **Example:** Use the `%timeit` magic in Jupyter Notebooks to assess the difference in execution time. The DataFrame below contains one million observations containing five attributes, each with a value between 0 and 10. Using `%timeit`, one compares the execution time of taking the average of each of the columns via:
# * the built-in `Numpy` method,
# * a 'user-defined' average function, applied using the `apply` method,
# * looping through the rows and columns of the data to compute the mean in pure python.

# In[3]:


data = pd.DataFrame(np.random.randint(0,10,size=(10**6,5)), columns='a b c d e'.split())
data


# In[12]:


get_ipython().run_line_magic('timeit', 'data.mean()')


# In[11]:


get_ipython().run_line_magic('timeit', 'data.apply(lambda x:sum(x) / len(x))')


# In[13]:


get_ipython().run_cell_magic('timeit', '', '\nmeans = []\nfor c in data.columns:\n    s = 0\n    for x in data[c]:\n        s += x\n    means.append(s / len(data[c]))')


# Notice that the built-in mean method is faster by a 30!
# 
# *Remark:* The `%timeit` times a single line of code, while `%%timeit` times the execution of an entire cell.

# **Example:** The DataFrame method `DataFrame.info()` displays the column types, along with memory consumption:

# In[17]:


data.info()


# ## Data Types
# 
# In Pandas, a **Data Type** is a classification that specifies the type of the values of a column. Understanding data types in Pandas leads to cleaner, better optimized (in both space and time), less error-prone code. In particular, Pandas handling of data types lends itself to the creation of hard-to-spot computational errors.
# 
# The data types Pandas stores in its DataFrames are roughly the same as the Numpy data types. However, Pandas diverges from Numpy in a few ways:
# * Pandas infers the correct data types from pure python types, or types on disk (and is sometimes wrong!).
# * Pandas allocates large amounts of memory for data types by default (prioritizes correctness over efficiency).
# * A column's data type determines which operations can be applied to it:
#     - Numpy arrays are by default of homogeneous data type.
#     - Pandas DataFrames are heterogeneous, column oriented tables. The columns are homogeneous, implying that column methods are fast.
# * Pandas makes heavy use of the `object` data-type, which contains generic 'object' values that may be of mixed type. Performing operations on these columns is slow.

# The DataFrame attribute `dtypes` returns a Series of data-types of each column, indexed by column name:

# In[22]:


data.dtypes


# The table below contains a list of Pandas data types and their equivalents in other common scenarios:
# 
# |Pandas dtype|Python type|NumPy type|SQL type|Usage|
# |---|---|---|---|---|
# |object|NA|object|NA|Mixed types|
# |object|str|string, unicode|NA|Text|
# |int64|int|int_, int8,...,int64, uint8,...,uint64|INT, BIGINT| Integer numbers|
# |float64|float|float_, float16, float32, float64|FLOAT| Floating point numbers|
# |bool|bool|bool_|BOOL|True/False values|
# |datetime64|datetime|datetime64[ns]|DATETIME|Date and time values|
# |timedelta[ns]|timedelta|NA|NA|Differences between two datetimes|
# |category|NA|NA|ENUM|Finite list of text values|
# 
# *Remark:* Numpy improves performance by explicitly controlling the precision of the values contained in an array. While great for speed and space, these options are a hassle to constantly specify; Pandas always defaults to 64 bits.

# **Example:** Numpy and Pandas follow different conventions for data type inference. Numpy coerces array values to a homogeneous type, whereas Pandas defaults to using a mixed 'object' type.
# 
# The data define below consists of a single observation with two attributes: a single character (string) and a single integer.
# 1. By default, Numpy coerces the integer to a string, resulting in an array of type `<U1` (unicode string of length 1).
# 1. Pandas stores each column in its own array, each of a different type.
# 1. The dtype can be explicitly when defining the array.

# In[35]:


data = [['a', 1]]


# In[36]:


np.array(data)


# In[37]:


pd.DataFrame(data).info()


# In[38]:


np.array(data, dtype=np.object)


# ### Representing missing data
# 
# Missing Data in Pandas is represented by a special value `NaN` that stands for 'Not a Number'. These values are common to many programming languages and share a common specification. In particular, `NaN` values are floating point numbers that behave in peculiar ways shared with other unusual mathematical objects like $\infty$. For example, it is often the output of methods returning a value for a zero-division.
# 
# A `NaN` value can be defined by hand using Numpy's implementation: `np.NaN`.

# The behavior of `NaN` with respect to comparisons is unusual, and is summarized in the following table:
# 
# |Comparison|	`NaN ≥ x`|	`NaN ≤ x`|	`NaN > x`	|`NaN < x`	|`NaN = x`|	`NaN ≠ x`|
# |---|---|---|---|---|---|---|
# |Result|	Always False|	Always False|	Always False|	Always False|	Always False|	Always True|
# 
# Thus, when checking when a given value is `NaN`, one should always use a function or method implemented for such comparisons:

# In[14]:


list_of_values = [0, 1, np.NaN, 3, 4, np.NaN, 6]
list_of_values


# The following code intends to loop through the elements of the list, stating when a value is present or missing. Since it uses a `==` comparison to `np.NaN`, the code incorrectly states that every element is there!

# In[19]:


for k, x in enumerate(list_of_values):
    if x == np.NaN:
        print('index %d is missing' % k)
    else:
        print('index %d is present' % k)


# Using the function `pd.isnull` returns the correct result:

# In[9]:


for k, x in enumerate(list_of_values):
    if pd.isnull(x):
        print('index %d is missing' % k)
    else:
        print('index %d is present' % k)


# The Pandas Series and DataFrame method `isnull()` returns a boolean array that indicates whether a given entry is missing.

# In[22]:


pd.Series(list_of_values).isnull()


# This boolean array, for example, can select all non-missing rows of a DataFrame.
# 
# *Remark:* There is also a `nan` value that is implemented in pure python. While similar to Numpy's version, it differs in one significant way: Numpy's `NaN` is behaves like a single value, that always occupies the same location in memory; Python's `nan` values are objects that occupy different locations in memory whenever a new object is instantiated. As Numpy's implementation is more performant, always fall back to using `np.NaN` when possible.
# 
# The example below illustrates this observation. A python `nan` value is instantiated by `float('nan')`.
# 
# The variables `a,b` are two different instances of python's `nan` object, whereas `c,d` both represent `np.NaN`. Notice all of them have the same string representation:

# In[27]:


a, b, c, d = float('nan'), float('nan'), np.NaN, np.NaN
a, b, c, d


# Using the deep comparison operator `is`, the observation above becomes clear:
# * The same python `nan` value verifies as occupying the same location in memory,
# * The two different python `nan` values occupy different locations in memory,
# * The two different `np.NaN` values occupy the same location in memory.

# In[33]:


a is a, a is b, c is d


# ## Copies and Views in Pandas
# 
# The `values` DataFrame attribute accesses the underlying Numpy array of a DataFrame. Accessing the underlying Numpy data structure allows one to resort to (in-place) Numpy operations when performance becomes an issue. However, it's unfortunately quite complicated and unpredictable to understand exactly how the underlying Numpy array is being stored and used by Pandas. 
# 
# In particular, Pandas stores a DataFrame in two different ways, depending on the context:
# 1. Pandas will create a DataFrame from a *copy* of an array in most situations. For such an array, the `.values` attribute, along with all DataFrame methods, will also return a copy.
# 1. Pandas will sometimes create a DataFrame as a *view* (or reference) of an existing array. This results in huge performance gains, at the expense of possible side-effects from in-place modifications. The most common situation for which this occurs is when a DataFrame contains homogeneous data (and thus can store an unmodified multidimensional Numpy array). However, such situations are not guaranteed and should *never* be depended upon!
# 
# **Example:** The follow example illustrates that an in-place modification of a DataFrame works in some cases (homogeneous data) and not others. The table `homogeneous` consists only of integers and `.values` returns a reference to an existing array:

# In[40]:


homogeneous = pd.DataFrame([[0,1,2],[3,4,5]], columns='a b c'.split())
homogeneous


# In[41]:


homogeneous.values


# An in-place reassignment of the upper-left value of the array `homogeneous.values` results in a changed DataFrame:

# In[42]:


homogeneous.values[0][0] = 1
homogeneous


# On the other hand, the table `heterogeneous` contains a column of strings alongside the *same DataFrame as before*. In this case, `.values` returns a copy that cannot be modified in-place!

# In[43]:


heterogeneous = pd.DataFrame([[0,1,2,'a'],[3,4,5,'b']], columns='a b c d'.split())
heterogeneous


# In[44]:


heterogeneous.values


# An in-place reassignment of the upper-left value of the array `heterogeneous.values` results in a unchanged DataFrame:

# In[45]:


heterogeneous.values[0][0] = 1
heterogeneous


# ## Method Chaining
# 
# Pandas usually returns copies:
# - this pattern is intentional (functional programming paradigm): no side-effects
# - can create problems when assigning variables to each sub-step (e.g. needlessly high memory usage)
# - fix this by chaining together methods!
#     - other pros: easy to read, understand, and change steps.
#     - don't litter the namespace with dozens of temporary variable names
#     
#     
# **Example:** Saw in previous section, calculating the top goal scorer in each world-cup was a three step process:
# * sort the table of players descending by the number of goals scored in each tournament (`sort_values`),
# * keep only the first (i.e. highest) number for each year (`drop_duplicates`),
# * sort the table by tournament year.
# 
# Procedurally, one can implement these steps exactly as outlined in English above:

# In[ ]:


uswnt_by_goals = uswnt.sort_values(by='Gls', ascending=False)
most_goals_per_year = uswnt_by_goals.drop_duplicates(subset=['Year'])
most_goals_per_year_sorted = uswnt_most_goals_per_year.sort_values(by='Year')


# This approach: 
# * is hard to read, due to introducing unnecessary variables at each step, 
# * creates copies of each DataFrame, assigning them to names at every step; due to these assignments, python does not release this memory until later than necessary.
# 
# Much better to use method chaining, applying methods directly to the output of the previous method application. Python's indentation makes the steps easy to read and parse. Moreover, as each subsequent copy is never assigned to a variable, the interpreter knows it can release the memory as soon as it executes the method.

# In[ ]:


(
    uswnt
    .sort_values(by='Gls', ascending=False)
    .drop_duplicates(subset=['Year'])
    .sort_values(by='Year')
)


# *Remark:* Many Pandas methods have an 'inplace' keyword. Surprisingly, this option **does not** result in an in-place operation. It still returns a copy that is reassigned to the variable being modified. As such, **the inplace keyword should never be used**; it will eventually be removed from the library entirely.
