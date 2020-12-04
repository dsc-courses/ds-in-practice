#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import seaborn as sns

pd.set_option('display.max_rows', 7)


# # Features with Data Pipelines
# ---
# 
# Feature engineering can create hundreds or thousands of variables, each capturing specialized domain knowledge, so a methodical approach to developing the code for such features is important. The use of data pipelines encourage the development of flexible, clean, and performant code by:
# 1. compartmentalizing the internal logic of each feature, allowing one to add and subtract them as desired,
# 1. controlling possible parameters for features in one place,
# 1. providing a uniform interface for composing data transformation logic.
# 
# Scikit-learn implements data pipelines as sequences of `Transformer` objects.
# 
# ## Data Transformation in Scikit-Learn
# 
# Features in Scikit-learn are generated using *Transformers*. These are classes that implement the following interface:
# * `Transformer.set_params` defines parameters needed for the internal logic of the feature.
# * `Transformer.fit` takes in data and determines any parameters from the data that are necessary for creating the feature, returning the 'fit' transformer.
# * `Transformer.transform` takes in data and returns the feature defined by the transformer.
# * `Transformer.fit_transform` first calls `fit` on the given data, then applies the `transform` method to the same data used to fit the Transformer.

# **Example:** The `Binarizer` transformer creates a binary feature from a quantitative attribute. For example, suppose `purchases` contains a list of dollar amounts of purchases from a person in a given year:

# In[16]:


purchases = pd.DataFrame([[1.0], [3.0], [25.0], [50.0], [6.0], [101.0]], columns=['Amount'])


# The `Binarizer` transformer can be used to create a binary feature `large_purchase` that is 1 if a purchase is above \$20 and 0 otherwise:

# In[17]:


from sklearn.preprocessing import Binarizer
binarizer = Binarizer(threshold=20)
binarizer.transform(purchases)


# This transformer is initialized with a 'threshold' parameter, then used to transform dollar amounts to binary values according to the threshold.
# 
# *Remark:* The logic of `Binarizer` depends only on the value of 'Amount' in a given observations. This transformer's `fit` method doesn't need to do anything, as it doesn't need to use any properties from the data.

# **Example:** The `MinMaxScaler` linearly scales a quantitative attribute so that the resulting feature is between 0 and 1. That is, `MinMaxScaler` transforms a dataset `X` according to the formula:
# ```
# (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
# ```
# For example, on the `purchases` data:

# In[18]:


from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler()
mms.fit(purchases)
mms.transform(purchases)


# *Remark:* The `fit` method is required before transforming the data, as the `MinMaxScaler` must determine the minimum and maximum values of the dataset to apply the formula.

# ### Custom Transformers
# 
# If a desired feature transformation isn't already implemented in Scikit-Learn, it can still be implemented in a straightforward way. 
# 
# If the custom feature transformation logic doesn't require fitting parameters from data, the `FunctionTransformer` class implements a transformer from a given function:
# 
# **Example:** To create a Transformer that log-scales the purchases array, pass `np.log` to the `FunctionTransformer` constructor:

# In[19]:


from sklearn.preprocessing import FunctionTransformer


# In[26]:


logscaler = FunctionTransformer(func=np.log, validate=False)
logscaler.transform(purchases)


# `FunctionTransformer` can also pass parameters into the custom function. For example, if instead the `purchases` data is log-scaled in a different base, this keyword argument can be specified in the `FunctionTransformer` constructor:

# In[36]:


def log_base(arr, base):
    '''Apply Log scaling to an array with the specified base.'''
    return np.log(arr) / np.log(base)


# In[38]:


logscaler = FunctionTransformer(func=log_base, kw_args={'base': 10}, validate=False)
logscaler.transform(purchases)


# A custom transformer that requires fitting is implemented by inheriting the `TransformerMixin` base class.
# 
# ### Applying Transformations to multiple columns
# 
# By default, Scikit-Learn Transformers apply a given transformation to every input column separately. However, most datasets contain various column types that require different transformation logic.

# In[49]:


rand = pd.DataFrame(np.random.randint(10, size=(7,3)), columns='a b c'.split())
rand


# In[48]:


binarizer = Binarizer(5)
binarizer.transform(rand)


# Passing a function that selects the specified columns by name requires passing `validate=False` to FunctionTransformer (allowing the function to act on objects other than numpy arrays).

# In[54]:


def select(df, cols):
    return df[cols]

columnSelector = FunctionTransformer(func=select, validate=False, kw_args={'cols': ['a', 'b']})
columnSelector.transform(rand)


# Composing these two transformers applies the binarizer to only the first two columns:

# In[55]:


selected = columnSelector.transform(rand)
out = binarizer.transform(selected)
out


# ## Data Transformation Pipelines in Scikit Learn
# 
# Composing many feature transformers by hand is tedious and error-prone. Scikit-Learn has a `Pipeline` class to manage the composition of multiple transformers.
# 
# A `Pipeline` object is instantiated with a sequence of *named* transformers:
# ```
# translist = [('trans1', t1), ('trans2', t2),..., ('transN', tN)]
# pl = Pipeline(translist)
# ```
# 
# Each transformer must be given a name, to ease readability and help debugging.
# 
# The resulting pipeline is itself a *transformer*, with `fit` and `transform` methods. Calling `pl.fit_transform(data)` results in iteratively calling `fit_transform` on the transformers in the pipeline. `fit_ transform` roughly executes the following logic:
# ```
# out = data
# for trans in translist:
#     out = trans.fit_transform(out)
#     
# out
# ```
# 
# Similar logic applies to both the `fit` and `transform` methods.
# 
# **Example:** To combine the `columnSelector` and `binarizer` transformations into a pipeline, merely pass them as a list:

# In[63]:


from sklearn.pipeline import Pipeline
translist = [
    ('selector', columnSelector), 
    ('binarizer', binarizer)
]

pl = Pipeline(translist)
pl.fit_transform(rand)


# ### Applying Separate Transformations to Subsets of Columns
# 
# So far, transformers and pipelines have only been used to compose one data transformation after another. Most realistic scenarios however, involve applying separate transformations to different subsets of columns and putting together the resulting features into a single dataset.
# 
# Scikit-Learn handles this logic with the `ColumnTransformer` class, which separately applies transformers to subsets of columns, returning the resulting features as the columns of an array.

# **Example:** Suppose the the following features are derived from the dataset `rand`:
# * For columns 'a' and 'c', return 1 if a value is in the top half of the range of the column; otherwise return 0.
# * For columns 'a' and 'b', return 1 if a value is greater than 1-standard-deviation above the mean of the column, otherwise return 0.

# In[66]:


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler


# To approach this, create a pipeline for each feature transformation:

# In[80]:


trans1 = Pipeline([
    ('minmax', MinMaxScaler()), 
    ('greater_than_half', Binarizer(threshold=0.5))
])

trans2 = Pipeline([
    ('stdscale', StandardScaler()), 
    ('greater_than_1std', Binarizer(threshold=1))
])


# These transformations are then applied to separate subsets of columns by passing then into `ColumnTransformer`:

# In[81]:


ct = ColumnTransformer(
    [
        ('top_half_of_range', trans1, ['a', 'c']), 
        ('above_one_stdev', trans2, ['a', 'b'])
    ]
)


# There are four resulting features, as each transformation is applied to two columns:

# In[82]:


ct.fit_transform(rand.astype(float))


# In[ ]:





# In[ ]:




