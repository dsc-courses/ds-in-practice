#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

pd.set_option('display.max_rows', 7)


# # Describing Different Kinds of Data
# ---
# 

# In order to understand and *describe* a dataset statistically, the observations need to be measured in a quantifiable way. However, the attributes of a dataset vary drastically based on the nature of what is being measured; datasets are often a mixture of numbers, labels, and language-based descriptions. 
# 
# Specifying the *kind of data* contained in an attribute helps define strategies to quantify and describe the population in terms of the attribute.

# **Example:** The dataset below contains information on Health Department inspections for restaurants in San Francisco. Each row describes a different inspection of a restaurant in the city.

# In[67]:


inspections = pd.read_csv('data/inspections.csv')
inspections.head()


# Broadly, there are three different kinds of attributes in the health inspections dataset. Examples of these three types are given below:
# 
# 1. The `inspection_score` is a numerical column; calculating mathematical quantities like sums and averages on this column makes sense and represent useful descriptions of the population of health inspections.
# 1. The `inspection_date` column(s), while composed of numbers, are used as a way to *order events*. That they are represented by numbers is coincidental (e.g. One could represent month "12" as "Dec") and computing statistics on these numbers often doesn't make sense.
# 1. The `business_name` is not represented in a usual way by numbers and there is no clear way to do so. Additionally, this fields have no inherent ordering.
# 
# As it is important to understand all of these attributes to understand the dataset; different strategies for describing the fields depends on the *kind* of data the attribute represents.

# ## Kinds of data
# 
# Attributes of a dataset generally fall into one of three types:
# 1. An attribute is **quantitative** if its values are numeric and standard mathematical operations (e.g. mean, sum, ratios) on those values make sense.
# 1. An attribute is **ordinal** if its values have an ordering from smallest to greatest. Equivalently, there is a one-to-one correspondence between the values and a subset of the number-line, *for which the order in the data is reflected in the ordering of the number-line*.
# 1. An attribute is **nominal** if the values are differentiated by only their label; they are neither quantitative nor ordinal.
# 
# An attribute is referred to as **categorical** if it is either ordinal or nominal.
# 
# *Remark:* The classification of attributes into these types are not strict; they merely serve to clarify how to view, process, and analyze an attribute. The classification of an attribute into a certain kind may depend on both *the dataset being considered* as well as *the question being asked*.

# **Example:** In the inspection dataset,
# * The `inspection_score` attribute is *quantitative*; for example, one could calculate the average inspection score.
# * The `inspection_date` attribute is ordinal, ordering the values from earliest date to the most recent date.
# * The `risk_category` attribute is ordinal, ordering the values from `Low Risk` to `High Risk`.
# * The `business_name` attribute is nominal, as there is no obvious ordering of restaurant names.
# * The `business_postal_code` attribute is nominal, as there is no obvious ordering of zip-codes using integers.

# **Example:** Even though the Month attribute is of numeric type, it is *nominal*. For example, 5 and 6 (May and June) are only close to each other in the data when the observations occur in the same year. However, there are a few subtleties illustrated by the following hypothetical situations:
# * If the dataset consists of only a single year, then the Month attribute is likely *ordinal*.
# * The meaning of "close to each other in the data" depends on the question being asked of the data. For example, if the dataset is answering questions on whether restaurant fail health inspections more often in summer than winter, then a comparison of the inspections between May of two different years may be "closer" than a comparison of the inspections that occurred between May and October of the same year.

# ## Empirical distributions and kinds of data
# 
# The typical starting point in understanding a fixed dataset is to understand the distribution of values of each attribute. 
# 
# The **Empirical Distribution** of an attribute is the distribution of observed data. That is, it describes the proportion of the whole made up by each value. If an attribute is quantitative (and continuous), then the empirical distribution describes the density of *binned* data.

# In[ ]:




