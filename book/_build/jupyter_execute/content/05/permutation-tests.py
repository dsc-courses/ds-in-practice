#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')

import pandas as pd
import numpy as np


# # Permutation Tests
# ---
# 
# ## Permutation Tests Overview
# 
# See the **permutation test chapter** in [Inferential Thinking](https://www.inferentialthinking.com/chapters/12/Comparing_Two_Samples.html)

# ## Measuring similarity of distributions
# 
# Permutation tests assess the likelihood that two observed distributions originate from the same process. This assessment requires measuring the similarity of two distributions. Below are test-statistics used for this purpose.
# 
# In each of the following sections, assume `table` is a dataframe containing two distributions in tidy-form:
# 
# |Value|Group|
# |---|---|
# |2.23|A|
# |2.13|A|
# |5.23|B|
# |...|..|
# |2.34|B|

# ### Difference of means
# 
# If two distributions have different means, then they must be different distributions. This is a coarse measurement of similarity, as different distribution might have identical means.
# 
# To calculate the difference of means:

# In[ ]:


diff_of_means = (
    table
    .pivot_table(index='Group', aggfunc='mean')
    .diff()
    .dropna(how='all')
    .squeeze()
)

diff_of_means


# ### Total Variation Distance
# 
# If the distributions being compared are categorical, then taking a mean doesn't make sense. In this case, one can use the *total variation distance* (or TVD). This test statistic measures the difference in proportions represented by each category across the two distributions:

# In[ ]:


distr = (
    table
    .pivot_table(index='Value', columns='Group', aggfunc='size', fill_value=0)
    .apply(lambda x: x / x.sum())
)


# In[ ]:


tvd = (
    distr
    .diff(axis=1)
    .dropna(axis=1, how='all')
    .squeeze()
    .abs()
    .sum()
) / 2

tvd


# *Remark:* See [inferential thinking](https://www.inferentialthinking.com/chapters/11/2/Multiple_Categories.html#A-New-Statistic:-The-Distance-between-Two-Distributions) for more on this test statistic.
# 
# *Remark:* The *absolute* difference is necessary; the computation would always sum to zero without it. This is because a positive difference between categories must always be balanced by a negative difference in another category. This observation is a consequence of each distribution adding up to one!
# 
# *Remark:* The factor of 2 accounts for the above observation. For example, given two mutually exclusive distributions with only two categories, the TVD would be:
# 
# $$|1-0| + |0-1| = 1 + 1 = 2$$
# 
# However, as the second term is completely determined by the first, the more natural quantity is half of the total.

# ### Kolmogorov–Smirnov Test Statistic
# 
# While the difference in means measures the similarity of two quantitative distributions, it declares any two distributions the same as long as their means are the same. Having a measure of similarity that compares the overall shape of the distribution provides a more refined notion of similarity.
# 
# The Kolmogorov–Smirnov test statistic (KS test statistic) measures the similarity between two quantitative distributions, analogous to the TVD. It asks how different can two distributions be among all ranges of values.
# 
# The KS-statistic of two distributions is defined as the maximum distance between their empirical CDFs.
# 
# In the case of the DataFrame `table`, with values `A` and `B` in the `Group` column, the KS-statistic is:

# In[ ]:


# two sample
sampleA = table.loc[table['Group'] == 'A', 'Value']
sampleB = table.loc[table['Group'] == 'B', 'Value']

# two  CDFs
cdfA = sampleA.value_counts(normalize=True).sort_index().cumsum()
cdfB = sampleB.value_counts(normalize=True).sort_index().cumsum()

# calculate the maximum distance between CDFs
ks = (
    pd.concat([cdfA, cdfB], axis=1)
    .sort_index()
    .fillna(method='ffill')
    .fillna(0)
    .diff(axis=1)
    .iloc[:, -1]
    .abs()
    .max()
)

ks


# *Remark:* This method is both space-inefficient and logically opaque, but is a nice application of table manipulation. Walk through the methods step-by-step. How is this equivalent to using the function definition of the empirical CDFs in the last chapter? (Convince yourself!)

# *Remark:* The `scipy.stats` library provides the function `ks_2samp` that runs a KS-test whose results are similar to the permutation test described in this section. The function returns:
# * the value of the KS-statistic (as computed above), 
# * the p-value resulting from the hypothesis test with the null-hypothesis: "the two samples come from the same distribution."

# In[ ]:




