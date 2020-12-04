#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')

import pandas as pd
import numpy as np


# # Definitions: mechanisms of missing data 
# 
# This section details the different ways in which the observations of a given measurement might be missing from a dataset. The type of missingness influences the correctness of conclusions drawn from the data; identifying how data are missing points toward the best way of handling such data.
# 
# Missing data falls into three general categories detailed below; each dictates different ways of reasoning with the data, which may depend on the task at hand.
# 
# **Notation** Each of the mechanisms have precise definitions using the language of probability. Each such definition $Y$ is an ideal 'dataset' reflecting the "true model", while $Y_{obs}, Y_{mis}$ denote the observed and missing data respectively. Effects that influence the missingness of the data, that are independent of the values of the data, are denoted by $\Psi$.
# 
# ## Missing by Design
# 
# An attribute has observations missing by design if all missing data is missing deliberately (i.e. built into the design of the data collection process). This occurs when
# 1. a value can be deterministically recovered from other attributes in the data, or
# 1. a value for an attribution does not make sense for a given observation (in which case, there's no "true value").
# 
# **Definition:** An attribute of a dataset $Y$ is *missing by design* if the missingness of the attribute can be determined logically from the *other* attributes of the observed dataset. That is, there is a boolean function of the remainder of the observed dataset that returns `True` if and only if the value of that attribute is missing.
# 
# **Example:** Suppose a contains information on household makeup and the resulting table consists of the number of people living in a household and the ages of the four oldest people:
# 
# |Number of People|Age1|Age2|Age3|Age4|
# |---|---|---|---|---|
# |5|27|25|25|45|
# |3|50|47|18|`NaN`|
# |4|30|34|29|29|
# |2|29|35|`NaN`|`NaN`|
# |...|...|...|...|
# 
# The age columns are missing by design, as they are missing in `AgeN` exactly when `N` is less than the number of people living in the household. There is no "true value" for the missing values.
# 
# *Remark:* This dataset isn't *tidy*. The observations correspond to households; reshaping the table so that the observations represent individuals who live in a household results in a much easier dataset to work with.

# ## Ignorable Missing Data
# 
# An attribute contains *ignorable missing data* if the missing values are explainable from the observed data itself. Ignorable missing data can be explained by a probability model, handled appropriately, and *ultimately* ignored. It doesn't mean you can ignore the problem entirely!
# 
# The question of "whether the missing values of an attribute are explainable from observed attributes a given dataset" is an *assumption on the data generating process*. The data scientist must assess what she knows of the dataset, and whether it's likely the missing values are a result of measurements contained in the dataset, or influenced by unknown, external factors.
# 
# ### Missing Completely at Random
# 
# Data missing completely at random is the most simple way that data can be missing. In this case, the likelihood an observation is missing is unconditionally uniform across the rows of the dataset; it doesn't depend on any values in dataset.
# 
# **Definition:** Data are missing completely at random if the likelihood a value is missing doesn't depend on any values in the dataset:
# 
# $$Pr({\rm data\ is\ present\ } | Y_{obs}, Y_{mis}, \Psi) = Pr({\rm data\ is\ present\ } |\ \Psi)$$
# 
# The following example, while contrived, illustrates the concept well:
# 
# **Example:** A group of columns for a dataset are being manually entered into a spreadsheet from paper records. The paper records have no particular order ("thoroughly shuffled") and are grouped into stacks of 100 people each. Suppose one of the stacks of records is misplaced; these columns would then be missing completely at random in the resulting dataset. There is no association between any which records have missing data and what is contained in those records.
# 
# **Example:** Data collection for surveys can be an expensive and time consuming endeavor. One cost-reducing technique is to administer an initial survey by mail to a large population, while randomly choosing a subset of the population for in-person follow-up interviews. Suppose that this follow-up population was selected with a coin-flip. In this case, the follow-up attributes in the completed (joined) survey data is *missing completely at random*. The follow-up questions were only answered for a random subset of the larger population, independent of any of their answers to the original survey.
# 
# *Remark:* The proportion of the respondents that received a follow-up (i.e. the bias of the 'coin-flip') is captured in the parameter $\Psi$.
# 
# ### Missing at Random
# 
# When an attribute in a dataset is missing at random, the missing values can be estimated using the values in the observed data. The 'Random' in the term 'Missing at Random' refers to the fact that the estimates are a result of a probability model derived from the observed data. In this more general case, the missing values may depend on the values observed in other attributes.
# 
# **Definition:** Data are missing at random if the likelihood a value is missing depends only the observed data itself:
# 
# $$Pr({\rm data\ is\ present\ } | Y_{obs}, Y_{mis}, \Psi) = Pr({\rm data\ is\ present\ } |\ Y_{obs}, \Psi)$$
# 
# *Remark:* That is, data missing at random is actually missing completely at random, conditional on $Y_{obs}$. In this way, one can reduce this missingness mechanism to the most simple one.
# 
# **Example:** Given a dataset consisting of the lab results, over five months, of cancer patients enrolled in a drug trial:
# 
# |Patient ID|Lab1|Lab2|Lab3|Lab4|Lab5|
# |---|---|---|---|---|---|
# |1|12.3|14.4|13.4|13.7|14.1|
# |2|12.6|12.4|12.9|13.0|13.5|
# |3|12.9|13.2|12.0|7.7|`NaN`|
# |4|13.3|12.4|12.4|13.1|12.9|
# 
# The results for `Lab5` might be missing at random conditional on `Lab4`, as very sick patients (who will often have low lab results) might not continue in the study.
# 
# **Example:** In a survey of worker profiles, people are asked questions about their employment, like what sort of industry they work in, as well as their annual salary. It's reasonable to argue that annual salaries might be missing at random conditional on industry. Those working for hourly wages are less likely to know their salary, thus those industries with a large number of hourly employees (e.g. the service industry) will more likely be associated to respondents with no recorded annual salary.
# 
# In this case, the observed dataset is likely over-estimating the true salaries of the respondent population; the sample of salaries is not representative. However, all is not lost, as this over-estimation is still explainable from the observed data.

# ## Non-ignorable Missing Data
# 
# An attribute contains *non-ignorable missing data* when the likelihood an observation is missing depends on the actual (unreported) value. This phenomenon cannot be determined from the observed data; it must be reasoned from domain expertise on the data-generating process. It's called *non-ignorable* as the missing-values must be modeled from external knowledge, as opposed to being explainable by the dataset itself.
# 
# **Definition:** An attribute contains *non-ignorable missing data* if 
# 
# $$Pr({\rm data\ is\ present\ }| Y_{obs}, Y_{mis}, \Psi)$$
# 
# does not simplify. That is, the missingness is dependent on the missing value itself.
# 
# **Example:** Given a dataset of patients scheduled for an employee drug test and their results, the results column likely exhibits non-ignorable missingness: those who would likely fail their drug test will be more likely to show up to their appointment.
# 
# **Example:** In the survey of worker profiles, it's likely that a respondent is less likely to report their annual salary when their income is high. This is an example of *non-ignorable missing data* as the likelihood an observation has a missing value (annual income) is associated to the missing value itself (high incomes are more likely missing).
# 
# In this case, collecting other attributes might explain some of this association and weaken the non-ignorable effects. For example, collecting geographical, educational, and family information might capture some of the reasons respondents with higher incomes more likely result in non-response.
