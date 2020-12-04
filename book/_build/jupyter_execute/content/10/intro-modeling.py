#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import seaborn as sns

pd.set_option('display.max_rows', 7)


# # Introduction to Modeling
# 
# ---
# 
# Statistical models attempt to capture relationships among properties of a phenomenon of interest. Understanding such a picture requires the following definitions:
# 
# * A **Data Generating Process** is a real-world phenomenon of interest, from which data are collected.
# * A **Model** is a theory, or explanation, of the data generating process.
# * A **Fit Model** is a particular instance of a model that is likely to explain a fixed dataset.
# 
# What makes a model good? (model/fit model). 
# 
# **Example:** Suppose a Data Scientist is interested in the relationship between the number of parking tickets given out and the weather.
# 
# * The data generating process is the process that generates the events that unfold in the real-world. The process of exactly who ends up with a parking ticket and what the weather was is impossible to observe directly and unfolds noisily.
# 
# * A model might posit that more parking tickets are issued when the weather is temperate (neither hot nor cold).
# 
# * A fit model would specify the quantitative relationship for a given dataset. For example, using a dataset of parking tickets and weather in San Diego, the data might suggest that a car is five times for likely to be issued a parking ticket when the weather is temperate. 
# 
# * A model fit on data from Minneapolis might specify a different quantity (e.g. three times more likely). However, the structure of the model remains the same.
# 
# ### What makes a model 'good'?
# 
# A model is fit to a dataset by finding the *most likely* parameters that explain the observed data under the given model. These parameters are found by minimizing a *loss function* -- typically some notion of 'error' or 'cost'.
# 
# A model is good if it effectively explains the phenomenon under investigation. This requires answering two questions:
# 1. Is the *model* choice reasonable? Does the structure of the model capture the general understanding of how the Data Generating Process behaves?
# 1. Does the fit model describe the data well? How small is the error?
# 
# The first question approaches the applicability of the model to new observations; the second question approaches the ability of a model to explain the observed data.

# ## Basic Definitions
# 
# What questions can models answer?
# 
# * Questions of *Prediction*: Given what I've seen, what is the most likely value I'll see in the future? Predictions forecast the most likely values of the data coming from the data generating process.
# * Questions of *(Statistical) Inference*: How likely is what I observed representative of the broader picture? Statistical Inference draws conclusions (with confidence) about the structure of the data generating process (*population*).
# 
# 
# ### Statistical Model: Inference
# 
# * A **statistical model** is a quantitative relationship between properties in observed data.
# * A *statistical model* is a function $S:X\to\mathbb{R}^n$ that measures properties of $X$.
# * **Example:** Is there a linear relationship between the heights of children and the height of their biological mother?
#     * $X = \texttt{mother_height} = \mathbb{R}$
#     * $Y = \texttt{child_height} = \mathbb{R}$
#     * $S$ is the correlation coefficient
#     
# Inference results in interpreting properties of the DGP from the parameters of the model (e.g. correlation).
# 
# ### Prediction Model: Regression
# 
# * **Regression Models** attempt to predict the most likely quantitative value associated to an observation (feature).
# * A *regressor* is a function $R:X\to \mathbb{R}$ that predicts the value $y\in \mathbb{R}$ of an observation $x\in X$.
# * **Example:** Given the heights of a child's parents, what is the height of their child?
#     * $X = (\texttt{father_height, mother_height}) = \mathbb{R}^2$
#     * $Y = \texttt{child_height} = \mathbb{R}$
#     * $C$ predicts child heights.
#     
# ### Prediction Model: Classification
# 
# * **Classification Models** attempt to predict the most likely *class* associated to an observation (feature).
#     - The *type* or *class* is a *nominal* attribute (e.g. 'YES' or 'NO').
# 
# * A *classifier* is a function $C:X\to Y$ that predicts whether an observation $x\in X$ belongs to class $y\in Y$.
# 
# * **Example:** Given product purchase attributes (item, price, age, state), can one predict whether the person was satisfied with their purchase?
#     - $X = \{(\texttt{item, price, age, state})\}$
#     - $Y = \{\texttt{'SATISFIED'}, \texttt{'NOT SATISFIED'}\}$
#     - $C$ predicts product satisfaction.
#     

# ## Modeling and Features
# 
# Building a statistical model requires quantitative input that (strongly) reflects the relationships between the variables of interest. A complex data generating process requires a complex modeling pipeline to represent the process. This may happen in one of two ways:
# * the features must encode a complex understanding of the data generating process, or
# * the proposed model must be capable of capturing complex phenomena.
# 
# For example, a linear model cannot capture nonlinear behavior. However an appropriately chosen feature transformation might turn a non-linear relationship into a linear one, making a linear model an effective choice. Without such a feature transformation, a more complex non-linear model would be required.
# 
# SEE LECTURE FOR EXAMPLES.

# In[ ]:




