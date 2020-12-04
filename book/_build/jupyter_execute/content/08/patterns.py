#!/usr/bin/env python
# coding: utf-8

# In[11]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import seaborn as sns

pd.set_option('display.max_rows', 7)

jobs = pd.read_csv('../01/data/san-diego-2017.csv', usecols=['Job Title'])
idx = jobs.sample(frac=0.3).index
jobs.loc[idx, 'Job Title'] = jobs.loc[idx, 'Job Title'].str.lower()


# # Text Data
# 
# ---
# 
# Once data are collected and transformed into a tabular format, with observations and attributes, the individual entries are often raw text. Initially, these text fields contain informations that are not quantitatively usable. This chapter covers extraction of information from text, resulting in a table that amenable to study using the techniques from the first part of the book.

# ## Pattern Matching
# 
# An effective, simple approach to extracting useful information from text is to find patterns that correlate with the concept being measured.
# 
# **Example:** The table `jobs` below contains the job title of every San Diego city employee in 2017. In chapter 1, the investigation into the salaries finished with the question: 
# > When controlling for 'job type', do women makes significantly less than their contemporaries?
# 
# However, the 'Job Title' field in the dataset is messy. Many related jobs are described in different ways; most job titles are distinct in text, even if their are similar in reality. When should two jobs be considered of the same type?

# In[12]:


jobs


# The empirical distribution of 'Job Title' meaningfully differentiate between different jobs:

# In[16]:


jobs['Job Title'].value_counts(normalize=True)


# ### Pattern matching: a general approach
# One approach to extracting information from text fields, is to 'measure the text' for useful information. For example, in the table of job titles:
# * How many employees do police work?
# * How many employees work in library-related work?
# * How many employees manage other people?
# 
# Approaching question like these follows a simple procedure:
# 1. Choose an initial pattern on which to match,
# 2. Assess whether the pattern is too narrow or broad:
#     - Examine at non-matching observations to assess if the pattern misses individuals,
#     - Examine the matched observations to assess if the patter captures unintended individuals
# 3. Make hypotheses and generalizations about what the text data look like, and test them.
# 
# Notice that the correctness of such an intuitive pattern matching approach is not verifiable! The most one can hope for is *falsification*: an accrual of evidence that the pattern captures the indended concept.

# **Example:** The question of how many employees do police work is one of the easier questions to ask of the `jobs` dataset. As seen from the empirical distribution of job titles, both 'Police Officer' and 'police officer' are common job titles. Ignoring the case of the words, a first reasonable pattern to consider is `police`.
# 
# The Series method `contains` in the `str` namespace performs matching on regular-expressions, returning a boolean array:

# In[37]:


contains_police = jobs['Job Title'].str.contains('police', case=False)
jobs.loc[contains_police, 'Job Title'].value_counts()


# This naive pattern returns approximately 2000 job titles relating to police. Did this pattern miss other police related jobs? The dataset contains abbreviations, so perhaps likely abbreviations of 'police' should also be tried:

# In[40]:


contains_pol = jobs['Job Title'].str.contains('pol', case=False)
jobs.loc[contains_pol & ~contains_police, 'Job Title'].value_counts()


# Eight possible observations related to police work are returned. If 'pol clrk' stands for 'Police Clerk', this more general pattern is better to use; outside research is needed to answer this question.
# 
# Are there job titles in the area of police work that might contain the word 'police'? A next step would be to propose new patterns that might capture police-related work. This may be approached via:
# * choosing police-related words that commonly appear in the job titles what contained the initial pattern, or
# * researching the domain and generating a list of keywords by hand.
# 
# For example, 'Police Dispatcher' appears in the pattern matching above. Does 'Dispatch' appear more broadly?

# In[41]:


contains_dispatch = jobs['Job Title'].str.contains('dispatch', case=False)
jobs.loc[contains_dispatch, 'Job Title'].value_counts()


# Other instances of 'Dispatcher' appear, including 'fire dispatch' and 'public works dispatch'. These observations raise the question of whether the generic 'Dispatcher' titles are police related or not. More domain research might be necessary.
# 
# Another possibility might be to search for the term 'Crime' in the job titles, which would almost certainly be police-related:

# In[42]:


contains_crime = jobs['Job Title'].str.contains('Crime', case=False)
jobs.loc[contains_crime, 'Job Title'].value_counts()


# This additional pattern match results in new police-related job titles. These patterns can be combined using a regular-expression:

# In[43]:


police_jobs = jobs['Job Title'].str.contains('pol|crime', case=False)
jobs.loc[police_jobs, 'Job Title'].value_counts()


# Of course, this pattern is likely not exhaustive; it is mere better than the initial pattern. This process must continue until the results are good enough to use.

# ## Canonicalization 
# 
# In the job titles dataset, the individual job titles often represented the same job title in different ways. For example, the job of a police officer was represented both in lower-case ('police officer') and upper-case ('Police Officer'). Transforming these different representations into a single form helps simplify the difficult process of finding appropriate patterns. This process is called *canonicalization*.
# 
# Datasets often have inconsistencies:
# * Some text might contain upper-case letters, while others are lower-case.
# * Some text may contain abbreviations. further, the abbreviations may not be consistent.
# * Punctuation may be used inconsistently.
# * Text may contains superfluous information.
# 
# In each of these examples, these inconsistencies require developing more sophisticated patterns for extracting the needed information from the text. Taking care of these inconsistencies *before* attempting to extract information simplifies the process. 
# 
# Canonicalization of text content refers to a function that chooses a standard form in which to represent each value.

# **Example:** Canonicalizing the job titles involves dealing with all of the inconsistencies listed above. In order of increasing difficulty, the canonicalizing job title will handing the following issues:
# 
# 1. handle mixed cases by transforming all characters to lower-case,
# 1. handle inconsistent use of punctuation by removing punctuation,
# 1. handle abbreviations by matching them to known words.
# 
# 
# **Inconsistent Case**. The method `lower` transforms the case of the characters in the 'Job Title' columns:

# In[44]:


jobs['Job Title'].str.lower()


# **Punctuation and Non-Alphanumeric Characters**.
# 
# Carefully replacing the punctuation involves understanding what punctuation is used and whether the existing punctuation is necessary information to keep.
# 
# Selecting job titles that contain non-alphanumeric characters, shows a number of usages of special characters:
# * The `&` symbol represents the word `and` (and that meaning should not be lost),
# * The `/` symbol separates two words without a space (and so should be replaced with a space),
# * `-` separates words *with* spaces, and should be replaced without spaces.

# In[47]:


jobs[jobs['Job Title'].str.contains('[^A-Za-z0-9\s]')]


# As 1000 observations are too many to look at by hand, use the `extract` method to create a full list of non-alphanumeric characters:

# In[62]:


(
    jobs['Job Title']
    .str.extractall('([^A-Za-z0-9\s])') # returns a multi-index for > 1 match
    .dropna()
    .reset_index(drop=True)
    .squeeze()
    .value_counts()
)


# The totality of non-alphanumeric characters includes `- ( ) / & ' ,`, each of which should be handled differently.

# In[65]:


(
    jobs['Job Title']
    .str.lower()  # lower case
    .str.replace('&', 'and') # replace '&' with 'and'
    .str.replace('[^A-Za-z0-9\s]', ' ') # replace all other punctuation with space
    .str.replace('\s+', ' ') # collapse multiple whitespace down to one.
)


# **Abbreviations**.
# 
# Lastly, job titles have different abbreviations that are used inconsistently across the dataset. For example, 'analyst' job titles may either be represented as either 'Analyst' or 'Anlyst':

# In[75]:


jobs[jobs['Job Title'].str.lower().str.contains('analyst|anlyst')]


# Canonicalizing abbreviations is a harder task and the way with which it's dealt depends on what's being done with the data. A few things to keep in mind:
# * Is it import to understand *what* the abbreviation means, or just that it's consistently used throughout the dataset?
# * To find instances of abbreviations that might not be used consistently
#     - use a dictionary to find non-words, 
#     - use edit-distance functions to find small variations between words that might have similar meaning.

# In[ ]:




