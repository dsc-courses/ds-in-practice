#!/usr/bin/env python
# coding: utf-8

# In[6]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import seaborn as sns

pd.set_option('display.max_rows', 7)

from io import StringIO
s = StringIO('''phrase
a two bedroom apartment with washer and dryer
a two bedroom house with a washer hookup
a three bedroom house with a fireplace''')
apts = pd.read_csv(s)


# # Information Extraction from Text
# 
# ---
# 
# ## The Limits of Pattern Matching
# 
# Pattern matching is an information extraction on technique on text that offer a way to introduce oneself to raw text. However, pattern matching has its limits:
# * How are common patterns proposed and found?
# * The ad-hoc development and exploration of information extraction with patterns does not scale to large amounts of text.
# * Assessing the efficacy of a pattern to effectively extract information does not scale well past visual inspection. When an large-scale analysis is done, it's time consuming and likely not applicable to similar patterns.
# 
# To move beyond these limits, one must approach information extraction from text more methodically, using a quantitative approach borrowed from math and statistics.

# ## Measuring Similarity between Text
# 
# Given two snippets of text, are they similar? At heart, this question is asking for *a distance measure* between words and phrases. While there are many such measures of distance, each capturing different aspects of the information in text, they all require a common setup: how should the text be embedded into a quantitative (e.g. Euclidean) space?

# ### The "Bag of Words" model
# 
# Consider the following listings for housing rentals:
# 
# |phrase|
# |---|
# |a two bedroom apartment with washer and dryer|
# |a two bedroom house with a washer hookup|
# |a three bedroom house with a fireplace|
# 
# Since a listing is made up of a collection of amenities, two listings might be considered similar if they share similar words. That is: 
# * `two bedroom apartment with washer and dryer` and 
# * `a two bedroom house with a washer hookup` 
# 
# share five words (`a`, `two`, `bedroom`, `with`, `washer`). This matching can be turned into a measure of similarity in a number of ways:
# * Using the raw number itself as a measure, where larger is more similar (e.g. the similarity is 5).
# * Using the proportion of possible matches, where 1 is the most similar (e.g. 5/7 words were matches).
# * Computing the empirical distribution of each phrase and using the Total Variation Distance (TVD).
# 
# *Remark:* The first measure is not normalized, which is may be a good property. The likelihood that two very long phrases are similar is much smaller than two short phrases.
# 
# The 'Bag of Words' model sets up this notion of similarity by embedding the words into a *vector space*. This vector space embedding allows one to easily compute different notions of similar and understand the distribution of words among the phrases in a dataset.
# 
# The **Bag of Words embedding** of a list of phrases is representation of the counts of words in each phrase in vector space whose basis consists of all words appearing in the dataset.
# 
# **Example:** The Bag of Words embedding of the three row table of housing listings transforms the phrases into a 12-dimensional vector space:
# 
# |a|two|three|bedroom|apartment|house|with|washer|hookup|and|dryer|fireplace|
# |---|---|---|---|---|---|---|---|---|---|---|---|
# |1|1|0|1|1|0|1|1|0|1|1|0|
# |2|1|0|1|0|1|1|1|1|0|0|0|
# |2|0|1|1|0|1|1|0|0|0|0|1|
# 
# *Remark:* Notice that the bag-of-words embedding is nothing but systematically pattern matching: for each word in the dataset, count the number of occurrences of each words in each phrase. However, the Bag of Words embedding doesn't know anything about the *meaning* of each words. The embedding works under the assumption that two phrases are similar if they share many of the same words.
# 
# Using a Bag of Words embedding, the **similarity** of two phrases can be measured using notions of similarity in the Bag of Words vector space. Under the Bag of Word embedding:
# * The similarity of two phrases is proportional to the dot product of the Bag of Words vectors.
# * The similarity of two phrases is given by the *cosine similarity* of the Bag of Words vectors: 
# 
# $$dist(v, w) = 1 - \cos(\theta) = 1 - \frac{v \cdot w}{|v||w|}$$
# 
# **Example:** In the above housing listings, which listings are most similar under the Bag of Words model?
# 
# |phrase pair|dot product|cosine similarity|
# |---|---|---|
# |0,1|2+1+0+1+0+0+1+1+0+0+0+0 = 6|0.33|
# |0,2|2+0+0+1+0+1+1+0+0+0+0+0 = 5|0.41|
# |1,2|4+0+0+1+0+1+1+0+0+0+0+0 = 7|0.26|
# 
# As measured by the cosine similarity, the most similar phrase pair is the middle and last phrases.

# *Remark:* The Bag of Words model has downsides, already seen in this example:
# * The model treats all words as *equally important*. For exaample, the word 'a' and the word 'apartment' are given equal weight.
# * The model treats words without context. The phrases 'I own a dog' and 'I don't own a dog' are similar in the bag of words model.
# 
# However, the perspective of the Bag of Words model is powerful. These downsides can be handled with straightforward improvements and modifications.

# ## Measures of Relevancy 
# 
# A shortcoming of the naive Bag of Words model is that it treats every words equally. This treatment can cause two phrases with similar content to appear dissimilar because of 'superfluous' words. What are ways to extract 'the most relevant' words from a phrase?
# 
# ### Term Frequency, Inverse Document Frequency (TF-IDF)
# 
# An intuitive heuristic for extracting the most relevant term of a phrase is *Term Frequency, Inverse Document Frequency* or TF-IDF. This method attempts to answer the question "how much does a given word summarize a phrase?".  TF-IDF attempts to balance the importance of a word in a given document with the uniqueness the word has to the document.
# 
# Suppose a dataset consists of a *collection of documents* $D$.
# 
# * The *term frequency* of a word $t$ in a document $d$, denoted ${\rm tf}(t,d)$, is the likelihood of the term appearing in the document:
# 
# $${\rm tf}(t, d) = \frac{\rm{number\: of\: times\: t\: appears\: in\: document\: d}}{\rm{total\: number\: of\: terms\: in\: document\: d}} $$
# 
# * The *inverse document frequency* of a word $t$ in a collection of documents $D$, denoted ${\rm idf}(t,d)$ is:
# 
# $${\rm idf}(t) = \log\left(\frac{\rm{total\: number\: of\: documents}}{\rm{number\: of\: documents\: in\: which\: t\: appears}}\right)$$
# 
# * The *tf-idf* of a term $t$ in document $d$ is given by the product: 
# 
# $${\rm tfidf}(t,d) = {\rm tf}(t,d) \cdot {\rm idf}(t)$$
# 
# *Remark:* There are different, related, ways of computing this quantity. As this method is a heuristic, there isn't a 'correct' formula with a probabilistic interpretation.

# Notice that if a term appears in *every* document in the collection, the ${\rm idf(t, d)}$ is zero. This fits the intuition that very common words should not be considered relevant to the information contained in a document.

# **Example:** The TF-IDF of the word `two` in the first apartment listing is computed as follows:
# 
# $${\rm tf}(\texttt{two}, \texttt{listing0}) = \frac{1}{8}$$
# $${\rm idf}(\texttt{two}) = \log(\frac{3}{2})$$
# $${\rm tf}(\texttt{two}, \texttt{listing0})\cdot {\rm idf}(\texttt{two}) = \frac{1}{8}\log(\frac{3}{2})$$

# This quantity naturally defines the most relevant words for a given document: the term with the highest TF-IDF for a given document *best summarizes* the document.

# **Example:** Computing the most relevant term for each listing is illustrated in the following code:

# In[7]:


apts


# While slower than leveraging optimized libraries, the Bag of Words embedding can be easily implemented with Pandas:

# In[19]:


bow = (
    apts['phrase']
    .str.split()
    .apply(lambda x:pd.Series(x).value_counts())
)
bow


# The term frequencies of each word, in each document, is represented in a matrix labeled by words and document number. Each word in a given document is part of an empirical distribution for that document:

# In[20]:


term_frequencies = bow.apply(lambda x:x / x.sum(), axis=1)
term_frequencies


# The inverse document frequency is calculated using a straightforward count of non-null entries:

# In[24]:


tot = bow.shape[0]
inverse_document_frequencies = np.log(tot / bow.count())
inverse_document_frequencies


# The resulting tfidf matrix represents the term frequency, inverse document of frequency of every term in every document:

# In[29]:


tfidf = term_frequencies * inverse_document_frequencies
tfidf


# The most relevant word in each document corresponds to the word with the largest tfidf in that document:

# In[28]:


tfidf.idxmax(axis=1)


# *Remark:* Why are these words good summaries of each listing? In what ways are they *not* good summaries? 
# 
# *Remark:* These words were not the only correct answers; search for ties in the table.
