#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import seaborn as sns

pd.set_option('display.max_rows', 7)


# # Parsing HTML
# ---
# 
# The previous section described the process of collecting data over a network via HTTP requests. In particular, when scraping websites, these requests result in the collection of raw source data in the form of HTML.
# 
# HyperText Markup Language, or HTML, defines the structure of web content rendered on in a web browser. Thus, if a dataset requires extracting information from a website, the content must be found in, and retrieved from, the HTML.
# 
# Understanding how HTML visually represents a website helps write more robust data-extraction code. While HTML can be treated solely as text, using the structure helps the developer write code that easily adapts to both changing requirements in the data collection, as well as evolving website source.

# ## Anatomy of HTML
# 
# A website, represented in HTML, is described using the framework of the Document Object Model:
# 
# * The *HTML Document* is the totality of the markup that makes up a website.
# * The *Document Object Model* (DOM) is the internal representation of an HTML document as a *tree* structure.
# * An HTML *Element* is a subtree of the document. Visually, elements are regions of the webpage.
# * HTML *Tags* are markers that denote the start and end of an element.
# 
# **Example:** The basic website below, is represented as: the document rendered by the browser, the HTML source code, and the DOM tree.

# <img src="imgs/html.png">

# The root of document tree is the `<html>` element, which contains all the HTML source. The root typically contains two children: the head, containing metadata for the page, and the body, which contains the information rendered on the page itself. The body of this page consists of three portions: the header and two numbered sections, each of which includes a section header and text. Notice that all of these portions consist of subtrees themselves.

# ### Common tags
# 
# Tags define the visual appearance of a particular element. They typically fall into two different types:
# 1. tags defining structural elements (regions of the page), and
# 1. tags defining stylistic elements (e.g. formatting).
# 
# The table below summarizes the most useful tags:
# 
# |Structure Elements|Description|Head/Body Elements|Description|
# |---|---|---|---|
# |`<html>`|the document|`<p>`|the paragraph|
# |`<head>`|the header|`<h1>, <h2>, ...`|header(s)|
# |`<body>`|the body|`<img>`|images|
# |`<div>` |a logical division of the document|`<a>`| anchor (hyper-link)|
# |`<span>`|an *in-line* logical division|[MANY MORE](https://en.wikipedia.org/wiki/HTML_element)||
# 
# For data collection, `div` tags are particularly important: when collecting data on a collection of websites, selecting the subtree defined by a `div` tag that defines the area containing the data yields clearer, less error-prone parsing code.

# ## Parsing HTML with Beautiful Soup
# 
# The python library Beautiful Soup 4, or `bs4`, parses strings or file-like objects representing HTML. The constructor `BeautifulSoup(page)` returns  a `BeautifulSoup` object representing a *parsed document* as a tree-structure.
# 
# **Example:** The simple HTML document defined in the string below serves to illustrate the attributes and methods of the `BeautifulSoup` class:

# In[1]:


s = '''
<body>

  <div id="content">
    <h1>Heading here</h1>
    <p>My First paragraph</p>
    <p>My <em>second</em> paragraph</p>
  </div>
  
  <div id="nav">
    <ul>
      <li>item 1</li>
      <li>item 2</li>
      <li>item 3</li>
    </ul>
  </div>

</body>
'''


# The HTML can be rendered in notebooks using the `IPython.display` module:

# In[2]:


from IPython.display import HTML
HTML(s)


# Parsing the HTML string into a document with BeautifulSoup, resulting object can be explored:

# In[3]:


import bs4
soup = bs4.BeautifulSoup(s)
type(soup)


# The elements of a document can be retrieved by specifying a desired tag to the `find` or `find_all` method. For example, to retrieve all list elements in the document:

# In[4]:


list_items = soup.find_all('li')
list_items


# Each of the items in the resulting list are document elements; the text displayed in those elements may be retrieved with the `text` attribute:

# In[5]:


type(list_items[0])


# In[6]:


list_items[0].text


# To select the top-portion of the page, consisting of the *content*, one can select the `div` with `find`, using the `attrs` keyword to specify the desired `div` element:

# In[48]:


content = soup.find('div', attrs={'id': 'content'})
content


# The `find` method also supports querying elements based on text they contain. For example, to retrieve the text of every element that contains the text 'paragraph', pass a predicate function matching the text 'paragraph':

# In[65]:


soup.find_all(text=lambda x:'paragraph' in x)


# The entirety of any document tree may be traversed, depth-first, with the `descendants` method, which returns an iterator.

# In[ ]:


for elt in soup.descendants:
    process(elt)
    ...


# In[ ]:




