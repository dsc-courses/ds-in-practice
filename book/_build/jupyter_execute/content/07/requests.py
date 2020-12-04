#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import seaborn as sns

pd.set_option('display.max_rows', 7)


# # HTTP Requests
# 
# ---
# 
# Data over the Internet are transmitted via *HTTP requests*. For example, a user browsing on a computer might want to watch a streaming video. By clicking on the link for the video, she makes a *request* for the video content. The request is sent via HTTP to a server (or computer) containing the video, which then *responds* with requested video.
# 
# HTTP requests follow a request-response protocol: a client (i.e. user) makes a request and waits for a response from the server.
# 
# <img src="imgs/req-resp.png">
# 
# Most communication between computers use this protocol. This includes:
# * Requesting the content of a webpage.
# * Requesting access to a secured user account.
# * One server requiring information from another (on an internal network).
# 
# A Data Scientist might explicitly leverage HTTP requests when:
# * Requesting updated data from an API (application programming interface).
# * Scraping data from a set of webpages.
# * Making the results of a prediction algorithm available to the public.
# 
# ## The anatomy of HTTP
# 
# The *hypertext transfer protocol* (HTTP) follows the 'request-response' message exchange pattern between clients and servers. 
# * The server is discoverable and always available.
# * The client initiates a request to a server, then waits for a response.
# 
# In order to communicate, the client and server must speak a common language.
# 
# ### HTTP Requests
# 
# The anatomy of an HTTP request consists of:
# 
# * The *request line*, consisting of a *method*, *resource*, and *protocol version*.
# * The request *headers*, that provides other information about the request, including context about who is making the request.
# * The request *body*, that provides any additional information needed to make the request (e.g. a user-name and password).
# 
# **Example:** The HTTP-request below makes a request from www.catphotos.net to render a photo of a cat in the browser:
# ```
# GET /photos/cat.png HTTP/1.1
# Host: www.catphotos.net
# User-agent: Mozilla/4.0
# Connection: close
# Accept-language: en
# 
# data
# ```
# 
# * The *method* is `GET`
# * The resource being requested is `/photos/cat.png`. This is also known as the request target.
# * The headers contain:
#     - The host server where the resource lives.
#     - The user-agent, which contains information on the browser in which the content will be displayed.
#     - Whether the connection should be kept open, or closed, after the response is sent.
#     - The preferred language of the client.
# * The body in the request may contain other data.
# 
# The method of an HTTP request lets the server know what is being requested of it. The most common requests are:
# 
# * `GET` requests data from a server (e.g. a webpage, a dataset).
# * `HEAD` is identical to `GET`, without returning a response body. This method allows a client to 'preview' a response.
# * `POST` sends data to a server (e.g. authentication, a file upload).
# * `DELETE` deletes the specified resource on the server.
# 
# This course will primarily request data using `GET`.
# 
# ### HTTP Responses
# 
# The anatomy of an HTTP response consists of:
# 
# * The *status line*, consisting of the *protocol version*, *status code*, and *status phrase*.
# * The *response headers*, that provide other information about the response, including information about the server that made the response, when the request was received, when the response was sent, etc.
# * The response *body* that includes the requested data.
# 
# **Example:** The HTTP response below corresponds to the request from a client to the server at www.catphotos.net to render a photo of a cat in the browser:
# 
# ```
# HTTP/1.1 200 OK
# Connection close
# Date: Thu, 06 Aug 2006 12:00:15 GMT
# Server: Apache/1.3.0 (Unix)
# Last-Modified: Mon, 22 Jun 2006 ...
# Content-Length: 682105
# Content-Type: image/png
# 
# iVBORw0KGgoAAAANSUhEUgAAANwAAA ...
# ```
# 
# * The *status* is `200`, meaning the request was successful.
# * The *status phrase* of a successful request is `OK`.
# * The *body* contains the data representing the image of the cat (encoded in base64, as text).
# 
# The status code allows the client to classify and assess the success of a response, clarifying what may have gone wrong when the response was unexpected. Status codes fall into the following categories, with a few examples of each:
# ```
# 1xx: Informational. (E.g. 100 Continue)
# 2xx: Success. (E.g. 200 Success)
# 3xx: Redirection.
# 4xx: Client Error. (E.g. 401 Bad Request; 403 Forbidden)
# 5xx: Server Error. (E.g. 500 Internal Server Error; 509 Bandwidth Limit Exceeded)
# ```

# ## Python's `requests` library
# 
# Python's `requests` library provides easy-to-use methods for making HTTP requests. The function `requests.get(url)` sends an HTTP request to a target `url`, returning an HTTP response:

# In[3]:


import requests
resp = requests.get('http://ucsd.edu')


# The parts of the response are available as attributes of the response object:

# In[23]:


# status (codes)
resp.status_code, resp.reason


# In[12]:


# Response Headers
resp.headers


# In[25]:


# Body (text of the webpage)
resp.text[:50]


# ## Responsible use of HTTP requests
# 
# Data collection tasks often involve sending HTTP requests *in bulk*. When sending a high volume of requests, one should always be careful with how the request logic is designed. Every request sent occupies the resources of the target server, *costing the owners of the server money* or even *bringing the website offline*. This observation mandates responsible use of HTTP requests, over both ethical and legal concerns.
# 
# **Example:** A [journalist](http://www.storybench.org/to-scrape-or-not-to-scrape-the-technical-and-ethical-challenges-of-collecting-data-off-the-web/), frustrated with a prison's failure to respond to a request for information, decided to take matters into their own hand by using HTTP requests to scape information straight from the prison's website. The requests were so numerous, the prison website shut down. During this time, the family members were unable to contact their loved ones in prison.
# 
# Guidelines for responsibly communicating via HTTP requests are:
# 
# * Be transparent about what is making the requests (e.g. by changing the headers).
# * Check the website or API for what is allowed (e.g. check `robots.txt` and terms of service).
# * Make requests as slowly as possible, avoiding repetitive request patterns.
# 
# Excessive use of HTTP requests may cause a server to block the client's IP Address, adversely affecting not only the irresponsible user, but those around them as well. Additionally, irresponsible use of rapid requests may cause the server to shut down (negatively affecting others) and bring about legal action.
# 
# 
# ### Designing `requests` code
# 
# Well designed code making HTTP requests will:
# * Responsibly limit request volume and velocity,
# * Be fault tolerant to failed requests,
# * Should ideally be *idempotent* (a successful request, made a second time, has no result).
# 
# To limit the volume of requests, avoid unnecessary calls to a target url. For example, when developing code to process the response data, request the data *once*, save it, and develop on the cached data:

# In[ ]:


resp = request.get(url)
body = resp.text
# develop `process` after saving body
process(body)


# When making many requests in succession, not every one is successful. For example, a request may fail because the target resource doesn't exist, or a website is limiting the rate at which the requests may be made. To handle unsuccessful requests, use:
# * response status codes to implement a 'try-retry' structure,
# * maintain a list of failed and successful client requests,
# * 'back-off' the velocity of requests when the server limits the request rates.
# 
# For example, making an HTTP request that retries upon failure (with so called "exponential back-off") is given in the function below. Such a function is useful when a target url attempts to limit the rate of requests from a client:

# In[35]:


def request_with_retries(url, max_retry, retry=0):
    '''returns a http reponse to the request to target url. If '''
    resp = requests.get(url)
    if resp.ok:
        return resp
    elif retry < max_retry:
        time.sleep(2**retry)
        return request_with_retries(url, max_retry, retry=(retry + 1))
    else:
        return resp


# Maintaining a simple log of successful and failed requests is useful to maintain a ledger of the current state of data collection:

# In[38]:


def request_with_logging(url, S, F):
    '''returns the request and updated lists of
    successful (S) and failed (F) target urls.
    It returns None if the request has already 
    been succesfully made.'''
    
    if url in S:
        return None, S, F
    
    resp = requests.get(url)
    if resp.ok:
        return resp, S + [url], F
    else:
        return resp, S, F + [url]


# ## Data APIs
# 
# Websites sometimes make data available especially for programmatic consumption through a *data API*, or application programming interface. Requesting data through a public API is the best way to collect data:
# * APIs are designed for programmatic consumption, giving implicit permission to access data.
# * The intentional design of APIs come with preferred best-practices for the given owner (permissions, rate-of-request).
# * APIs are typically orgranized, documented, and kept up-to-date.
# 
# A common type of API is a REST API, which returns Javascript Object Notation (JSON).

# In[ ]:


requests.get('http://www.getmydata.com/v1/articles').text


# In[ ]:


[
    {'articleId': 1, 'name': 'article name 1', 'text': '...'},
    {'articleId': 2, 'name': 'article name 2', 'text': '...'}
]


# *Remarks:*
# * The `v1` in the target url describes the version of the API-endpoint (for reference in documentation).
# * The output JSON is parsed as a list of python dictionaries.
# 
# One can also include complex queries in the data by including additional data into the target url; this process is called 'url encoding'. For example, one might request only science articles from 2015 by using the request:

# In[ ]:


requests.get('http://www.getmydata.com/v1/articles?date=2015&genre=science').text


# The query fields are usually found in the API documentation, corresponding to version `v1`.

# ## Scraping Basics
# 
# Screen scraping refers to programmatically browsing the web, downloading and parsing the source HTML of each webpage. Screen scraping is a useful data collection technique, as it's possible whenever a page is visible to the public eye. However, it should be considered as a last resort for a number of reasons:
# * Data derived from raw HTML and javascript are difficult to parse and clean.
# * The source of webpages changes often, making scraping code hard to maintain.
# * Scraping costs webpage owners money.
# 
# There are good reasons for scraping as well:
# * Not all scraping is unethical; sometimes it brings value to the webpage owner.
# * Search engines scrape webpages to index them and make them searchable.
# * Making an API takes work and the site owner has no qualms with moderate amount of data collection.
# 
# Best practices for scraping include keeping the rate of requests slow, respecting the policy laid out in the `robots.txt` file, and following the website's terms of service.
# 
# ### Designing scraping code
# 
# Collecting data from websites through scraping usually involves requesting and parsing HTML from many different websites. Such code should be separated into functionally different pieces. For example, the pieces of scraping code may break up into the following pieces:
# * Managing all urls requested,
# * Executing the request-response protocol,
# * Parsing the HTML to extract need data from the source.
# 
# In particular the HTML extraction should be developed on data that is only requested *once*. Don't make unneeded requests before data is even being collected!
# 
# **Example:** The simple scraper below makes simple attempts to implement responsible scraping concepts:
# * Logging to keep track of urls that have been successfully or unsuccessfully requested,
# * A `parse_html` function that processes a successful http-response.

# In[ ]:


class SimpleScraper(object):
    
    def __init__(self):
        self.success = []
        self.failure = []
        
    def make_request(self, url):
        if url in self.success:
            return None

        resp = request_with_retries(url, max_retries=10)
        if resp.ok:
            self.success.append(url)
            return resp
        else:
            self.failure.append(url)
            return resp
        
    def parse_html(resp):

        if not (resp and resp.ok):
            return None
        
        parsed = ... # code for parsing HTML
        return parsed
    
    def scrape(urls):
        for url in urls:
            resp = self.make_request(url)
            parsed = parse_html(resp)
            yield parsed

