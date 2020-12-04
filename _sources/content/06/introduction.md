# Missing Data

---

## Content Sumamry

This chapter explores in detail the different ways in which missing data appear and the statistical implications of handling it. Missing data occur in many different ways, which affects how the data should be used and interpreted.

The topics covered include:
* Identifying the mechanisms that explain *when* data are missing in terms of the data generating process,
* Understanding how to handle missing data -- whether it can safely ignored, replaced with an approximation (i.e. imputed), or modeled using assumptions based on domain expertise.

As with most reasoning about the unknown, the language developed in this chapter encourages critical thinking about the data in terms of what is known about the process that generates it. Generally, there is no 'correct way' of handling missing data; such concepts are rarely black-and-white and relies on reasoning about the severity of effect.

A detailed look at much of this subject is nicely covered in Van Buuren's [Flexible Imputation of Missing Data](https://stefvanbuuren.name/fimd/). While the book is aimed at an advanced audience, much the exposition is still quite accessible.

## Identifying Mechanisms of Missing Data

When assessing the ability of a dataset to work for an investigation into a problem, there are two important concepts that lurk behind the dataset:
1. The 'true model' is the phenomenon under investigation that one hopes to understand,
1. The 'data generating process' are the (multitude of) real world mechanisms that create the data that will be recorded (e.g. probabilistic, noisy)

![missing-mechanisms](imgs/missing-mech.png)

Ideally, a dataset well-represents whatever problem is being
investigated; this similarity is the foundation of drawing inferences
about actual events from data. However, what if the data is a poor
sample of events? Or what if the data recording process is flawed or
incomplete?

These problems create systematic biases in results drawn from the
data. However, no data generating process perfectly reflect the true
model, and data generating processes always most always produce noisy
data. The question isn't "is this dataset representative?", but rather
"is this dataset representative *enough*?" 

Data quality is measured on a spectrum and evaluated using knowledge
of the domain: "Does the description of the data look like what's
understood about the data generating process?"

Incomplete measurements result in missing data; this affects the
quality of the sample. Understanding how that missing data arises is
crucial to understanding the quality of a sample. As the observations
are missing, this understanding cannot be made using the dataset
itself. Instead, it involves reasoning about the data generating
process

**Example:** Suppose the owner of a company selling custom phone cases wants to understand how satisfied her customers are with their purchase. To do so, she sends a text message to all previous customers asking to reply, rating their purchase on a scale of 1-5. In this scenario,
1. The true model is the true opinion of every customer
2. The data generating process describes the detailed mechanisms under which customer's responded with an answer. This may include: different customer's satisfaction with the product *at the time* of receiving the message, their mood at the time, whether their schedule upon receiving the message made them too busy to reply, the receiver's cell-service at the time, etc...
3. The data consist of all replies, entered into the dataset.

The true model is only inferred through the data generating process, which is full of noise and random events -- some of which result in data not being recorded at all. Understanding and modeling how missing data come to be is a difficult but necessary part of inference from resulting data. In this case, it's likely that customer who *really like* their purchase are more likely to respond to the question if they were busy when they received the text message. This leads to *non-response* bias: ignoring the missing data leads to under-emphasizing customers with weak-feelings toward the product.
