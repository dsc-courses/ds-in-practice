# Bias and Variance

---

The previous section introduced the basic concepts and techniques for
building statistical models. This section focuses on how to assess
whether a given model is a *good* model.

In creating a model, one specifies a plausible model to explain a data
generating process and determines the most likely model, among all
models of that kind, that explains the observed data. Fitting such a model
raises a number of questions:

1. Does the fit model explain the observed data well?
1. Does the fit model explain the data generating process well?
1. Can *any* model explain the observed data well?
1. Is there another model specification that describes the data
   generating process more effectively?

These questions are answered using the concepts of bias and variance.

## Content Summary

* Evaluating the fit of a model.
* Cross-validation.
* Parameter searches.
