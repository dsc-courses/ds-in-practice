# Practical Data Science: Introduction

---

## Content Summary

This chapter introduces the term 'data science', describing the
over-arching methodologies that data scientists follow, while
illustrating these methods through a simple, non-trivial example on
the fairness of public salaries.

## Datasets

The primary dataset of this chapter is the San Diego Employee Salary
dataset from [Transparent
California](https://transparentcalifornia.com). This dataset includes
a complete record of city employees, their job titles, and a detailed
breakdown of their salaries in 2017. Other years are available at the
source.

### Remark on privacy and ethics

This dataset contains personal information on real people. On the one
hand, city employees are considered 'public figures' and the public
has a right to know how tax-dollars are being spent. On the other
hand, many of these individual employees have typical, modest jobs
outside of the public spotlight, and should reasonably be able to
expect privacy and the responsible use of their data.

Working this this dataset, one should exercise caution to:
* Understand the data only to answer the overarching questions that
  require the dataset's use. Do not needlessly 'poke around' the data;
  keep the investigation as 'anonymous as possible'.
* Proactively strip identifying information from the dataset whever
  it's not needed for analysis. In this case, only the first names of
  employees are kept.
* Do not propogate peoples information across the internet (e.g. on a
  blog post, or in the versioned project on GitHub). Just because data
  is public record, doesn't mean it should be the first result on a
  Google search for that individual!
  
