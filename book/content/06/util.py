import pandas as pd
import numpy as np


def ks(valcol, grpcol):

    val1, val2 = grpcol.unique()

    # two samples
    col1 = valcol.loc[grpcol == val1]
    col2 = valcol.loc[grpcol == val2]

    # two CDFs
    cdf1 = col1.value_counts(normalize=True).sort_index().cumsum()
    cdf2 = col2.value_counts(normalize=True).sort_index().cumsum()
    
    # KS: max difference between the two CDFs
    ks = (
        pd.concat([cdf1, cdf2], axis=1)
        .fillna(method='ffill')
        .fillna(0)
        .diff(axis=1)
        .iloc[:, -1]
        .abs()
        .max()
    )
    
    return ks


def tvd(valcol, grpcol):

    tvd = (
        pd.concat([
            valcol.rename('val'), 
            grpcol.rename('grp')
        ], axis=1)
        .pivot_table(
            index='val',
            columns='grp',
            aggfunc='size',
            fill_value=0)
        .apply(lambda x: x / x.sum())
        .diff(axis=1)
        .iloc[:, -1]
        .abs()
        .sum()
    ) / 2

    return tvd        
    
