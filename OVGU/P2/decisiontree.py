'''
    Machine Learning
    Programming Assigment 2

    ID3 Algorithm

    Alejandro Rodriguez - 229564
    12.11.2019

    http://wwwiti.cs.uni-magdeburg.de/iti_dke/Lehre/Materialien/WS2019_2020/ML/p_sheet2.pdf
'''
# Supports column selection, but will crash when given a column that does not sort the tree perfectly

import sys
import pandas as pd
from math import log


def entropy(s, n=None):
    '''
        calculate Entropy function
        s: pd.Series of values whose entropy is to be calculated
        n: (optional) ammount of unique values in original y column,
            to make sure the log's base is consistent
    '''
    t = len(s)
    ct = s.value_counts()/t
    if len(ct)==1:
        return 0
    if n:
        ct = ct.append(pd.Series([0]*(n-s.nunique())))
    return -sum([0 if i==0 else i*log(i,len(ct)) for i in ct])

def ig(s, y, n):
    '''
        Information Gain function
        s: pd.Series to be evaluated
        y: pd.Series comparison column
        n: to be passed down to entropy function
    '''
    e = entropy(s)
    v = s.unique()
    for i in v:
        m = s==i
        e -= entropy(y[m], n)*len(y[m])/len(s)
    return e

def select_att(df, y, n):
    '''
        Select best attribute function
        Compares the info gain of all remaining columns of dataframe
        df: pd.DataFrame of x values
        y: pd.Series of answers
        n: to be passed down to entropy function
    '''
    ma = None
    v = -1
    for i in df:
        info = ig(df[i], y, n)
        if info > v:
            ma = i
            v = info
    if ma is None:
        assert False, "Tree cannot be perfectly sorted"
    return ma

def build_tree(df, y, n, lst):
    '''
        Build tree function
        Main recursive loop that creates the tree
        df: pd.Dataframe of x values
        y: pd.Series of answers
        n: to be passed down to entropy function
        lst: a reference to the current working branch
    '''
    global att_val
    lst.append(f'<node entropy="{float(entropy(y, n))}" feature="att{att_val[0]}" value="{att_val[1]}">')
    if y.nunique() == 1:
        lst[-1] += f'{y.iloc[0]}</node>'
        return
    a = select_att(df, y, n)
    lst.append([])
    s = df[a]
    v = s.unique()
    for i in v:
        att_val = (a, i)
        m = s==i
        df_ = df[m].drop(a, axis=1)
        build_tree(df_, y[m], n, lst[-1])
    lst.append('</node>')
    return

def build_tree_wrapper(df, ycol):
    '''
        Wrapper build tree function
        Call this to create the tree
        Sets up first layer and important variables
        df: pd.DataFrame of all values (including y)
        ycol: int that indicates which column will be considered the answer column
    '''
    # global variable of current attribute-value pair
    global att_val
    y = df[ycol]
    df_ = df.drop(ycol, axis=1)
    n = y.nunique()
    lst = [f'<tree entropy="{float(entropy(y))}">', [], '</tree>']
    a = select_att(df_, y, n)
    s = df[a]
    v = s.unique()
    for i in v:
        att_val = (a, i)
        m = s==i
        df_2 = df_[m].drop(a, axis=1)
        build_tree(df_2, y[m], n, lst[1])
    return lst

def main():
    # Setup
    v = sys.argv
    try:
        path = v[v.index('--data')+1]
        output = v[v.index('--output')+1]
    except:
        print('usage: python3 decisiontree.py --data <input.csv> --output <output.xml> [--ycol <int>]')
        return

    df = pd.read_csv(path, header=None)

    # Choose y column. defaults to the last column 
    try: 
        ycol = int(v[v.index('--ycol')+1])
    except:
        ycol = len(df.columns)-1

    lst = build_tree_wrapper(df, ycol)

    ans = str(lst).replace("'",'').replace(',','').replace('[','').replace("]",'').replace('> <','><')

    with open(output, 'w') as f:
        f.write(ans)

if __name__ == "__main__":
    main()