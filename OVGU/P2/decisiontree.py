'''
    Machine Learning
    Programming Assigment 2

    ID3 Algorith

    Alejandro Rodriguez - 229564
    12.11.2019

    http://wwwiti.cs.uni-magdeburg.de/iti_dke/Lehre/Materialien/WS2019_2020/ML/p_sheet2.pdf
'''

import sys
import pandas as pd
from math import log

# global variable of current attribute-value pair
att_val = (None, None)

def entropy(s, n=None):
    '''
        Entropy function
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
        df: pd.DataFrame of all values
    '''
    ma = 0
    v = 0
    for i in df.columns[:-1]:
        info = ig(df[i], y, n)
        #print(info)
        if info > v:
            ma = i
            v = info
    return ma

def build_tree(df, ycol, n, lst):
    global att_val
    y = df[ycol]
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
        build_tree(df_, ycol, n, lst[-1])
    lst.append('</node>')
    return lst

def main():
    # Setup
    v = sys.argv
    try:
        path = v[v.index('--data')+1]
        output = v[v.index('--output')+1]
    except:
        print('usage: python3 decisiontree.py --data <input.csv> --output <output.xml>')
        return

    df = pd.read_csv(path, header=None)

    ycol = len(df.columns)-1
    lst = build_tree(df, ycol, df[ycol].nunique(), [])
    i = lst[0].index(' feat')
    lst[0] = f'<tree {lst[0][6:i]}>'
    lst[2] = '</tree>'

    ans = str(lst).replace("'",'').replace(',','').replace('[','').replace("]",'').replace('> <','><')

    with open(output, 'w') as f:
        f.write(ans)

if __name__ == "__main__":
    main()