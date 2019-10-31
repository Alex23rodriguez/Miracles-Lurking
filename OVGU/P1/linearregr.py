'''
    Machine Learning
    Programming Assigment 1

    Batch Linear Regression using Gradient Descent

    Alejandro Rodriguez - 229564
    31.10.2019

    http://wwwiti.cs.uni-magdeburg.de/iti_dke/Lehre/Materialien/WS2019_2020/ML/p_sheet1.pdf
'''

import sys
import pandas as pd

def iteration(w, df, y, lr):
    y_hat = (df*w).sum(axis=1)
    grad = df.mul(y-y_hat, axis=0).sum()
    return w+grad*lr


def main():
    # Setup
    v = sys.argv
    try:
        path = v[v.index('--data')+1]
        learning_rate = float(v[v.index('--learningRate')+1])
        threshold = float(v[v.index('--threshold')+1])
    except:
        print('usage: python3 linearregr.py --data <inputfile> --learningRate <float> --threshold <float>')
        return
    
    # Read file, setup dataframes and weights
    df_ = pd.read_csv(path, header=None)

    cols = df_.columns
    df = df_.drop(cols[-1], axis=1) # X matrix
    df.columns = [i+1 for i in range(len(cols)-1)]
    df.insert(0, 0, 1) # inserts a col of ones, named 0
    
    y = df_[len(cols)-1] # y vector
    w = pd.Series([0]*len(cols)) # weights

    # Main loop
    y_hat = (df*w).sum(axis=1)
    sse = (y-y_hat).apply(lambda x: x**2).sum() #sum of squared errors

    sse_old = 0
    it = 0
    while abs(sse-sse_old)>threshold:
        w_str = ''
        for i in w:
            w_str+='{0:.4f}'.format(i)+','
        w_str += '{0:.4f}'.format(sse)
        print(f'{it},{w_str}')

        it+=1
        w = iteration(w, df, y, learning_rate) # update weights
        y_hat = (df*w).sum(axis=1)
        sse_old = sse
        sse = (y-y_hat).apply(lambda x: x**2).sum()
    w_str = ''
    for i in w:
        w_str+='{0:.4f}'.format(i)+','
    w_str += '{0:.4f}'.format(sse)
    print(f'{it},{w_str}')


if __name__ == "__main__":
    main()