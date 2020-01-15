'''
    Machine Learning
    Programming Assigment 5

    k Nearest Neighbors

    Alejandro Rodriguez - 229564
    Anjo Ditsche - 228052
    12.01.2020

    http://wwwiti.cs.uni-magdeburg.de/iti_dke/Lehre/Materialien/WS2019_2020/ML/p_sheet5.pdf
'''

import pandas as pd
import sys

v = sys.argv
try:
    data = v[v.index('--data')+1]
    output = v[v.index('--output')+1]
except:
    print('usage: python3 nb.py --data <inputfile> --output <outputfile>')
    quit()
df = pd.read_csv(data, sep='\t', header=None)

def dist(a, b):
    return ((a[0]-b[0])**2 + (a[1]-b[1])**2 )**0.5

def calc_weight(di, d1, dk):
    if d1==dk:
        return 1
    return (dk - di)/(dk - d1)

#IB2 algorithm using kNN
def guess(cb, x):    
    dists = []
    for coord in cb:
        d = dist(x, coord)
        dists.append((d,cb[coord]))
    
    dists.sort()
    
    sumA = 0
    sumB = 0
    
    k = min(K, len(dists))
    
    min_dist = dists[0][0]
    max_dist = dists[k-1][0]
    
    for elem in dists[:k]:
        d, klasse = elem
        
        weight = calc_weight(d, min_dist, max_dist)
        
        if klasse == 'A':
            sumA += weight
        else:
            sumB += weight
            
    return 'A' if sumA > sumB else 'B'


y = df[0]
X = df.drop(0, axis=1)

ans = ''

for K in (2,4,6,8,10):
    # training
    cb = {tuple(X.iloc[0]) : y.iloc[0]}
    for i in range(1, len(X)):
        x = tuple(X.iloc[i])
        if guess(cb, x) != y[i]:
            cb[x] =  y[i]
    
    if K == 4:
        cb4 = cb

    # classif
    missclassif = 0

    for i in range(len(X)):
        if guess(cb, tuple(X.iloc[i])) != y[i]:
            missclassif += 1
    
    ans += f'{missclassif}\t'

ans += '\n'
for x, y in cb4:
    ans+=f'{cb4[(x,y)]}\t{round(x,6)}\t{round(y, 6)}\n'

with open(output, 'w') as f:
    f.write(ans)