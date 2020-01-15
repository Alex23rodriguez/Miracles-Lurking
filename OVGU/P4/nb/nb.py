'''
    Machine Learning
    Programming Assigment 4

    Naïve Bayes

    Alejandro Rodriguez - 229564
    Anjo Ditsche - 228052
    15.12.2019

    http://wwwiti.cs.uni-magdeburg.de/iti_dke/Lehre/Materialien/WS2019_2020/ML/p_sheet4.pdf
'''

from math import pi
from math import e as exp
import pandas as pd
import sys

v = sys.argv
try:
    path = v[v.index('--data')+1]
    output = v[v.index('--output')+1]
except:
    print('usage: python3 nb.py --data <inputfile> --output <outputfile>')
    quit()

df = pd.read_csv(path, sep='\t', header=None)
y = df[0]
X = df.drop(0, axis=1)

vc = y.value_counts()
prob = {}
means = {}
vrs = {}

for c in y.unique():
    prob[c] = vc[c]/len(y)
    means[c] = {}
    vrs[c] = {}
    for i in [1,2]:
        means[c][i] = X[i][y==c].mean()
        vrs[c][i] = X[i][y==c].var()

def norm(x, a, c):
    up = -(x-means[c][a])**2/(2*vrs[c][a])
    return exp**up / (2*pi* vrs[c][a])**0.5

def calcLike(z, c):
    return prob[c] * norm(X[1][z], 1, c) * norm(X[2][z], 2, c)

def chooseClass(z):
    if calcLike(z, 'A') > calcLike(z, 'B'):
        return 'A'
    return 'B'

miss = 0
for i in range(len(y)):
    if chooseClass(i) != y[i]:
        miss+=1

s = ''
for c in ['A', 'B']:
    s += f'{means[c][1]}\t{vrs[c][1]}\t{means[c][2]}\t{vrs[c][2]}\t{prob[c]}\n'
s += str(miss)

with open(output, 'w') as f:
    f.write(s)
