'''
    Machine Learning
    Programming Assigment 6

    k Means

    Alejandro Rodriguez - 229564
    Anjo Ditsche - 228052
    15.01.2020

    http://wwwiti.cs.uni-magdeburg.de/iti_dke/Lehre/Materialien/WS2019_2020/ML/p_sheet6.pdf
'''

import pandas as pd
import sys
import os

v = sys.argv
try:
    data = v[v.index('--data')+1]
    output = v[v.index('--output')+1]
except:
    print('usage: python3 nb.py --data <inputfile> --output <outputfile>')
    quit()
df = pd.read_csv(data, sep='\t', header=None)

X = df.drop(0, axis=1)
if len(X.columns)==3:
    X.drop(3, axis=1, inplace=True)
    
X.columns = ['x', 'y']

def dist(p1, p2):
    return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5


ans1 = ''
ans2 = ''

C = [(0,5), (0,4), (0,3)]

J_old=None
J = 0

while J_old != J:
    J_old = J
    J = 0
    klasse = [set(), set(), set()]
    
    for i in range(len(X)):
        ds = [dist(tuple(X.iloc[i]), C[j]) for j in range(len(C))]
        smallest = min(ds)
        closest = ds.index(smallest)    
        klasse[closest].add(i)

        J += smallest**2    

    if J_old != J:
        ans1 += f'{J}\n'
        ans2 += f'{C[0][0]},{C[0][1]}\t{C[1][0]},{C[1][1]}\t{C[2][0]},{C[2][1]}\n'

    for k in range(len(C)):
        sumx = sum(X['x'][i] for i in klasse[k])
        sumy = sum(X['y'][i] for i in klasse[k])
        C[k] = (sumx/len(klasse[k]), sumy/len(klasse[k]))


with open(os.path.join(output,f'{data.split(".")[0]}-Progr.tsv'), 'w') as f:
    f.write(ans1)

with open(os.path.join(output,f'{data.split(".")[0]}-Proto.tsv'), 'w') as f:
    f.write(ans2)