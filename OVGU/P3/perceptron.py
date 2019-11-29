import pandas as pd
import sys

v = sys.argv
try:
    path = v[v.index('--data')+1]
    output = v[v.index('--output')+1]
except:
    print('usage: python3 perceptron.py --data <inputfile> --output <outputfile>')
    quit()

df = pd.read_csv(path, sep='\t', header=None)
y = df[0].map(lambda x: 1 if x=='A' else 0)
X = df.drop(0, axis=1) 
X.insert(0, 0, 1) # inserts a col of ones, named 0

x = ""
w = pd.Series([0,0,0])

for _ in range(101):
    y_hat = (X*w).sum(axis=1).map(lambda x: 1 if x>0 else 0)

    diff = y - y_hat
    ups = X[diff != 0]
    diff = diff[diff != 0]

    x += f'{len(diff)}\t'

    w = w + ups.mul(diff, axis=0).sum()

x2 = ""
w = pd.Series([0,0,0])

eta = 1
for i in range(101):
    y_hat = (X*w).sum(axis=1).map(lambda x: 1 if x>0 else 0)

    diff = y - y_hat
    ups = X[diff != 0]
    diff = diff[diff != 0]

    x2 += f'{len(diff)}\t'

    w = w + eta/(i+1)*ups.mul(diff, axis=0).sum()

with open(output, 'w') as f:
    f.write(x)
    f.write('\n')
    f.write(x2)