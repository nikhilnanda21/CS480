import csv
import io
import urllib.request

url = "https://cs.uwaterloo.ca/~y328yu/mycourses/480/assignments/spambase_X.csv"
webpage = urllib.request.urlopen(url)
datareader = csv.reader(io.TextIOWrapper(webpage))

X = []

for row in datareader:
    #print(row)
    X.append(row)

#X = list(datareader)
#print(X[1])
#x1 = X[1]
#print(len(x1))

#print(len(X))
#print(len(X[1]))

url = "https://cs.uwaterloo.ca/~y328yu/mycourses/480/assignments/spambase_y.csv"
webpage = urllib.request.urlopen(url)
datareader = csv.reader(io.TextIOWrapper(webpage))

Y = list(datareader)
#print(Y)
#print(len(Y[1]))

#mistake = []
from itertools import repeat
mistake = list(repeat(0, 500))

from itertools import repeat
w = list(repeat(0, 57))

b = 0

import numpy as np

#print(len(Y))
#print((X[:,0]))
Y_1 = np.array(Y, dtype=float)
w_1 = np.array(w, dtype=float)
X_1 = np.array(X, dtype=float)
#print((X_1[:,0]))

for i in range(500):
    mistake[i] = 0
    for j in range(4601):
        if Y_1[j][0]*(np.dot(X_1[:,j], w) + b)<=0 :
        #if np.dot(X[:][j], w)<=0 :
            w += X_1[:,j]*Y_1[j]
            b += Y_1[j]
            mistake[i] += 1


import matplotlib.pyplot as plt
plt.plot(mistake)
plt.show()
