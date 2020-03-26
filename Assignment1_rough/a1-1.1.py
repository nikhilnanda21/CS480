import numpy as np
import matplotlib.pyplot as plt

filename1 = "spambase_X.csv"
filename2 = "spambase_y.csv"

X = np.loadtxt(filename1, delimiter=",")
y = np.loadtxt(filename2, delimiter=",")

#print(len(X))
#print(len(X[0]))

X = np.transpose(X)

#print(y)


b = 0
w = np.zeros(57)
max_pass = 500
n = 4601
mistake = np.zeros(max_pass)

def Perceptron(max_pass, b, w):

    for i in range(max_pass):
        mistake[i] = 0
        for j in range(n):
            if (y[j]*((np.dot(X[j], w))+b))<=0 :
                w += y[j]*X[j]
                b += y[j]
                mistake[i] += 1


    plt.plot(mistake)
    plt.xlabel('number of passes')
    plt.ylabel('number of mistakes')
    plt.show()

Perceptron(max_pass, b, w)
