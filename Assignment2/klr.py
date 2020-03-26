import numpy as np
import matplotlib.pyplot as plt

def sigmoid(a):
    #z = np.dot(X, weight)
    return 1 / (1 + np.exp((-1)*a))

filename1 = "train_X_dog_cat.csv"
filename2 = "train_y_dog_cat.csv"

X = np.loadtxt(filename1, delimiter=",")
y = np.loadtxt(filename2, delimiter=",")

#alpha = np.zeros([1953, 1])
alpha = np.zeros(3072)
g = np.zeros([3072, 1])

# print(X.shape)
# print(X.shape)
# print(y.shape)
# print(y)

K = np.zeros([3072, 3072])
K_1 = np.zeros([3072, 3072])
n = 0.001
flag = 0
lam = 5

#Linear Kernel
# for i in range(3072):
#     for j in range(3072):
#         K_1[i][j] = np.dot(X[:, i].T, X[:, j])

inp = input("Enter 1-Linear, 2-Poly, 3-Gaussian")

for t in range(1, 2001):
    g = np.zeros(3072)
    flag = 0
    #print("t=", t)
    for i in range(1953):
        #p = 1/(1+np.exp(-1*np.dot(alpha.T, K[:, i])))
        #g += (p - ((y[i]+1)/2) + (2*lam*alpha[i]))*(K[:, i])
        #print(g.shape)
        ex = (np.exp((-1)*(y[i])*(np.dot(alpha.T, K[:, i]))))
        #print(float((y[i]*ex)/(1+ex)))
        #print(K[i, :])
        g += ((float((y[i]*ex)/(1+ex))) * (K[i, :]))
        #g += K[i, :]

        if np.abs(np.mean(g)) < 1e-6:
            #print("Tolerance reached at iteration", i)
            #return
            flag = 1
            break

    if flag==1:
        break
    #dummy = np.array(n*g)
    #print("Dummy: ", dummy.shape)
    alpha -= n*g

#print(K[0][0])
filename1 = "test_X_dog_cat.csv"
filename2 = "test_y_dog_cat.csv"

X_test = np.loadtxt(filename1, delimiter=",")
y_test = np.loadtxt(filename2, delimiter=",")

K_test = np.zeros([X_test.shape[0], X_test.shape[0]])

# for i in range(X_test.shape[0]):
#     for j in range(X_test.shape[0]):
#         K_test[i][j] = np.dot(X_test[i, :].T, X_test[j, :])

s = np.dot(K, alpha)
h = sigmoid(s)

correct_c = 0
correct_d = 0

# Compare prediction results
for i in range(y_test.shape[0]):
  if h[i] < 0.5 and y_test[i] == -1: correct_c += 1
  if h[i] >= 0.5 and y_test[i] == 1: correct_d += 1

# Print results
acc = (correct_c + correct_d) / y_test.shape[0]
print("Test accuracy % = ", acc*100)
