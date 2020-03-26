import numpy as np

def weight(X, Y, lambdas):
    w = np.zeros((X.shape[1], 1), dtype=np.float)
    xw = np.multiply(X, w.T)
    denom = np.sum(X**2, axis=0)

    y_hat   = np.sum(xw, axis = 1)
    y_res   = Y - np.reshape(y_hat, (-1, 1))

    change = 1

    while change>0.001 :
        w_dup = np.array(w)
        for j in range(w.shape[0]):

            x_j = np.reshape(X[:, j], (1, -1))
            #prod = np.dot(x_j, (y_res + np.reshape(xw[:, j], (-1, 1))))
            prod = np.dot(x_j, (Y - np.reshape(xw[:, j], (-1, 1))))

            den = denom[j]

            w[j] = (np.sign(prod))*(max(0, (prod-lambdas)/den))

            w_sub = w[j] - w_dup[j]
            xw_sub = np.multiply(X[:, j], w_sub)
            xw[:, j] += xw_sub
            y_res -= np.reshape(xw_sub, (-1, 1))

        change = np.max(abs(w_dup - w))

    #print(w)
    return w

def errors(X, Y, w):
    error = Y - np.dot(X, w)
    error_sq = error**2
    return (np.sum(error_sq)/len(error_sq))

# n = 10
# d = 5
#
# X = np.random.randn(d, n)
# y = np.random.randn(n, 1)
# w = np.zeros([d, 1])
# z = 0
# lambda = 10
# change = 1
#
# while change > 0.001:
#     for j in range(d):
#         z = np.sign(w[j])*max(0, (abs(w[j])-lambda))
#         w[j] = z
#         #wrong

n = 100
d = 50

X_dummy = np.random.randn(d, n)
X = X_dummy.T
Y = np.random.randn(n, 1)
#Padding 1

lamb = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

perf = np.zeros(0)

for lambdas in lamb:
    k = 10
    perf_val = 0
    perf_train = 0

    for i in range(k):
        left_k = int(np.round((i)*X.shape[0]/k))
        right_k = int(np.round((i+1)*X.shape[0]/k))

        X_train = np.concatenate((X[0:left_k, :], X[right_k:X.shape[0], :]))
        Y_train = np.concatenate((Y[0:left_k, :], Y[right_k:Y.shape[0], :]))

        X_val = X[left_k:right_k, :]
        Y_val = Y[left_k:right_k, :]

        w = weight(X_train, Y_train, lambdas)
        print(w[0])

        perf_train += errors(X_train, Y_train, w)

        perf_val += errors(X_val, Y_val, w)

    perf_train = perf_train/k
    perf_val = perf_val/k

    #print(perf_val)

    perf = np.append(perf, perf_val)

#print("Perf:", perf)
#print("argmin", perf.argmin())
#print("Perf shape", perf.shape[0])
best_lambda = lamb[perf.argmin()]
print("Best Lambda:", best_lambda)

#lambdas = 10

#weight(X, Y, lambdas)



#weight(X, Y, lambdas)
