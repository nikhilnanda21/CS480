import numpy as np

n = 100
d = 50

X_dummy = np.random.randn(d, n)
X = X_dummy.T
Y = np.random.randn(n, 1)

lambdas = 10

def weight(X, Y, lambdas):
    w = np.zeros((X.shape[1], 1), dtype=np.float)
    xw = np.multiply(X, w.T)
    denom = np.sum(X**2, axis=0)

    y_hat   = np.sum(xw, axis = 1)
    y_res   = Y - np.reshape(y_hat, (-1, 1))

    change = 1

    while change>0.001 :
        w_dup = np.array(w)
        print(change)
        for j in range(w.shape[0]):

            x_j = np.reshape(X[:, j], (1, -1))
            prod = np.dot(x_j, (y_res + np.reshape(xw[:, j], (-1, 1))))
            #prod = np.dot(x_j, (Y - np.reshape(xw[:, j], (-1, 1))))

            den = denom[j]

            w[j] = (np.sign(prod))*(max(0, (prod-lambdas)/den))

            w_sub = w[j] - w_dup[j]
            xw_sub = np.multiply(X[:, j], w_sub)
            xw[:, j] += xw_sub
            y_res -= np.reshape(xw_sub, (-1, 1))

        change = np.max(abs(w_dup - w))

    print(w)
    return w

weight(X, Y, lambdas)
