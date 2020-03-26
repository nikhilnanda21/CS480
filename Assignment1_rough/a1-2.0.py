import numpy as np

n = 50
d = 30

w = np.zeros([d, 1])
X = np.random.randn(d, n)
Y = np.random.randn(n)

lam = 1

diff = 1

count = 0

while diff > 0.001 :
    w_old = np.array(w)
    count += 1
    for j in range(d):
        x_j = np.array(X[j])

        den = np.dot(x_j, x_j.T)

        x_no_j = np.delete(X, j, 0)

        w_no_j = np.delete(w, j)

        prod = np.dot(x_no_j.T, w_no_j)

        num = np.dot(x_j, (Y - prod))


        w[j] = np.sign(num/den)*max(0, (abs(num/den))-(lam/den))

    w_diff = w - w_old

    diff = np.sqrt(np.sum(w_diff**2))

print("w", w)
