import numpy as np
import timeit
import time
import matplotlib.pyplot as plt

n = [100, 1000, 5000]
d = 100

time_loop = np.zeros([len(n), 1])
time_no_loop = np.zeros([len(n),1])

for i in range(len(n)):

    X = np.random.randn(d, n[i])

    D = np.zeros([n[i], n[i]], dtype=float)
    D_fast = np.zeros([n[i], n[i]], dtype=float)

    #start = timeit.default_timer()
    start = time.time()

    for a in range(n[i]):
        for j in range(n[i]):
            D[a][j] = np.sqrt(np.sum(np.power((X[:, a] - X[:, j]), 2)))

    time_loop[i] = time.time() - start

    start = time.time()

    D_fast = np.sqrt(np.sum(X.T**2, axis=1)[:, np.newaxis] + np.sum(X.T**2, axis=1) - 2*(np.dot(X.T, X)))

    time_no_loop[i] = time.time() - start



plt.plot(n, time_loop, marker = 'o')


plt.plot(n, time_no_loop, marker = 'o')
plt.legend(['2 Loops', 'No Loops'], loc='upper left')
plt.xlabel('n')
plt.ylabel('Time (in seconds)')
plt.show()

print(time_loop)
print(time_no_loop)
