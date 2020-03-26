import matplotlib.pyplot as plt
import numpy as np

lambdas = 1

w = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
z = np.zeros(11)

for i in range(len(w)):
    z[i] = np.sign(w[i])*max(0, abs(w[i])-lambdas)

plt.plot(w, z)
plt.xlabel('w')
plt.ylabel('z - soft thresholding operator')
plt.show()
