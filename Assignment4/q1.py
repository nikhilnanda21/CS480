from keras.datasets import mnist
from sklearn.mixture import GaussianMixture as mix
import numpy as np

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape)
#print(x_train.type())

#x_train = np.flatten(x_train)
#x_train = x_train.flatten()

X = []

for i in range(60000):
    dummy = x_train[0].flatten()
    X.append(dummy)

X = np.array(X)

print(X.shape)

X_dummy = []

for i in range(10000):
    pappu = x_test[0].flatten()
    X_dummy.append(pappu)

X_dummy = np.array(X_dummy)

print(X_dummy.shape)

model = mix(n_components = 5)
#model.fit(X, y_train)

logprob = model.fit(X).score_samples(X)
responsibilities = model.predict_proba(X)

# preds = model.predict(X_dummy)
# preds_dummy = model.predict_proba(X_dummy)

#print(preds)
#print(y_test)
#print(preds-y_test)
#print(preds_dummy)

print(logprob)
print(responsibilities)
