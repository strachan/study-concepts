from sklearn.model_selection import train_test_split
from sklearn.datasets import make_circles
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization
from keras.optimizers import SGD
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-b', '--batch_normalization', action='store_true')
args = vars(parser.parse_args())

X, y = make_circles(n_samples=1000, noise=0.1, random_state=1)
y = y.reshape((y.shape[0], 1))

for i in range(2):
    samples_ix = np.where(y == i)
    plt.scatter(X[samples_ix, 0], X[samples_ix, 1], label=str(i))
plt.legend()
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5,
                                                    random_state=42)

model = Sequential()
model.add(Dense(50, input_dim=X_train.shape[1], activation='relu',
                kernel_initializer='he_uniform'))
if args['batch_normalization']:
    model.add(BatchNormalization())
model.add(Dense(1, activation='sigmoid'))
opt = SGD(lr=0.01, momentum=0.9)
model.compile(loss='binary_crossentropy', optimizer=opt,
              metrics=['accuracy'])

history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                    epochs=100, verbose=1)

_, train_acc = model.evaluate(X_train, y_train, verbose=0)
_, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f'Train: {train_acc}, Test: {test_acc}')

plt.plot(history.history['acc'], label='train')
plt.plot(history.history['val_acc'], label='test')
plt.legend()
plt.show()
