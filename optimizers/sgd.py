from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np
import argparse


def sigmoid_activation(x):

    return 1.0 / (1 + np.exp(-x))


def predict(X, W):

    preds = sigmoid_activation(np.dot(X, W))

    preds[preds <= 0.5] = 0
    preds[preds > 0] = 1

    return preds


def next_batch(X, y, batch_size):

    for i in np.arange(0, X.shape[0], batch_size):
        yield (X[i:i + batch_size, :], y[i:i + batch_size, :])


parser = argparse.ArgumentParser()
parser.add_argument('-e', '--epochs', default=100, type=int,
                    help='# of epochs')
parser.add_argument('-a', '--alpha', default=0.01, type=float,
                    help='Learning rate')
parser.add_argument('-b', '--batch_size', default=32, type=int,
                    help='Batch size')
args = vars(parser.parse_args())

X, y = make_blobs(n_samples=1000, n_features=2, centers=2,
                  cluster_std=1.5, random_state=1)
y = y.reshape((y.shape[0], 1))
X = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5,
                                                    random_state=42)

W = np.random.randn(X.shape[1], 1)
losses = []

for epoch in np.arange(0, args['epochs']):

    epoch_loss = []

    for (batch_X, batch_y) in next_batch(X_train, y_train, args['batch_size']):

        preds = sigmoid_activation(np.dot(batch_X, W))
        error = batch_y - preds
        epoch_loss.append(np.sum(error ** 2))

        gradient = -np.dot(batch_X.T, error)
        W -= args['alpha'] * gradient

    loss = np.average(epoch_loss)
    losses.append(loss)

    if epoch == 0 or (epoch + 1) % 5 == 0:
        print(f"[INFO] Epoch {epoch + 1}, loss {loss}")

preds = predict(X_test, W)
print(classification_report(y_test, preds))

plt.style.use("ggplot")
plt.figure()
plt.title("Data")
plt.scatter(X_test[:, 0], X_test[:, 1], marker='o', c=y_test.ravel(), s=30)

plt.figure()
plt.plot(np.arange(0, args['epochs']), losses)
plt.title('Training Loss')
plt.xlabel('# of Epochs')
plt.ylabel('Loss')

plt.show()
