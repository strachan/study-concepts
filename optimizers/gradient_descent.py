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


parser = argparse.ArgumentParser()
parser.add_argument('-e', '--epochs', default=100, type=int,
                    help='# of epochs')
parser.add_argument('-a', '--alpha', default=0.01, type=float,
                    help='learning rate')
args = vars(parser.parse_args())

# create random dataset
(X, y) = make_blobs(n_samples=1000, n_features=2, centers=2,
                    cluster_std=1.5, random_state=1)
y = y.reshape((y.shape[0], 1))

# create a column of 1 to incorporate bias within the weight matrix
X = np.concatenate((X, np.ones([X.shape[0], 1])), axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5,
                                                    random_state=42)

W = np.random.randn(X.shape[1], 1)
losses = []

for epoch in np.arange(0, args['epochs']):

    preds = sigmoid_activation(np.dot(X_train, W))

    error = y_train - preds
    loss = np.sum(error ** 2)
    losses.append(loss)

    gradient = - np.dot(X_train.T, error)
    W -= args['alpha'] * gradient

    if epoch == 0 or (epoch + 1) % 5 == 0:
        print(f"[INFO] Epoch {epoch + 1}, Loss function = {loss}")

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
