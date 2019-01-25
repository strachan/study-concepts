from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import datasets
import models
import numpy as np
import argparse
import locale
import os


ap = argparse.ArgumentParser()
ap.add_argument('-d', '--dataset', type=str, required=True,
                help='path to input dataset of house images')
args = vars(ap.parse_args())

# Loading dataset
path = os.path.sep.join([args['dataset'], 'HousesInfo.txt'])
df = datasets.load_house_attributes(path)

# Splitting in train and test
(train, test) = train_test_split(df, test_size=0.25, random_state=42)

# Check outliers
df.drop(df[df['price'] > 3000000].index, inplace=True)
# plt.hist(df['price'])
# plt.show()

# plt.figure()
# plt.scatter(df['area'], df['price'])
# plt.show()

# Scale output between [0, 1]
max_price = train['price'].max()
train_y = train['price'] / max_price
test_y = test['price'] / max_price

# Processing data
(train_x, test_x) = datasets.process_house_attributes(df, train, test)

# Create model
model = models.create_mlp(train_x.shape[1], regress=True)
opt = Adam(lr=0.001, decay=0.001 / 200)
model.compile(loss='mean_absolute_percentage_error', optimizer=opt)

# Train model
model.fit(train_x, train_y, validation_data=(test_x, test_y),
          epochs=200, batch_size=8)

# Predict house prices
preds = model.predict(test_x)

diff = preds.flatten() - test_y
percent_diff = (diff / test_y) * 100
abs_percent_diff = np.abs(percent_diff)

mean = np.mean(abs_percent_diff)
std = np.std(abs_percent_diff)

locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
print(f'avg house price: {locale.currency(df["price"].mean(), grouping=True)},'
      +
      f'std house price: {locale.currency(df["price"].std(), grouping=True)}')
print(f'mean: {mean}, std: {std}')
