from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import datasets
import models
import numpy as np
import argparse
import locale
import os


parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset', type=str, required=True,
                    help='path to input dataset of house images')
args = vars(parser.parse_args())

path = os.path.sep.join([args['dataset'], 'HousesInfo.txt'])
df = datasets.load_house_attributes(path)

df.drop(df[df['price'] > 3000000].index, inplace=True)

images = datasets.load_house_images(df, args['dataset'])
images = images / 255.0

split = train_test_split(df, images, test_size=0.25, random_state=42)
(train_attr_x, test_attr_x, train_images_x, test_images_x) = split

max_price = train_attr_x['price'].max()
train_y = train_attr_x['price'] / max_price
test_y = test_attr_x['price'] / max_price

model = models.create_cnn(64, 64, 3, regress=True)
opt = Adam(lr=1e-3, decay=1e-3 / 200)
model.compile(loss='mean_absolute_percentage_error', optimizer=opt)

model.fit(train_images_x, train_y, validation_data=(test_images_x, test_y),
          epochs=200, batch_size=8)

preds = model.predict(test_images_x)

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
