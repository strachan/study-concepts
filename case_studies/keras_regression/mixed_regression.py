from sklearn.model_selection import train_test_split
from keras.layers import Dense
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import concatenate
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

print('[INFO] loading house attributes...')
input_path = os.path.sep.join([args['dataset'], 'HousesInfo.txt'])
df = datasets.load_house_attributes(input_path)

# drop outliers before loading house images
df.drop(df[df['price'] > 3000000].index, inplace=True)

print('[INFO] loading house images...')
images = datasets.load_house_images(df, args['dataset'])
images = images / 255.0

print('[INFO] processing data...')
split = train_test_split(df, images, test_size=0.25, random_state=42)
(train_attr_x, test_attr_x, train_images_x, test_images_x) = split

max_price = train_attr_x['price'].max()
train_y = train_attr_x['price'] / max_price
test_y = test_attr_x['price'] / max_price

(train_attr_x, test_attr_x) = datasets.process_house_attributes(df,
                                                                train_attr_x,
                                                                test_attr_x)

mlp = models.create_mlp(train_attr_x.shape[1], regress=False)
cnn = models.create_cnn(64, 64, 3, regress=False)

combined_input = concatenate([mlp.output, cnn.output])

x = Dense(4, activation='relu')(combined_input)
x = Dense(1, activation='linear')(x)

model = Model(inputs=[mlp.input, cnn.input], outputs=x)

opt = Adam(lr=1e-3, decay=1e-3 / 200)
model.compile(loss='mean_absolute_percentage_error', optimizer=opt)

print('[INFO] training model...')
model.fit([train_attr_x, train_images_x], train_y,
          validation_data=([test_attr_x, test_images_x], test_y),
          epochs=200, batch_size=8)

print('[INFO] predicting house prices...')
preds = model.predict([test_attr_x, test_images_x])

diff = preds.flatten() - test_y
percent_diff = (diff / test_y) * 100
abs_percentage_diff = np.abs(percent_diff)

mean = np.mean(abs_percentage_diff)
std = np.std(abs_percentage_diff)

locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
print(f'avg house price: {locale.currency(df["price"].mean(), grouping=True)},'
      +
      f'std house price: {locale.currency(df["price"].std(), grouping=True)}')
print(f'mean: {mean}, std: {std}')
