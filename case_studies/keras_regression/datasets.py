from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import glob
import cv2
import os


def load_house_attributes(path):
    cols = ['bedrooms', 'bathrooms', 'area', 'zipcode', 'price']
    df = pd.read_csv(path, sep=' ', header=None, names=cols)

    zipcodes = df['zipcode'].value_counts().keys().tolist()
    counts = df['zipcode'].value_counts().tolist()

    # drop houses with less than 25 houses per zipcode
    for (zipcode, count) in zip(zipcodes, counts):
        if count < 25:
            idxs = df[df['zipcode'] == zipcode].index
            df.drop(idxs, inplace=True)

    return df


def process_house_attributes(df, train, test):
    continuous = ['bedrooms', 'bathrooms', 'area']

    cs = MinMaxScaler()
    train_continuous = cs.fit_transform(train[continuous])
    test_continuous = cs.transform(test[continuous])

    zip_binarizer = LabelBinarizer().fit(df['zipcode'])
    train_categorical = zip_binarizer.transform(train['zipcode'])
    test_categorical = zip_binarizer.transform(test['zipcode'])

    train_x = np.hstack([train_continuous, train_categorical])
    test_x = np.hstack([test_continuous, test_categorical])

    return (train_x, test_x)


def load_house_images(df, path):

    images = []

    for i in df.index.values:
        base_path = os.path.sep.join([path, f'{i + 1}_*'])
        house_paths = sorted(list(glob.glob(base_path)))

        input_images = []
        output_image = np.zeros((64, 64, 3), dtype='uint8')

        for house_path in house_paths:
            image = cv2.imread(house_path)
            image = cv2.resize(image, (32, 32))
            input_images.append(image)

        output_image[0:32, 0:32] = input_images[0]
        output_image[0:32, 32:64] = input_images[1]
        output_image[32:64, 32:64] = input_images[2]
        output_image[32:64, 0:32] = input_images[3]

        images.append(output_image)

    return np.array(images)
