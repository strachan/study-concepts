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
