from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import pickle
import cv2
import os


parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', required=True,
                    help='path to trained model')
parser.add_argument('-l', '--labelbin', required=True,
                    help='path to label binarizer')
parser.add_argument('-i', '--image', required=True,
                    help='path to input image')
args = vars(parser.parse_args())

image = cv2.imread(args['image'])
output = image.copy()

# image preprocessing
image = cv2.resize(image, (96, 96))
image = image.astype('float') / 255.0
image = img_to_array(image)
image = np.expand_dims(image, axis=0)

print('[INFO] loading network...')
model = load_model(args['model'])
mlb = pickle.loads(open(args['labelbin'], 'rb').read())

print('[INFO] classifying image...')
prob = model.predict(image)[0]
idxs = np.argsort(prob)[::-1][:2]

for (i, j) in enumerate(idxs):
    label = f'{mlb.classes_[j]}: {prob[j] * 100}%'
    cv2.putText(output, label, (10, (i * 30) + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

for (label, p) in zip(mlb.classes_, prob):
    print(f'{label}: {p * 100}%')

cv2.imshow('Output', output)
cv2.waitKey(0)
