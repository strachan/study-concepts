import matplotlib
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from smallervggnet import SmallerVGGNet
import matplotlib.pyplot as plt
from imutils import paths
import numpy as np
import argparse
import random
import pickle
import cv2
import os

matplotlib.use('Agg')

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset', required=True,
                    help='path to input dataset')
parser.add_argument('-m', '--model', required=True,
                    help='path to output model')
parser.add_argument('-l', '--labelbin', required=True,
                    help='path to output label binarizer')
parser.add_argument('-p', '--plot', type=str, default='plot.png',
                    help='path to output accuracy/loss plot')
args = vars(parser.parse_args())

# initialize model meta parameters
EPOCHS = 75  # number of epochs to train
INIT_LR = 1e-3  # initial learning rate
BS = 32  # batch size
IMAGE_DIMS = (96, 96, 3)  # image dimensions

# grab the image paths and suffle them
print('[INFO] loading images...')
image_paths = sorted(list(paths.list_images(args['dataset'])))
random.seed(42)
random.shuffle(image_paths)

data = []
labels = []

for image_path in image_paths:
    image = cv2.imread(image_path)
    image = cv2.resize(image, (IMAGE_DIMS[1], IMAGE_DIMS[0]))
    image = img_to_array(image)
    data.append(image)

    label = image_path.split(os.path.sep)[-2].split('_')
    labels.append(label)

# scale the pixels intensities
data = np.array(data, dtype='float') / 255.0
labels = np.array(labels)
print(f'[INFO] data matrix: {len(image_paths)} images ' +
      f'({data.nbytes / (1024 * 1000.0)} MB)')

print('[INFO] class labels:')
mlb = MultiLabelBinarizer()
labels = mlb.fit_transform(labels)

for (i, label) in enumerate(mlb.classes_):
    print(f'{i + 1}. {label}')

(train_x, test_x, train_y, test_y) = train_test_split(data, labels,
                                                      test_size=0.2,
                                                      random_state=42)

aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
                         height_shift_range=0.1, shear_range=0.2,
                         zoom_range=0.2, horizontal_flip=True,
                         fill_mode='nearest')

print('[INFO] compiling model...')
model = SmallerVGGNet.build(width=IMAGE_DIMS[1], height=IMAGE_DIMS[0],
                            depth=IMAGE_DIMS[2], classes=len(mlb.classes_),
                            final_act='sigmoid')
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

print('[INFO] training network...')
H = model.fit_generator(aug.flow(train_x, train_y, batch_size=BS),
                        validation_data=(test_x, test_y),
                        steps_per_epoch=len(train_x) // BS,
                        epochs=EPOCHS, verbose=1)

print('[INFO] serializing network...')
model.save(args['model'])

print('[INFO] serializing label binarizer')
with open(args['labelbin'], 'wb') as f:
    f.write(pickle.dumps(mlb))

plt.style.use('ggplot')
plt.figure()
N = EPOCHS
plt.plot(np.arange(0, N), H.history['loss'], label='train_loss')
plt.plot(np.arange(0, N), H.history['val_loss'], label='val_loss')
plt.plot(np.arange(0, N), H.history['acc'], label='train_acc')
plt.plot(np.arange(0, N), H.history['val_acc'], label='val_acc')
plt.title('Training Loss and Accuracy')
plt.xlabel('Epoch #')
plt.ylabel('Loss/Accuracy')
plt.legend(loc='upper left')
plt.savefig(args['plot'])
