import matplotlib
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from fashionnet import FashionNet
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import pickle
import cv2
import os

matplotlib.use("Agg")

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", required=True,
                    help="path to input dataset")
parser.add_argument("-m", "--model", required=True,
                    help="path to output model")
parser.add_argument("-l", "--categorybin", required=True,
                    help="path to output category label binarizer")
parser.add_argument("-c", "--colorbin", required=True,
                    help="path to output color label binarizer")
parser.add_argument("-p", "--plot", type=str, default="output",
                    help="base filename for generated plots")
args = vars(parser.parse_args())

# constants
EPOCHS = 50
INIT_LR = 1e-3
BS = 32
IMAGE_DIMS = (96, 96, 3)

print("[INFO] loading images...")
image_paths = sorted(list(paths.list_images(args["dataset"])))
random.seed(42)
random.shuffle(image_paths)

data = []
category_labels = []
color_labels = []

for image_path in image_paths:
    image = cv2.imread(image_path)
    image = cv2.resize(image, (IMAGE_DIMS[1], IMAGE_DIMS[0]))
    # we convert because the model need a RGB image to transform to grayscale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = img_to_array(image)
    data.append(image)

    (color, cat) = image_path.split(os.path.sep)[-2].split("_")
    category_labels.append(cat)
    color_labels.append(color)

data = np.array(data, dtype="float") / 255.0
print(f"[INFO] data matrix: {len(image_paths)} " +
      f"({data.nbytes / (1024 * 1000)}MB)")

category_labels = np.array(category_labels)
color_labels = np.array(color_labels)

print("[INFO] binarizing labels...")
category_lb = LabelBinarizer()
color_lb = LabelBinarizer()
category_labels = category_lb.fit_transform(category_labels)
color_labels = color_lb.fit_transform(color_labels)

split = train_test_split(data, category_labels, color_labels, test_size=0.2,
                         random_state=42)
(train_x, test_x, train_category_y, test_category_y,
    train_color_y, test_color_y) = split

model = FashionNet.build(IMAGE_DIMS[1], IMAGE_DIMS[0],
                         num_categories=len(category_lb.classes_),
                         num_colors=len(color_lb.classes_),
                         final_act="softmax")

losses = {
    "category_output": "categorical_crossentropy",
    "color_output": "categorical_crossentropy"
}
loss_weights = {
    "category_output": 1.0,
    "color_output": 1.0
}

print("[INFO] compiling model...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(optimizer=opt, loss=losses, loss_weights=loss_weights,
              metrics=["accuracy"])

H = model.fit(train_x, {"category_output": train_category_y,
                        "color_output": train_color_y},
              validation_data=(test_x, {"category_output": test_category_y,
                                        "color_output": test_color_y}),
              epochs=EPOCHS,
              verbose=1)

print("[INFO] serializing network...")
model.save(args["model"])

print("[INFO] serializing category label binarizer...")
with open(args["categorybin"], "wb") as f:
    f.write(pickle.dumps(category_lb))

print("[INFO] serializing color label binarizer...")
with open(args["colorbin"], "wb") as f:
    f.write(pickle.dumps(color_lb))

loss_names = ["loss", "category_output_loss", "color_output_loss"]
plt.style.use("ggplot")
(fig, ax) = plt.subplots(3, 1, figsize=(13, 13))

for (i, l) in enumerate(loss_names):
    title = f"Loss for {l}" if l != "loss" else "Total Loss"
    ax[i].set_title(title)
    ax[i].set_xlabel("Epoch #")
    ax[i].set_ylabel("Loss")
    ax[i].plot(np.arange(0, EPOCHS), H.history[l], label=l)
    ax[i].plot(np.arange(0, EPOCHS), H.history["val_" + l], label="val_" + l)
    ax[i].legend()

plt.tight_layout()
plt.savefig(f'{args["plot"]}_losses.png')
plt.close()

accuracy_names = ["category_output_acc", "color_output_acc"]
plt.style.use("ggplot")
(fig, ax) = plt.subplots(2, 1, figsize=(8, 8))

for (i, l) in enumerate(accuracy_names):
    ax[i].set_title(f"Accuracy for {l}")
    ax[i].set_xlabel("Epoch #")
    ax[i].set_ylabel("Accuracy")
    ax[i].plot(np.arange(0, EPOCHS), H.history[l], label=l)
    ax[i].plot(np.arange(0, EPOCHS), H.history["val_" + l], label="val_" + l)
    ax[i].legend()

plt.tight_layout()
plt.savefig(f'{args["plot"]}_accs.png')
plt.close()
