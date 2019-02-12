from keras.preprocessing.image import img_to_array
from keras.models import load_model
import tensorflow as tf
import numpy as np
import argparse
import pickle
import cv2


parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", required=True,
                    help="path to trained model")
parser.add_argument("-l", "--categorybin", required=True,
                    help="path to output category label binarizer")
parser.add_argument("-c", "--colorbin", required=True,
                    help="path to output color label binarizer")
parser.add_argument("-i", "--image", required=True,
                    help="path to input image")
args = vars(parser.parse_args())

image = cv2.imread(args["image"])
output = image.copy()
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

image = cv2.resize(image, (96, 96))
image = image.astype("float") / 255.0
image = img_to_array(image)
# add dimension for the batch
image = np.expand_dims(image, axis=0)

print("[INFO] loading network...")
model = load_model(args["model"], custom_objects={"tf": tf})
category_lb = pickle.loads(open(args["categorybin"], "rb").read())
color_lb = pickle.loads(open(args["colorbin"], "rb").read())

print("[INFO] classifying image...")
(category_prob, color_prob) = model.predict(image)

category_idx = category_prob[0].argmax()
color_idx = color_prob[0].argmax()
category_label = category_lb.classes_[category_idx]
color_label = color_lb.classes_[color_idx]

category_text = f"category: {category_label}" + f"({category_prob[0][category_idx] * 100})"
color_text = f"color: {color_label} ({color_prob[0][color_idx] * 100})"
cv2.putText(output, category_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (0, 255, 0), 2)
cv2.putText(output, color_text, (20, 55), cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (0, 255, 0), 2)

print(f"[INFO] {category_text}")
print(f"[INFO] {color_text}")

cv2.imshow("Output", output)
cv2.waitKey(0)
