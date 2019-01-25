from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from pyimagesearch.preprocessing import simple_preprocessor
from pyimagesearch.datasets import simple_dataset_loader
from imutils import paths
import argparse


ap = argparse.ArgumentParser()
ap.add_argument('-d', '--dataset', required=True,
                help='path to input dataset')
ap.add_argument('-k', '--neighbors', type=int, default=3,
                help='# of nearest neighbors for classification')
ap.add_argument('-j', '--jobs', type=int, default=-1,
                help='# of jobs for k-NN distance' +
                     '(-1 uses all available cores)')
args = vars(ap.parse_args())

print('[INFO] loading images...')
image_paths = list(paths.list_images(args['dataset']))

# initialize the image preprocessor, load the dataset from disk,
# and reshape the data matrix for the classification model
sp = simple_preprocessor(32, 32)
sdl = simple_dataset_loader(preprocessors=[sp])
(data, labels) = sdl.load(image_paths, verbose=500)
data = data.reshape((data.shape[0], 3072))

# show memory consumption of the images
print(f'[INFO] features matrix: {data.nbytes / (1024 * 1000.0)}MB')

# encode the label as integers
le = LabelEncoder()
labels = le.fit_transform(labels)

# partition the data into training and testing
(train_x, test_x, train_y, test_y) = train_test_split(data, labels,
                                                      test_size=0.25,
                                                      random_state=42)

# train and evaluate a k-NN classifier
print('[INFO] evaluating k-NN classifier...')
model = KNeighborsClassifier(n_neighbors=args['neighbors'],
                             n_jobs=args['jobs'])
model.fit(train_x, train_y)
print(classification_report(test_y, model.predict(test_y),
                            target_names=le.classes_))
