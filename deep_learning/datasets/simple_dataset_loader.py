import numpy as np
import cv2
import os


class SimpleDatasetLoader:

    def __init__(self, preprocessors=None):
        """
        Dataset loader

        Parameters:
            preprocessors (list): list of preprocessors

        Attributes:
            preprocessors (list): list of preprocessors to be applied
                                  to each image
        """

        self.preprocessors = preprocessors

        if self.preprocessors is None:
            self.preprocessors = []

    def load(self, image_paths, verbose=-1):
        """
        Load the image and extract the class label assuming that our path
        had the following format: /path/to/dataset/{class}/{image}.jpg

        Parameters:
            preprocessors (list): list of preprocessors
            verbose (int): verbosity level. Default: -1

        Returns:
            Tuple with array of data and array of labels
        """
        data = []
        labels = []

        for (i, image_path) in enumerate(image_paths):
            image = cv2.imread(image_path)
            label = image_path.split(os.path.sep)[-2]

            if self.preprocessors is not None:
                for processor in self.preprocessors:
                    image = processor.preprocess(image)

            data.append(image)
            labels.append(label)

            if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
                print(f"[INFO] processed {i + 1}/{len(image_paths)}")

            return (np.array(data), np.array(labels))
