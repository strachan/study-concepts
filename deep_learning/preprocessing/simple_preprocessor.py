import cv2


class SimplePrepocessor:

    def __init__(self, width, height, inter=cv2.INTER_AREA):
        """
        Image preprocessor that resizes the image, ignoring the aspect ratio

        Store the target image width, height, and interpolation method
        used when resizing

        Parameters:
            width (int): desired image width
            height (int): desired image height
            inter (obj): OpenCV interpolation method. Default: cv2.INTER_AREA

        Attributes:
            width (int): desired image width
            height (int): desired image height
            inter (obj): OpenCV interpolation method. Default: cv2.INTER_AREA
        """
        self.width = width
        self.height = height
        self.inter = inter

    def preprocess(self, image):
        """
        Resize the image to a fized size, ignoring the aspect ratio

        Parameters:
            image (numpy.array): image to be resized

        Returns:
            The image resized
        """
        return cv2.resize(image, (self.width, self.height),
                          interpolation=self.inter)
