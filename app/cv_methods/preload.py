import argparse
import cv2
from .index import CvTests


class MainCVClass():
    def __init__(self, resize):

        ap = argparse.ArgumentParser()
        ap.add_argument("-i", "--image", required=True, help="path to input image")
        args = vars(ap.parse_args())

        if resize:
            image = CvTests(args["image"])
            self.image = image.resize_with_imutils(400.0)
        else:
            self.image = cv2.imread(args["image"])
