import numpy as np
import argparse
import imutils
import cv2


class OpenCVTest:

    def __init__(self, resize):

        ap = argparse.ArgumentParser()
        ap.add_argument("-i", "--image", required=True, help="path to input image")
        args = vars(ap.parse_args())

        if resize:
            image = index.CvTests(args["image"])
            self.image = image.resize_with_imutils(400.0)
        else:
            self.image = cv2.imread(args["image"])

    def sort_contours(self, contours, method="left-to-right"):
        reverse = False
        i = 0

        if method == "right-to-left" or method == "bottom-to-top":
            reverse = True

        boundingBoxes = [cv2.boundingRect(contour) for contour in contours]
        (contours, boundingBoxes) = zip(*sorted(zip(contours, boundingBoxes), key=lambda b: b[1][i], reverse=reverse))

        return (contours, boundingBoxes)

