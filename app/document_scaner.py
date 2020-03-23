from imutils.perspective import four_point_transform
from skimage.filters import threshold_local
from cv_methods.preload import MainCVClass

import numpy as np
import argparse
import cv2
import imutils


class DocScan:
    def __init__(self, resize):
        self.image = MainCVClass(resize).image

    def get_edget(self):

        ratio = self.image.shape[0] / 500
        orig = self.image.copy()

        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(gray, 75, 200)

        # edged = imutils.resize(edged, height=500)

        cv2.imshow('edged', edged)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


test = DocScan(False)
test.get_edget()
