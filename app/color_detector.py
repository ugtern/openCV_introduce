import cv2
import argparse
import numpy as np
from cv_methods import main


class OpenCVTest:

    def __init__(self, resize):

        self.image = main.MainCVClass(resize).image

        self.boundaries = [
            ([5, 5, 100], [80, 80, 200]),
            ([86, 31, 4], [220, 88, 50]),
            ([25, 146, 190], [62, 174, 250]),
            ([103, 86, 65], [145, 133, 128]),
            ([10, 30, 60], [140, 180, 250])
        ]

    def finding(self):
        # loop over the boundaries
        for (lower, upper) in self.boundaries:
            # create NumPy arrays from the boundaries
            lower = np.array(lower, dtype="uint8")
            upper = np.array(upper, dtype="uint8")
            # find the colors within the specified boundaries and apply
            # the mask
            mask = cv2.inRange(self.image, lower, upper)
            output = cv2.bitwise_and(self.image, self.image, mask=mask)
            # show the images
            cv2.imshow("images", np.hstack([self.image, output]))
            cv2.waitKey(0)


test = OpenCVTest(True)
test.finding()
