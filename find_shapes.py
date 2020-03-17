import numpy as np
import argparse
import imutils
import cv2
import index


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

        cv2.imshow('image', self.image)
        cv2.waitKey(0)

    def finding_shapes(self):

        # lower = np.array([60, 250, 90])
        # upper = np.array([70, 255, 100])

        lower = np.array([100, 110, 140])
        upper = np.array([145, 145, 180])

        shapeMask = cv2.inRange(self.image, lower, upper)

        cv2.imshow('mask', shapeMask)

        return shapeMask

    def get_contours(self):

        contours = cv2.findContours(self.finding_shapes().copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)

        for contour in contours:
            cv2.drawContours(self.image, [contour], -1, (0, 255, 0), 2)
        cv2.imshow('contours', self.image)
        cv2.waitKey(0)


test = OpenCVTest(True)
test.get_contours()
