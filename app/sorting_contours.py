import numpy as np
import argparse
import imutils
import cv2
from cv_methods import main


class OpenCVTest:

    def __init__(self, resize):

        self.image = main.MainCVClass(resize).image
        self.accum_edged = np.zeros(self.image.shape[:2], dtype="uint8")

        for chan in cv2.split(self.image):
            chan = cv2.medianBlur(chan, 11)
            edged = cv2.Canny(chan, 50, 200)
            self.accum_edged = cv2.bitwise_or(self.accum_edged, edged)

        cv2.imshow('edged map', self.accum_edged)
        cv2.waitKey(0)

    def sort_contours(self, contours, method="left-to-right"):
        reverse = False
        i = 0

        if method == "right-to-left" or method == "bottom-to-top":
            reverse = True

        if method == "top-to-bottom" or method == "left-to-right":
            i = 1

        bounding_boxes = [cv2.boundingRect(contour) for contour in contours]
        (contours, bounding_boxes) = zip(*sorted(zip(contours, bounding_boxes), key=lambda b: b[1][i], reverse=reverse))

        return (contours, bounding_boxes)

    def draw_contour(self, image, i, c):

        M = cv2.moments(c)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        cv2.putText(image, "#{}".format(i + 1), (cX - 20, cY), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        return image

    def sorting_contours(self):

        contours = cv2.findContours(self.accum_edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
        orig = self.image.copy()

        for (i, c) in enumerate(contours):
            orig = self.draw_contour(orig, i, c)

        cv2.imshow("unsorted", orig)
        cv2.waitKey(0)

        (contours, bounding_boxes) = self.sort_contours(contours, method=self.args["method"])

        for (i, c) in enumerate(contours):
            self.draw_contour(self.image, i, c)

        cv2.imshow("sorted", self.image)
        cv2.waitKey(0)


test = OpenCVTest(True)
test.sorting_contours()
