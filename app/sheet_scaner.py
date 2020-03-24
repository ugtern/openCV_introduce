# import from local files
from cv_methods.preload import MainCVClass

# import from pypi packages
from imutils.perspective import four_point_transform
from imutils import contours

# import whole pypi packages
import numpy as np
import argparse
import imutils
import cv2


class SheetScaner:
    def __init__(self, resize):

        self.image = MainCVClass(resize).image
        self.gray = []
        self.current_contour_gray = []
        self.current_contour_color = []
        self.edged = []

        self.ANSWER_KEY = {
            0: 1,
            1: 4,
            2: 0,
            3: 3,
            4: 1
        }

    def edged_image(self):

        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(self.gray, (5, 5), 0)
        self.edged = cv2.Canny(blurred, 75, 200)

        cv2.imshow('edged', imutils.resize(self.edged, height=1000))
        cv2.waitKey(0)

    def find_contours(self):

        contours = cv2.findContours(self.edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)

        if len(contours) > 0:

            contours = sorted(contours, key=cv2.contourArea, reverse=True)

            for contour in contours:

                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

                if len(approx) == 4:

                    cv2.drawContours(self.image, [approx], -1, (255, 0, 170), 2)

        self.current_contour_gray = four_point_transform(self.gray, approx.reshape(4, 2))
        self.current_contour_color = four_point_transform(self.image, approx.reshape(4, 2))

        cv2.imshow('contours', self.image)
        cv2.imshow('current_contour', self.current_contour_color)
        cv2.waitKey(0)

    def find_answers(self):

        thresh = cv2.threshold(self.current_contour_gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

        cv2.imshow('thresh', thresh)
        cv2.waitKey(0)

        contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)

        for contour in contours:

            (x, y, w, h) = cv2.boundingRect(contour)
            ar = w / float(h)

            if w >= 20 and h >= 20 and 0.9 <= ar <= 1.1:

                cv2.drawContours(self.current_contour_color, [contour], -1, (255, 0, 0), 2)

        cv2.imshow('answers', self.current_contour_color)
        cv2.waitKey(0)


test = SheetScaner(False)
test.edged_image()
test.find_contours()
test.find_answers()
