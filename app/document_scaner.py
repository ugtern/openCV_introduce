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
        self.ratio = self.image.shape[0] / 500
        self.orig = self.image.copy()

    def get_edget(self):

        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        th3 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

        # th3 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        edged = cv2.Canny(th3, 80, 200)

        edged_for_watching = imutils.resize(edged.copy(), height=500)

        # cv2.imshow('gray', gray)
        cv2.imshow('th3', imutils.resize(th3, height=1000))
        cv2.imshow('edged', imutils.resize(edged, height=1000))
        cv2.waitKey(0)

        return th3

    def get_contours(self):

        contours = cv2.findContours(self.get_edget().copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

        screen_contour = np.array([])

        for contour in contours:

            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            
            print(len(approx))

            if len(approx) == 4:
                screen_contour = approx
                cv2.drawContours(self.image, [screen_contour], -1, (0, 255, 0), 2)
                # break

        # current_contour = cv2.boundingRect(screen_contour)
        # print(current_contour)
        # current_contour = self.image[current_contour[0]:current_contour[1], current_contour[2]:current_contour[3]]
        # current_contour = imutils.resize(current_contour, height=1000)
        
        # print(self.image.shape)
        # M = cv2.getPerspectiveTransform(screen_contour, np.float32([0, 0], []))
        # current_contour = (self.image, M)
        
        current_contour = imutils.resize(four_point_transform(self.image, screen_contour.reshape(4, 2)), height=1000)

        gray_current_contour = cv2.cvtColor(current_contour, cv2.COLOR_BGR2GRAY)
        gray_current_contour = cv2.GaussianBlur(gray_current_contour, (5, 5), 0)
        gray_current_contour = cv2.Canny(gray_current_contour, 70, 200)
        
        cv2.imshow('outline', imutils.resize(self.image, height=1000))
        cv2.imshow('current_contour', current_contour)
        cv2.imshow('gray_current_contour', gray_current_contour)
        cv2.waitKey(0)

        return screen_contour

    def rotate_image(self):

        warped = four_point_transform(self.orig, self.get_contours().reshape(4, 2))

        warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        T = threshold_local(warped, 11, offset=10, method='gaussian')
        warped = (warped > T).astype('uint8') * 255

        cv2.imshow('warped', imutils.resize(warped, height=1000))
        cv2.waitKey(0)


test = DocScan(False)
# test.get_edget()
# test.get_contours()
test.rotate_image()
