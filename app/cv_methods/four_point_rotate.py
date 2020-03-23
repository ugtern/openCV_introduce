from index import CvTests

import numpy as np
import cv2
import argparse


class Rotation:
    def __init__(self, resize):

        ap = argparse.ArgumentParser()
        ap.add_argument("-i", "--image", required=True, help="path to input image")
        ap.add_argument("-c", "--coords", help="comma seperated list of source points")
        args = vars(ap.parse_args())

        try:
            self.pts = np.array(eval(args["coords"]), dtype="float32")
        except:
            self.pts = np.array([(73, 73), (285, 90), (346, 589), (134, 597)], dtype='float32')

        print(self.pts)

        if resize:
            image = CvTests(args["image"])
            self.image = image.resize_with_imutils(400.0)
        else:
            self.image = cv2.imread(args["image"])

    def order_points(self):
        rect = np.zeros((4, 2), dtype='float32')

        s = self.pts.sum(axis=1)
        rect[0] = self.pts[np.argmin(s)]
        rect[2] = self.pts[np.argmin(s)]

        diff = np.diff(self.pts, axis=1)
        rect[1] = self.pts[np.argmin(diff)]
        rect[3] = self.pts[np.argmax(diff)]

        return rect

    def four_point_transform(self):

        rect = self.order_points()
        (tl, tr, br, bl) = rect

        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))

        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))

        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth -1, maxHeight - 1],
            [0, maxHeight]
        ], dtype='float32')

        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(self.image, M, (maxWidth, maxHeight))

        cv2.imshow('image', self.image)
        cv2.imshow('warped', warped)
        cv2.waitKey(0)

        return warped


test = Rotation(False)
test.four_point_transform()
