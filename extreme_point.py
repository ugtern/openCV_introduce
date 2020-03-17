import cv2
import imutils
import argparse


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

    def to_gray(self):

        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        cv2.imshow('image gray', gray)
        cv2.waitKey(0)

        thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)

        cv2.imshow('image contour', thresh)
        cv2.waitKey(0)

        contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        contour = max(contours, key=cv2.contourArea)

        # determine the most extreme points along the contour
        extLeft = tuple(contour[contour[:, :, 0].argmin()][0])
        extRight = tuple(contour[contour[:, :, 0].argmax()][0])
        extTop = tuple(contour[contour[:, :, 1].argmin()][0])
        extBot = tuple(contour[contour[:, :, 1].argmax()][0])

        cv2.drawContours(self.image, [contour], -1, (0, 255, 255), 2)
        cv2.circle(self.image, extLeft, 8, (0, 0, 255), -1)
        cv2.circle(self.image, extRight, 8, (0, 255, 0), -1)
        cv2.circle(self.image, extTop, 8, (255, 0, 0), -1)
        cv2.circle(self.image, extBot, 8, (255, 255, 0), -1)

        cv2.imshow('image', self.image)
        cv2.waitKey(0)


test = OpenCVTest(False)
test.to_gray()
