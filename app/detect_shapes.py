from cv_methods.shape_detector import ShapeDetector
from cv_methods.colorlabeler import ColorLabeler
from cv_methods.preload import MainCVClass
import argparse
import imutils
import cv2


class DetectShapes():
    def __init__(self, resize):

        self.image = MainCVClass(resize).image

    def threshold_image(self):

        # load the image and resize it to a smaller factor so that
        # the shapes can be approximated better

        resized = imutils.resize(self.image, width=300)
        ratio = self.image.shape[0] / float(resized.shape[0])

        # convert the resized image to grayscale, blur it slightly,
        # and threshold it

        blurred = cv2.GaussianBlur(resized, (5, 5), 0)
        gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
        lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
        thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY)[1]

        cv2.imshow('thresh', thresh)
        cv2.waitKey(0)

        # find contours in the thresholded image and initialize the
        # shape detector

        contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)

        contours = imutils.grab_contours(contours)
        shape_detector = ShapeDetector()
        color_detector = ColorLabeler()

        for contour in contours:

            M = cv2.moments(contour)

            try:

                cX = int((M['m10'] / M['m00']) *ratio)
                cY = int((M['m01'] / M['m00']) *ratio)
                shape = shape_detector.detect(contour)
                color = color_detector.label(lab, contour)

            except:
                continue

            else:
                contour = contour.astype('float')
                contour *= ratio
                contour = contour.astype('int')
                text = "{} {}".format(color, shape)
                cv2.drawContours(self.image, [contour], -1, (0, 255, 0), 2)
                cv2.putText(self.image, text, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        cv2.imshow('complite', self.image)
        cv2.waitKey(0)


test = DetectShapes(False)
test.threshold_image()
