import imutils
import cv2
import argparse
from cv_methods.preload import MainCVClass


class OpenCVTest():

    def __init__(self, resize):
        self.image = MainCVClass(resize).image

    def show_image(self):

        cv2.imshow("Image", self.image)
        cv2.waitKey(0)

    def turn_image_to_gray(self, show):

        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        if show:
            cv2.imshow("Gray", gray)
            cv2.waitKey(0)
        else:
            return gray

    def edge_datection(self):

        edged = cv2.Canny(self.turn_image_to_gray(False), 30, 150)

        cv2.imshow("edged", edged)
        cv2.waitKey(0)
        return edged

    def thresholding(self):
        thresh = cv2.threshold(self.turn_image_to_gray(False), 225, 255, cv2.THRESH_BINARY_INV)[1]
        cv2.imshow("Thresh", thresh)
        cv2.waitKey(0)
        return thresh
        
    def find_contours(self):
        cnts = cv2.findContours(self.edge_datection().copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        output = self.image.copy()

        for c in cnts:
            cv2.drawContours(output, [c], -1, (240, 0, 159), 3)

        text = "founded {} objects".format(len(cnts))
        cv2.putText(output, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 0, 159), 2)

        cv2.imshow("Countours", output)
        cv2.waitKey(0)

    def erosion(self):

        mask = self.thresholding().copy()
        mask = cv2.erode(mask, None, iterations=5)
        cv2.imshow("Eroded", mask)
        cv2.waitKey(0)

    def dilation(self):

        mask = self.thresholding().copy()
        mask = cv2.dilate(mask, None, iterations=5)
        cv2.imshow("Eroded", mask)
        cv2.waitKey(0)

    def masking(self):

        mask = self.thresholding().copy()
        output = cv2.bitwise_and(self.image, self.image, mask=mask)
        cv2.imshow("Output", output)
        cv2.waitKey(0)


test = OpenCVTest(False)
# test.show_image()
# test.turn_image_to_gray(True)
# test.edge_datection()
# test.thresholding()
# test.find_contours()
# test.erosion()
# test.dilation()
# test.masking()
