import imutils
import cv2
import argparse


class OpenCVTest:

    def __init__(self):

        ap = argparse.ArgumentParser()
        ap.add_argument("-i", "--image", required=True, help="path to input image")
        args = vars(ap.parse_args())

        self.image = cv2.imread(args["image"])

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

    def thresholding(self):
        thres = cv2.threshold(self.turn_image_to_gray(False), 225, 255, cv2.THRESH_BINARY_INV)[1]
        cv2.imshow("Thresh", thres)
        cv2.waitKey(0)


test = OpenCVTest()
# test.show_image()
# test.turn_image_to_gray(True)
# test.edge_datection()
test.thresholding()
