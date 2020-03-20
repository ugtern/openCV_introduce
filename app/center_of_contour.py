import argparse
import imutils
import cv2
from cv_methods.preload import MainCVClass


class OpenCVTest:

    def __init__(self, resize):

        self.image = MainCVClass(resize).image

    def get_a_thresh(self):

        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]

        cv2.imshow("thresh", thresh)
        cv2.waitKey(0)

        return thresh

    def get_contours(self):

        # find contours in the thresholded image

        cnts = cv2.findContours(self.get_a_thresh().copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        
        print(len(cnts))

        if len(cnts) < 2:

            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            thresh = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY_INV)[1]
            cv2.imshow('thresh', thresh)
            cv2.waitKey(0)
    
            cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            
            print(len(cnts))
            
        # loop over the contours
        for c in cnts:
            
            # compute the center of the contour
            M = cv2.moments(c)
            print(M['m10'], M['m00'], M['m01'])
            
            try:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                
            except:
                continue
                
            else:
                
                # draw the contour and center of the shape on the image
                cv2.drawContours(self.image, [c], -1, (0, 255, 0), 2)
                cv2.circle(self.image, (cX, cY), 7, (255, 255, 255), -1)
                cv2.putText(self.image, "center", (cX - 20, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
        # show the image
        cv2.imshow("Image", self.image)
        cv2.waitKey(0)


test = OpenCVTest(True)
# test.get_a_thresh()
test.get_contours()
