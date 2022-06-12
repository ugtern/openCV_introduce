from imutils.perspective import four_point_transform
from imutils import contours as contours_lib

import cv2
import imutils
import numpy as np

DIGITS_LOOKUP = {
    (1, 1, 1, 0, 1, 1, 1): 0,
    (0, 0, 1, 0, 0, 1, 0): 1,
    (1, 0, 1, 1, 1, 1, 0): 2,
    (1, 0, 1, 1, 0, 1, 1): 3,
    (0, 1, 1, 1, 0, 1, 0): 4,
    (1, 1, 0, 1, 0, 1, 1): 5,
    (1, 1, 0, 1, 1, 1, 1): 6,
    (1, 0, 1, 0, 0, 1, 0): 7,
    (1, 1, 1, 1, 1, 1, 1): 8,
    (1, 1, 1, 1, 0, 1, 1): 9
}

image = cv2.imread('example.jpg')

image = imutils.resize(image, height=500)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(blurred, 50, 200, 255)

cv2.imshow('image', image)
cv2.waitKey(0)

cv2.imshow('edged', edged)
cv2.waitKey(0)

contours = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(contours)
contours = sorted(contours, key=cv2.contourArea, reverse=False)
displayContour = None

print(len(contours))

for contour in contours:
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 *peri, True)

    if len(approx) == 4:
        displayContour = approx
        break

warped = four_point_transform(gray, displayContour.reshape(4, 2))
output = four_point_transform(image, displayContour.reshape(4, 2))

cv2.imshow('warped', warped)
cv2.waitKey(0)

thresh = cv2.threshold(warped, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 5))
thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

cv2.imshow('thresh', thresh)
cv2.waitKey(0)

contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(contours)
digitContours = []

for contour in contours:

    (x, y, w, h) = cv2.boundingRect(contour)

    if w >= 15 and (h >= 30 and h <= 40):
        digitContours.append(contour)

digitContours = contours_lib.sort_contours(digitContours, method='left-to-right')[0]
digits = []

for contour in digitContours:
    (x, y, w, h) = cv2.boundingRect(contour)
    roi = thresh[y:y + h, x:x + w]

    (roiH, roiW) = roi.shape
    (dW, dH) = (int(roiW * 0.25), int(roiH * 0.15))
    dHC = int(roiH * 0.05)

    segments = [
        ((0, 0), (w, dH)),
        ((0, 0), (dW, h // 2)),
        ((w - dW, 0), (w, h//2)),
        ((0, (h // 2) - dHC), (w, (h // 2) + dHC)),
        ((0, h // 2), (dW, h)),
        ((w - dW, h // 2), (w, h)),
        ((0, h - dH), (w, h))
    ]
    on = [0] * len(segments)

    # loop over the segments
    for (i, ((xA, yA), (xB, yB))) in enumerate(segments):
        # extract the segment ROI, count the total number of
        # thresholded pixels in the segment, and then compute
        # the area of the segment
        segROI = roi[yA:yB, xA:xB]
        total = cv2.countNonZero(segROI)
        area = (xB - xA) * (yB - yA)
        # if the total number of non-zero pixels is greater than
        # 50% of the area, mark the segment as "on"
        if total / float(area) > 0.5:
            on[i] = 1
    # lookup the digit and draw it on the image
    digit = DIGITS_LOOKUP[tuple(on)]
    digits.append(digit)
    cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 1)
    cv2.putText(output, str(digit), (x - 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

print(u"{}{}.{} \u00b0C".format(*digits))
cv2.imshow("Input", image)
cv2.imshow("Output", output)
cv2.waitKey(0)
