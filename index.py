import cv2
import imutils


class CvTests:
    def __init__(self, filename):

        self.image = cv2.imread(filename)
        (self.h, self.w, self.d) = self.image.shape

        print("width={}, height={}, depth={}".format(self.h, self.w, self.d))

    def get_color(self):
        # get color in the checked pixel

        (r, g, b) = self.image[50, 150]

        print("R={}, G={}, B={}".format(r, g, b))

    def show_image(self):
        # show all image

        cv2.imshow('Image', self.image)
        cv2.waitKey(0)

    def show_images_part(self):
        # show checked part of image

        roi = self.image[60:180, 310:410]

        cv2.imshow("ROI", roi)
        cv2.waitKey(0)

    def resize_image(self, new_size):
        # show resized image

        resized = cv2.resize(self.image, new_size)

        cv2.imshow("resized image", resized)
        cv2.waitKey(0)

    def resize_with_imutils(self, new_size):
        # resized with saved the start proportions of the image

        r = new_size / self.w
        dim = (int(new_size), int(self.h*r))
        resized = cv2.resize(self.image, dim)

        cv2.imshow("resized proportionals", resized)
        cv2.waitKey(0)

    def rotate_image(self, degrees, scale):
        # rotate image

        center = (self.w // 2, self.h // 2)
        M = cv2.getRotationMatrix2D(center, degrees, scale)
        rotated = cv2.warpAffine(self.image, M, (self.w, self.h))

        cv2.imshow("Rotated image", rotated)
        cv2.waitKey(0)


test = CvTests('jp.jpg')
test.get_color()
# test.show_image()
# test.show_images_part()
# test.resize_image((200, 200))
# test.resize_with_imutils(200.0)
test.rotate_image(180, 0.5)

