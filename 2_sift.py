import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def rotate(image, angle):
    (h, w) = image.shape[:2]

    center = (w / 2, h / 2)
    scale = 1.0

    # Perform the rotation
    M = cv.getRotationMatrix2D(center, angle, scale)
    rotated = cv.warpAffine(image, M, (w, h))

    return rotated


def partition(arr, l, r):
    areas = [cv.contourArea(contour) for contour in arr]
    x = areas[r]
    i = l
    for j in range(1, r):
        if areas[j] >= x:
            temp = arr[i]
            arr[i] = arr[j]
            arr[j] = temp
            i += 1
    temp = arr[i]
    arr[i] = arr[r]
    arr[r] = temp
    return i

def quick_select(contours, l, r, k):
    if (k > 0 and k <= r - l + 1):
        index = partition(contours, l, r)
        if (index - l == k - 1):
            return contours[index]
        if (index - l > k - 1):
            return quick_select(contours, l, index - 1, k)

        return quick_select(contours, index + 1, r, r, k - index + l - 1)
        

if __name__ == "__main__":

    suits = ['spades', 'hearts', 'diamonds', 'clubs']
    ranks = ['2', '3', '4', '5', '6', '7', '8', '9', 'jack', 'queen', 'king', 'ace']

    imgs = {}
    for suit in suits:
        for rank in ranks:
            imgs[(suit, rank)] = cv.imread("cards/" + rank + "_of_" + suit + ".png", cv.IMREAD_COLOR)
    
    templates = []
    templates_gray = []
    for suit in suits:
        rank = 'queen'
        img = imgs[(suit, rank)]
        img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        _, img_thresh = cv.threshold(img_gray, 200, 255, cv.THRESH_BINARY_INV)
        img_edge = cv.Canny(img_gray, 100, 200)
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
        img_dilated = cv.dilate(img_edge, kernel)
        contours, hierarchy = cv.findContours(img_thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        arr = [contour for contour in contours]
        img_over = img.copy()
        contour = quick_select(arr, 0, len(arr) - 1, 1)
        cv.drawContours(img_over, [contour], 0, 255, 3)
        x, y, w, h = cv.boundingRect(contour)
        templates.append(img[y:y+h, x:x+w])
        templates_gray.append(img_dilated[y:y+h, x:x+w])

    cv.namedWindow("Image")
    cv.namedWindow("Template")
    cv.namedWindow("Output")
    sift = cv.SIFT_create()
    while True:
        for rank in ranks:
            for suit, template, template_gray in zip(suits, templates, templates_gray):

                img = imgs[(suit, rank)]
                img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
                _, img_thresh = cv.threshold(img_gray, 200, 255, cv.THRESH_BINARY_INV)
                img_edge = cv.Canny(img_gray, 100, 200)

                kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
                img_dilated = cv.dilate(img_edge, kernel)

                kp_gray, des_gray = sift.detectAndCompute(img_dilated, None)
                kp_template, des_template = sift.detectAndCompute(template_gray, None)

                bf = cv.BFMatcher(cv.NORM_L1, crossCheck=True)

                matches = bf.match(des_gray, des_template)
                matches = sorted(matches, key = lambda x:x.distance)

                img_out = cv.drawMatches(img, kp_gray, template, kp_template, matches, None, flags=2)
                cv.imshow("Image", img_dilated)
                cv.imshow("Template", template_gray)
                cv.imshow("Output", img_out)
                cv.waitKey()

                