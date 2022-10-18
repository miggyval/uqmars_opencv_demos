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
        templates.append(img_dilated[y:y+h, x:x+w])

    cv.namedWindow("Location")
    cv.namedWindow("Template")
    cv.namedWindow("Result")
    while True:
        for rank in ranks:
            for suit, template in zip(suits, templates):

                img = imgs[(suit, rank)]
                img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

                _, img_thresh = cv.threshold(img_gray, 200, 255, cv.THRESH_BINARY_INV)
                img_edge = cv.Canny(img_gray, 100, 200)
                kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
                img_dilated = cv.dilate(img_edge, kernel)
                method = cv.TM_CCOEFF_NORMED
                powers = np.linspace(-0.7, 0, 50)
                scales = 10 ** powers
                rotations = np.linspace(0, 360, 4)
                vals = np.zeros((scales.shape[0], rotations.shape[0]))
                imgs_located = []
                for idx, scale in enumerate(scales):
                    scaled_template = cv.resize(template, (0, 0), fx=scale, fy=scale)
                    for idy, rotation in enumerate(rotations):
                        rotated_template = rotate(scaled_template, rotation)
                        res = cv.matchTemplate(img_dilated, rotated_template, cv.TM_CCOEFF_NORMED)
                        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
                        img_copy = img.copy()
                        if method == cv.TM_SQDIFF:
                            tl = min_loc
                            val = min_val
                        else:
                            tl = max_loc
                            val = max_val
                        vals[idx, idy] = val
                        cv.rectangle(img_copy, min_loc, (tl[0] + template.shape[1], tl[1] + template.shape[0]), 128, 1, cv.LINE_AA)
                        imgs_located.append(img_copy)
                        cv.imshow("Location", img_copy)
                        cv.imshow("Skel", img_dilated)
                        cv.imshow("Template", rotated_template)
                        cv.imshow("Result", res / max_val)
                        c = cv.waitKey(1)
                
                if False:
                    plt.subplot(4, 1, 1)
                    plt.plot(scales, vals[:, 0])
                    plt.subplot(4, 1, 2)
                    plt.plot(scales, vals[:, 1])
                    plt.subplot(4, 1, 3)
                    plt.plot(scales, vals[:, 2])
                    plt.subplot(4, 1, 4)
                    plt.plot(scales, vals[:, 3])
                    plt.show()
                 
                cv.imshow("Located", imgs_located)
                c = cv.waitKey()