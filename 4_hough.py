from calendar import c
import cv2 as cv
import numpy as np
import numba as nb

cap = cv.VideoCapture(1)
if not cap.isOpened:
    print('--(!)Error opening video capture')
    exit(0)
while True:
    ret, frame = cap.read()
    if frame is None:
        print('--(!) No captured frame -- Break!')
        break

    img = frame.copy()
    img_lines = img.copy()
    img_lines_p = img.copy()
    img_circles = img.copy()
    img_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    img_edges = cv.Canny(img_gray, 200, 100)
    
    lines = cv.HoughLines(img_edges, 1, np.pi / 180, 150, None, 0, 0)
    

    
    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
            pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
            cv.line(img_lines, pt1, pt2, (0,0,255), 3, cv.LINE_AA)
    
    linesP = cv.HoughLinesP(img_edges, 1, np.pi / 180, 75, None, 50, 10)
    
    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv.line(img_lines_p, (l[0], l[1]), (l[2], l[3]), (255,0,0), 3, cv.LINE_AA)

    circles = cv.HoughCircles(img_gray, cv.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=20, maxRadius=40)
    if circles is not None:
        for i in circles[0, :]:
            # draw the outer circle
            cv.circle(img_circles, (int(i[0]), int(i[1])), int(i[2]), (0, 255, 0), 2)
            # draw the center of the circle
            cv.circle(img_circles, (int(i[0]), int(i[1])), int(i[2]), (0, 0, 255), 3)
    cv.imshow('Image Circles', img_circles)
    cv.imshow('Image Prob', img_lines_p)
    cv.imshow('Image', img_lines)

    cv.waitKey(1)