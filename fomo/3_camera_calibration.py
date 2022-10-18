import numpy as np
import cv2 as cv
import glob
# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((9 * 6, 3), np.float32)
objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.



cap = cv.VideoCapture(1)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    ret, img = cap.read()
    
    if not ret:
        break

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, corners = cv.findChessboardCorners(gray, (9, 6), None)

    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)

        corners_draw = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        cv.drawChessboardCorners(img, (9, 6), corners_draw, ret)

    cv.imshow('img', img)
    c = cv.waitKey(1)
    if c == ord('q'):
        break

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

np.save("ret.npy", ret)
np.save("mtx.npy", mtx)
np.save("dist.npy", dist)
np.save("rvecs.npy", rvecs)
np.save("tvecs.npy", tvecs)

while True:
    ret, img = cap.read()
    
    if not ret:
        break

    h,  w = img.shape[:2]
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    x, y, w, h = roi
    img_undistorted = cv.undistort(img, mtx, dist, None, newcameramtx)[y:y+h, x:x+w]
    cv.imshow('img', img)
    cv.imshow('img_undistorted', img_undistorted)
    c = cv.waitKey(1)

    if c == ord('q'):
        break


cv.destroyAllWindows()