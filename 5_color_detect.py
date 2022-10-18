import cv2 as cv
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import sklearn.cluster as sklc
import numba as nb

fig_bgr = plt.figure(1)
fig_hsv = plt.figure(2)
fig_xyz = plt.figure(3)
ax_bgr = fig_bgr.add_subplot(projection='3d')
ax_hsv = fig_hsv.add_subplot(projection='3d')
ax_xyz = fig_xyz.add_subplot(projection='3d')


@nb.jit(nopython=True)
def segment(idx_arr, centers, frame_seg_hsv, rows, cols):
    for i in nb.prange(rows * cols):
        row = i // cols
        col = i % cols
        idx = idx_arr[row * cols + col]
        frame_seg_hsv[row, col, :] = centers[idx]
    return frame_seg_hsv

@nb.jit(nopython=True)
def get_xyz(frame_hsv, rows, cols):
    frame_xyz = np.zeros((rows, cols, 3), dtype=np.float64)
    for i in nb.prange(rows * cols):
        row = i // cols
        col = i % cols
        h = 2.0 * np.pi * frame_hsv[row, col, 0] / 179.0
        s = frame_hsv[row, col, 1] / 255
        v = frame_hsv[row, col, 2] / 255
        x = v * s * np.cos(h)
        y = v * s * np.sin(h)
        z = v
        frame_xyz[row, col, :] = np.array([x, y, z])
    return frame_xyz

def vectorize(img):
    vec_x = img[:, :, 0].flatten()
    vec_y = img[:, :, 1].flatten()
    vec_z = img[:, :, 2].flatten()
    x = np.vstack((vec_x, vec_y, vec_z))
    return x

cap = cv.VideoCapture(0)
if not cap.isOpened:
    print('--(!)Error opening video capture')
    exit(0)
flag = False
km = sklc.KMeans(n_clusters=8, init='k-means++')
while True:
    ret, frame = cap.read()
    if frame is None:
        print('--(!) No captured frame -- Break!')
        break
    downscale = 8
    cv.imshow('Frame', frame)
    frame = cv.resize(frame, (frame.shape[1] // downscale, frame.shape[0] // downscale), interpolation=cv.INTER_NEAREST)
    frame_hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    frame_xyz = get_xyz(frame_hsv, frame_hsv.shape[0], frame_hsv.shape[1])
    x = vectorize(frame)
    x_hsv = vectorize(frame_hsv)
    x_xyz = vectorize(frame_xyz)

    c = cv.waitKey(1)
    x_color = [(r / 255, g / 255, b / 255) for b, g, r in zip(x[0, :], x[1, :], x[2, :])]
    if c == 27:
        break
    elif c == ord(' '):
        kmeans = km.fit(x_hsv.T)
        ax_bgr.scatter3D(x[0, :], x[1, :], x[2, :], c=x_color)
        ax_hsv.scatter3D(x_hsv[0, :], x_hsv[1, :], x_hsv[2, :], c=x_color)
        ax_xyz.scatter3D(x_xyz[0, :], x_xyz[1, :], x_xyz[2, :], c=x_color)
        ax_hsv.scatter3D(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], km.cluster_centers_[:, 2], c='k', marker='o', s=100)
        plt.show()
        flag = True
    
    if flag:
        idx_arr = km.predict(x_hsv.T)
        frame_seg_hsv = np.zeros_like(frame)
        rows = frame_seg_hsv.shape[0]
        cols = frame_seg_hsv.shape[1]
        frame_seg = segment(idx_arr, km.cluster_centers_, frame_seg_hsv, rows, cols)
        frame_seg = cv.cvtColor(frame_seg_hsv, cv.COLOR_HSV2BGR)
        frame_seg = cv.resize(frame_seg, (frame_seg.shape[1] * downscale, frame_seg.shape[0] * downscale), interpolation=cv.INTER_NEAREST)
        cv.imshow('Segmentation', frame_seg)