import cv2 as cv
import argparse
import numpy as np

offset_x_l = np.random.uniform(-1.0, 1.0)
offset_y_l = np.random.uniform(-1.0, 1.0)
offset_x_r = np.random.uniform(-1.0, 1.0)
offset_y_r = np.random.uniform(-1.0, 1.0)

def detectAndDisplay(frame):
    global offset_x_l
    global offset_y_l
    global offset_x_r
    global offset_y_r
    offset_x_l += 0.9 * np.random.uniform(-1.0, 1.0)
    offset_y_l += 0.9 * np.random.uniform(-0.8, 1.0)
    offset_x_l += 0.9 * np.random.uniform(-1.0, 1.0)
    offset_y_r += 0.9 * np.random.uniform(-0.8, 1.0)
    if (offset_x_l > 1.0): offset_x_l = 1.0
    if (offset_y_l > 1.0): offset_y_l = 1.0
    if (offset_x_r > 1.0): offset_x_r = 1.0
    if (offset_y_r > 1.0): offset_y_r = 1.0
    if (offset_x_l < -1.0): offset_x_l = -1.0
    if (offset_y_l < -1.0): offset_y_l = -1.0
    if (offset_x_r < -1.0): offset_x_r = -1.0
    if (offset_y_r < -1.0): offset_y_r = -1.0
    frame = cv.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame_gray = cv.equalizeHist(frame_gray)
    #-- Detect faces
    faces = face_cascade.detectMultiScale(frame_gray)
    for (x, y, w, h) in faces:
        faceROI = frame_gray[y:y+h,x:x+w]

        face_canny = cv.Canny(frame_gray, 180, 100)[y:y+h,x:x+w]
        face_overlay = np.zeros(frame_gray.shape, dtype=np.uint8)
        face_overlay[y:y+h,x:x+w] = face_canny
        face_overlay = cv.cvtColor(face_overlay, cv.COLOR_GRAY2BGR)
        cv.copyTo(face_overlay, face_overlay, frame)
        #-- In each face, detect eyes
        eyes = eyes_cascade.detectMultiScale(faceROI)

        for (x2, y2, w2, h2) in eyes:
            eye_center = (x + x2 + w2//2, y + y2 + h2//2)
            radius = int(round((w2 + h2) * 0.3))
            frame = cv.circle(frame, eye_center, radius, (255, 255, 255), -1)
            frame = cv.circle(frame, eye_center, radius, (0, 0, 0), 4)
    
        if len(eyes) >= 2:
            xl, yl, wl, hl = eyes[0]
            xr, yr, wr, hr = eyes[1]
            iris_center_l = (int(10 * offset_x_l + x + xl + wl // 2), int(10 * offset_y_l + y + yl + hl // 2))
            iris_center_r = (int(10 * offset_x_r + x + xr + wr // 2), int(10 * offset_y_r + y + yr + hr // 2))
            frame = cv.circle(frame, iris_center_l, 10, (0, 0, 0), -1)
            frame = cv.circle(frame, iris_center_r, 10, (0, 0, 0), -1)

    cv.imshow('Capture - Face detection', frame)
parser = argparse.ArgumentParser(description='Code for Cascade Classifier tutorial.')
parser.add_argument('--face_cascade', help='Path to face cascade.', default='data/lbpcascades/lbpcascade_frontalface.xml')
parser.add_argument('--eyes_cascade', help='Path to eyes cascade.', default='data/haarcascades/haarcascade_eye_tree_eyeglasses.xml')
parser.add_argument('--camera', help='Camera divide number.', type=int, default=0)
args = parser.parse_args()
face_cascade_name = args.face_cascade
eyes_cascade_name = args.eyes_cascade
face_cascade = cv.CascadeClassifier()
eyes_cascade = cv.CascadeClassifier()
#-- 1. Load the cascades
if not face_cascade.load(cv.samples.findFile(face_cascade_name)):
    print('--(!)Error loading face cascade')
    exit(0)
if not eyes_cascade.load(cv.samples.findFile(eyes_cascade_name)):
    print('--(!)Error loading eyes cascade')
    exit(0)
camera_device = args.camera
#-- 2. Read the video stream
cap = cv.VideoCapture(1)
if not cap.isOpened:
    print('--(!)Error opening video capture')
    exit(0)
while True:
    ret, frame = cap.read()
    if frame is None:
        print('--(!) No captured frame -- Break!')
        break
    detectAndDisplay(frame)
    if cv.waitKey(1) == 27:
        break