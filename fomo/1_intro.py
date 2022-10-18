import cv2
import urllib3
import numpy as np


# Get the Image
url = 'https://upload.wikimedia.org/wikipedia/en/7/7d/Lenna_%28test_image%29.png'
response = urllib3.PoolManager().request('GET', url)
with open('lena.png', 'wb') as download:
    download.write(response.data)

img = cv2.imread('lena.png', cv2.IMREAD_COLOR)

img_b = img.copy()
img_b[:, :, 1] = 0
img_b[:, :, 2] = 0

img_g = img.copy()
img_g[:, :, 2] = 0
img_g[:, :, 0] = 0

img_r = img.copy()
img_r[:, :, 0] = 0
img_r[:, :, 1] = 0

cv2.imshow('Lena', np.hstack([img, img_b, img_g, img_r]))
cv2.waitKey()