import cv2 as cv
import numpy as np 

img = cv.imread('Photos/cats.jpeg')
cv.imshow('Cats', img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)

# Laplacian
lap = cv.Laplacian(gray, cv.CV_64F)
lap = np.uint8(np.absolute(lap))

cv.imshow('Laplacian', lap)

# Sobel
sobelx = cv.Sobel(gray, cv.CV_64F, 1, 0)
sobely= cv.Sobel(gray, cv.CV_64F, 0, 1)

cv.imshow('Soble X', sobelx)
cv.imshow('Soble Y', sobely)

combined_sobel = cv.bitwise_or(sobelx, sobely)
cv.imshow('Soble Combined', combined_sobel)

canny = cv.Canny(gray, 150, 175)
cv.imshow('Canny', canny)

cv.waitKey(3000)