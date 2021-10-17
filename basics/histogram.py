import cv2 as cv
import numpy as np 
import matplotlib.pyplot as plt

img = cv.imread('Photos/cats.jpeg')

cv.imshow('Cats', img) 
"""
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray) 

blank = np.zeros(img.shape[:2], dtype='uint8')
cv.imshow('Blank', blank)

mask = cv.circle(blank, (img.shape[1]//2, img.shape[0]//2 - 45), 200, 255, -1)
cv.imshow('Mask', mask)

masked = cv.bitwise_and(gray, gray, mask=mask)
cv.imshow('Masked', masked)


gray_hist = cv.calcHist([gray], [0], masked, [256], [0, 256] )
plt.figure()
plt.title('Grayscale Histogram')
plt.xlabel('Bins')
plt.ylabel('# of pixels')
plt.plot(gray_hist)
plt.xlim([0, 256])
plt.show()
"""
# colour histogram
plt.figure()
plt.title('Grayscale Histogram')
plt.xlabel('Bins')
plt.ylabel('# of pixels')
colors = ('b', 'g', 'r')
for i, col in enumerate(colors):
    hist = cv.calcHist([img], [i], None, [256], [0, 256])
    plt.plot(hist, color=col)
    plt.xlim([0, 256])
plt.show()
cv.waitKey(3000)