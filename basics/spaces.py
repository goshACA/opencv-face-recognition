import cv2 as cv
import numpy as np 
import matplotlib.pyplot as plt
 
img = cv.imread('Photos/cats.jpeg')

cv.imshow('Cat', img)


gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)

# BGR TO HSV
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
cv.imshow('HSV', hsv)

# BGR TO L*a*b
lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
cv.imshow('LAB', lab)

#BGR TO RGB
rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
cv.imshow('RGB', lab)

plt.imshow(rgb)
plt.show()

cv.waitKey(3000)