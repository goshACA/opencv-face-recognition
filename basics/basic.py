import cv2 as cv

img = cv.imread('Photos/cat.jpeg')
"""
# Converting to grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)

cv.imshow('Cat', img)
"""
# Blur 
blur = cv.GaussianBlur(img, (11, 11), cv.BORDER_DEFAULT)
cv.imshow('Cat', blur)



# Edge Cascade
canny = cv.Canny(blur, 125, 175)
cv.imshow('Blur', canny)

# Dilating the img
dilated  = cv.dilate(canny, (3, 3), iterations=3)

cv.imshow('Dilated', canny)

# Eroding
eroded = cv.erode(dilated, (3, 3), iterations=3)
cv.imshow('Eroded', canny)

# Resize 
resized = cv.resize(img, (500, 500), interpolation=cv.INTER_CUBIC)
cv.imshow('Resized', resized)

# Cropping
cropped = img[50:200, 200:400]
cv.imshow('Cropped', cropped)

cv.waitKey(3000)