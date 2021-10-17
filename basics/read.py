import cv2 as cv

img = cv.imread('Photos/cat.jpeg')
"""
#name of window, matrix of px
cv.imshow('Cat', img)
cv.waitKey(0)
"""

#takes integer if use camera or path to video file
capture = cv.VideoCapture('Videos/dog.mp4')
while True:
    isTrue, frame = capture.read()   
    cv.imshow('Video', frame)
    if cv.waitKey(20) & 0xFF ==ord('d'):
        break

capture.release()
cv.destroyAllWindows()
