import cv2 as cv

img = cv.imread('Photos/cat.jpeg')
"""
#name of window, matrix of px
cv.imshow('Cat', img)
cv.waitKey(0)
"""

def rescaleFrame(frame, scale=0.75):
    # works for images, videos, live videos
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

def changeRes(width, height):
    # works for live videos
    capture.set(3, width)
    capture.set(4, height)

#takes integer if use camera or path to video file
capture = cv.VideoCapture('Videos/dog.mp4')
while True:
    isTrue, frame = capture.read()
    frame_resized = rescaleFrame(frame)
     
    cv.imshow('Video', frame)
    cv.imshow('Video Resized', frame_resized)

    if cv.waitKey(20) & 0xFF ==ord('d'):
        break

capture.release()
cv.destroyAllWindows()
