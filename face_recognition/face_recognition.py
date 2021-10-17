import numpy as np
import cv2 as cv
import os
import random

DIR = 'Faces/val/shaffled' 

face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

features = np.load('features.npy', allow_pickle=True)
labels = np.load('labels.npy')
people = np.load('people.npy')

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')

# Picking an image to check the recognizer
dir = os.listdir(DIR)
img_index = random.randrange(0, len(dir))
img = cv.imread(os.path.join(DIR, dir[img_index]))

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
faces_rect = face_cascade.detectMultiScale(gray, 1.1, 4)

for (x, y, w, h) in faces_rect:
    face_roi = gray[y:y+h, x:x+w]
    
    label, confidence = face_recognizer.predict(face_roi)
    print(f'Label = {label} with a confidence of {confidence}')

    cv.putText(img, str(people[label]), (20, 20), cv.FONT_HERSHEY_COMPLEX, 1.0, (0, 255, 0), thickness=2)
    cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), thickness=2)

cv.imshow('Detected Face', img)

cv.waitKey(6000)