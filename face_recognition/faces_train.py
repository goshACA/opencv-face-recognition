import cv2 as cv
import numpy as np
import os

DIR = 'Faces/train'

features = []
labels = []

face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

people = []
for i in os.listdir(DIR):
    people.append(i)
print(people)

def create_train():
    for person in people:
        path = os.path.join(DIR, person)
        label = people.index(person)

        for img in os.listdir(path):
            img_path = os.path.join(path, img)

            img_array = cv.imread(img_path)
            gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)

            faces_rect = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

            for (x, y, w, h) in faces_rect:
                    face_roi = gray[y:y+h, x:x+w]
                    features.append(face_roi)
                    labels.append(label)
create_train()

face_recognzer = cv.face.LBPHFaceRecognizer_create()

features = np.array(features, dtype='object')
labels = np.array(labels)
people = np.array(people)

# Train the recognizer
face_recognzer.train(features, labels)
face_recognzer.save('face_trained.yml')
np.save('features.npy', features)
np.save('labels.npy', labels)
np.save('people.npy', people)




