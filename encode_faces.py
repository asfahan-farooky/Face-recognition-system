import face_recognition
import cv2
import os
import pickle

path = 'images'
images = []
classNames = []
myList = os.listdir(path)

for cl in myList:
    img = cv2.imread(f'{path}/{cl}')
    images.append(img)
    classNames.append(os.path.splitext(cl)[0])

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

print("[INFO] Encoding images...")
encodeListKnown = findEncodings(images)

with open('encodings.pkl', 'wb') as f:
    pickle.dump((encodeListKnown, classNames), f)

print("[INFO] Encodings saved successfully.")
