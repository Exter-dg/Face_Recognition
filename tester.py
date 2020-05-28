import cv2
import os
import numpy as np 
import faceRecognition as fr 

#takes an image an performs face recognition on it
test_img = cv2.imread('TestImages/p1.jpg') #insert image
faces_detected,gray_img = fr.faceDetection(test_img)
print("faces_detected ",faces_detected)

"""for (x,y,w,h) in faces_detected:
    cv2.rectangle(test_img,(x,y),(x+w,y+h),(255,0,0),thickness=5)

resized_img=cv2.resize(test_img,(1000,700))
cv2.imshow("face detection ",resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows"""

#To train the dataset
#faces,faceID = fr.labels_for_training_data('trainingImages')
#face_recognizer = fr.train_classifier(faces,faceID)
#face_recognizer.save('trainingData.yml') #To save data so that need not to be trained again

#to load previously trained dataset stored in 'trainingData.yml'
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('trainingData.yml')

#dictionary of known faces
name = {0:"Parth",1:"Khushbu",2:"Miraj",3:"Urja"}

for face in faces_detected:
    (x,y,w,h) = face
    roi_gray = gray_img[y:y+h , x:x+h]
    label,confidence = face_recognizer.predict(roi_gray) #predicting the label of given image
    print("Confidence : ",confidence)
    print("Label : ",label)
    fr.draw_rect(test_img,face)
    predicted_name = name[label]
    if confidence>55 : # Do not consider if confidence is greater than 55, lower confidence value - more accurate results
        continue 

    fr.put_text(test_img,predicted_name,x,y)

resized_img=cv2.resize(test_img,(1000,700))
cv2.imshow("face detection ",resized_img)
cv2.waitKey(0) # wait until a key is pressed
cv2.destroyAllWindows

