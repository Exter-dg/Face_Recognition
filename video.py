import cv2
import os
import numpy as np 
import csv
import faceRecognition as fr 


face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('trainingData.yml')


name = {0:"Parth",1:"Khushbu",2:"Miraj",3:"Urja"}

cap = cv2.VideoCapture(0)

flag = False

while True:
    flag = False
    ret,test_img = cap.read()
    faces_detected,gray_img = fr.faceDetection(test_img)

    for (x,y,w,h) in faces_detected:
        cv2.rectangle(test_img,(x,y),(x+w,y+h),(255,0,0),thickness=7)
    
    #resized_img = cv2.resize(test_img, (1000,700))
    #cv2.imshow('Face Detection ',resized_img)
    #cv2.waitKey(10)

    for face in faces_detected:
        (x,y,w,h) = face
        roi_gray = gray_img[y:y+h , x:x+h]
        label,confidence = face_recognizer.predict(roi_gray)
        print("Confidence : ",confidence)
        print("Label : ",label)
        fr.draw_rect(test_img,face)
        predicted_name = name[label]
        if confidence<50 :
            fr.put_text(test_img,predicted_name,x,y)
            flag = True



    
    resized_img = cv2.resize(test_img,(1000,700))
    cv2.imshow('Face Recognition',resized_img)

    if(flag):

        found = False #person not in house yet
        status=""


        # csv file of format -> Name , entry/exit 
        with open('entry.csv','r') as file:
            reader = csv.reader(file, delimiter=',')
            for row in reversed(list(reader)):
                s1=row[0]
                s2=row[1]
                if(s1 == predicted_name):
                    found = True
                    status=s2
                    break
        
        with open('entry.csv', 'a',newline='') as file:
            if(found):
                if(s2=="entry"):
                    s2="exit"
                else:
                    s2="entry"
                data = [predicted_name,s2]
                writer = csv.writer(file,delimiter = ',')
                writer.writerow(data)
                
            else:
                

                data = [predicted_name,"entry"]
                writer = csv.writer(file,delimiter = ',')
                writer.writerow(data)




        cv2.waitKey(6000) # wait for 3 seconds if face is recognized
        print("Parth Detected")
    if cv2.waitKey(10) == ord('n'):
        break

cap.release()
cv2.destroyAllWindows