import cv2
import os
import numpy as np 
import csv #to access csv files
import faceRecognition as fr #refer faceRecognition.py for functions' working and use
import plotly.graph_objects as go #to plot table
from datetime import datetime

#captures images via video and detects faces
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
#load previously trained dataset
face_recognizer.read('trainingData.yml')

#dictionary of known faces
name = {0:'Parth',1:'Khushbu',2:'Miraj',3:'Urja'}

#keeps a count on frequency of the person going out of door
frequencyGoingOut={'Parth':0,'Khushbu':0,'Miraj':0,'Urja':0}

cap = cv2.VideoCapture(0)

flag = False

while True:
    flag = False
    ret,test_img = cap.read()# captures frame and returns boolean value and captured image
    faces_detected,gray_img = fr.faceDetection(test_img)

    for (x,y,w,h) in faces_detected:
        cv2.rectangle(test_img,(x,y),(x+w,y+h),(255,0,0),thickness=7)
    
    #resized_img = cv2.resize(test_img, (1000,700))
    #cv2.imshow('Face Detection ',resized_img)
    #cv2.waitKey(10)
    if len(faces_detected)>0:
        textToDisplay="You are not authorise person to enter in house."
    else:
        textToDisplay=""
    for face in faces_detected:
        (x,y,w,h) = face
        roi_gray = gray_img[y:y+h , x:x+h]
        label,confidence = face_recognizer.predict(roi_gray) #predicting the label of given image
        print("Confidence : ",confidence)
        print("Label : ",label)
        fr.draw_rect(test_img,face)
        predicted_name = name[label]
        if confidence<50 : # only consider the face if it is detected with a confidence less <50 , lower confidence value depicts more accurate answer
            fr.put_text(test_img,predicted_name,x,y)
            flag = True
            textToDisplay = "Door is Open"
            timeFound = datetime.now() #2015-09-09 12:25:00.983745
            timeInString = timeFound.strftime("%H:%M:%S")#"12:25:00"
            timeFound = datetime.strptime(timeInString, "%H:%M:%S") #12:25:00



    

    if(flag): # If a face is detected with confidence value < 50

        newEntry = True
        found = False #person not in house yet
        status=""


        # csv file of format -> Name , entered in house./ went out ,time
        with open('entry.csv','r') as file:
            reader = csv.reader(file, delimiter=',')
            for row in reversed(list(reader)): # Read csv file in reverse order to check the latest entry (went out or entered)   
                    s1=row[0] #name of the person
                    s2=row[1] # status (went in/out)
                    s3=row[2] # time
                    lastFound = datetime.strptime(s3, "%H:%M:%S")
                    difference = timeFound - lastFound
                    duration_in_s = difference.total_seconds() #Difference in seconds between timeFound and last known time
                    

                    if(s1 == predicted_name):
                        if duration_in_s <4: #If it detects same face within 4 seconds, it is not a new entry and hence data will not be updated
                            found = False
                            newEntry = False
                            break
                        else:
                            newEntry = True
                            found = True
                            status=s2
                            break
        
        with open('entry.csv', 'a',newline='') as file: #make a entry if entry already existed
            if(found):
                if(s2=="entered in house."): #if went out, increment frequencyGoingOut
                    s2="went out"
                    frequencyGoingOut[predicted_name]+=1
                    if frequencyGoingOut[predicted_name]>5: # if a person goes out more than 5 times , display this message
                        textToDisplay="Stay at home in this pandemic situation and stay safe."
                        
                else:
                    s2="entered in house."


                data = [predicted_name,s2,timeInString]
                writer = csv.writer(file,delimiter = ',')
                writer.writerow(data) #insert data into csv
                
            else: #create a new entry as this person's entry doesn't exist in the csv file
                if(newEntry):   
                    data = [predicted_name,"entered in house.",timeInString]
                    writer = csv.writer(file,delimiter = ',')
                    writer.writerow(data) #insert data into csv




        #cv2.waitKey(3000) # wait for 3 seconds if face is recognized so that multiple instance of same face at the same time are not recorded
       # print("Parth Detected")

    font = cv2.FONT_HERSHEY_SIMPLEX 
  
    # Use putText() method for 
    # inserting text on video 
    cv2.putText(test_img,  
                textToDisplay,  
                (50, 50),  
                font, 1,  
                (0, 255, 255),  
                2,  
                cv2.LINE_4) 

    resized_img = cv2.resize(test_img,(1000,700))

    cv2.imshow('Face Recognition',resized_img)

    if textToDisplay is "Stay at home in this pandemic situation and stay safe.":
        cv2.waitKey(3000) #pause for 3 seconds and display the above message

    if cv2.waitKey(10) == ord('n'): #keep running until 'n' is pressed
        break

cap.release()
cv2.destroyAllWindows

#To display a table using plotly in browser
headerColor = 'grey'
rowEvenColor = 'lightgrey'
rowOddColor = 'white'

fig = go.Figure(data=[go.Table(
    header=dict(
    values=['<b>Name</b>','<b>No of times went out</b>'], #table header
    line_color='darkslategray',
    fill_color=headerColor,
    align=['left','center'],
    font=dict(color='white', size=12)),
    cells=dict(values=[['Parth','Khushbu','Miraj','Urja'], # 1st column
                       [frequencyGoingOut['Parth'],frequencyGoingOut['Khushbu'],frequencyGoingOut['Miraj'],frequencyGoingOut['Urja']]], # 2nd column
               line_color='darkslategray',
               fill_color = [[rowOddColor,rowEvenColor,rowOddColor, rowEvenColor,rowOddColor]*5], 
               align = ['left', 'center']))
])


fig.show()