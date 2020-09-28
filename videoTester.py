import os
import cv2
import numpy as np
import faceRecognition as fr
from twilio.rest import Client
account_sid="AC5deba8411242413c04e2ae0e2d755891"
auth_token="1f8b66cb03354aa25568ab01a80ebfd9"
client=Client(account_sid,auth_token)
client.messages.create(
    to="+919482548971",
    from_="+12059315464",
    body="It seems some one entering to your home"
    )

#This module captures images via webcam and performs face recognition
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('trainingData.yml')#Load saved training data

name = {0 : "not allowed",1 : "allowed"}


cap=cv2.VideoCapture(0)

while True:
    ret,test_img=cap.read()# captures frame and returns boolean value and captured image
    faces_detected,gray_img=fr.faceDetection(test_img)



    for (x,y,w,h) in faces_detected:
      cv2.rectangle(test_img,(x,y),(x+w,y+h),(255,0,0),thickness=1)

    resized_img = cv2.resize(test_img, (1000, 700))
    cv2.imshow('face detection',resized_img)
    cv2.waitKey(10)


    for face in faces_detected:
        (x,y,w,h)=face
        roi_gray=gray_img[y:y+w, x:x+h]
        label,confidence=face_recognizer.predict(roi_gray)#predicting the label of given image
        print("confidence:",confidence)
        print("label:",name[label])
        fr.draw_rect(test_img,face)
        predicted_name=name[label]
        
        fr.put_text(test_img,predicted_name,x,y)


    resized_img = cv2.resize(test_img, (1000, 700))
    cv2.imshow('HOME SECURITY SYSTEM-Team 18',resized_img)
    if cv2.waitKey(10) == ord('q'):#wait until 'q' key is pressed
        break


cap.release()
cv2.destroyAllWindows
