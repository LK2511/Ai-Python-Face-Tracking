# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=invalid-name
# pylint: disable=trailing-newlines
# pylint: disable=trailing-whitespace
# pylint: disable=missing-final-newline

import cv2

#Load prelearned data from opencv
trained_face_data = cv2.CascadeClassifier('src\haarcascade_frontalface_default.xml')

""" Detect Faces in Webcam """
#Capture WebCam
webcam = cv2.VideoCapture("src\VideoTest.mp4")

#Iterate all Frames of WebCam
while True:
    
    #Read current Frame
    succesful_frame_read, frame = webcam.read()

    #Convert frame to grayscale
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #Detect Faces
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

    #Draw Rectangle around face
    for (x,y,w,h) in face_coordinates:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
    
    key = cv2.waitKey(1)
    cv2.imshow('Face Detector', frame)

    #Stop if Q key is pressed
    if key==81 or key==113:
        break

#Release webcam
webcam.release()
""" """

""" For photo
#Test picture
img = cv2.imread('src\T-Pose.jpg')

#Convert img to grayscale
grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#Detect Faces
face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

#Draw Rectangle around face
for (x,y,w,h) in face_coordinates:
    cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)

#print(face_coordinates)

#Display Face with Rectangle
cv2.imshow('Face Detector', img)
cv2.waitKey()
"""

print("Code Completed")
