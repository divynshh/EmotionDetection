import cv2
import os

video_capture = cv2.VideoCapture(0)
# Load the cascade
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')  

def clickCropSave(image):

    # Read the input image
    #img = cv2.imread('sad.jpg')

    # Convert into grayscale
    img = image;
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(50, 50),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Draw rectangle around the faces and crop the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
    faces = img[y:y + h, x:x + w]
    #cv2.imshow("face",faces)
    cv2.imwrite('face.jpg', faces)

    os.system('python predictEmotion.py')
    # Display the output
    #cv2.imwrite('detcted.jpg', img)
    #cv2.imshow('img', img)
    #cv2.waitKey()
        







while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect faces


    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(50, 50),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    print (faces)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('c'):
        img = frame
        clickCropSave(img)
        break

        

            
video_capture.release()

