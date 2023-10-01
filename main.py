import cv2

#init los cascados
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

#start up the webam
cap = cv2.VideoCapture(0)


cv2.namedWindow('Face Detection', cv2.WINDOW_NORMAL)
#resize it 
cv2.resizeWindow('Face Detection', 800, 600)  
launched = 1
while True:

    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    

    for (x, y, w, h) in faces:

        face_roi = frame[y:y+h, x:x+w]

        face_roi = cv2.GaussianBlur(face_roi, (0, 0), 30)

        frame[y:y+h, x:x+w] = face_roi
        
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        
        eyes = eye_cascade.detectMultiScale(face_roi)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(face_roi, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
    
    text = "Foodcam: Press Q to quit"
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow('Face Detection', frame)
    
    key = cv2.waitKey(1)
    #27 is the ascii code for ESC
    if key == ord('q') or key == 27:  
        print("Program Quit")
        break
    if launched == 1:
        print("Window Launched. Please press Q to quit")
        launched = 2

cap.release()
cv2.destroyAllWindows()
