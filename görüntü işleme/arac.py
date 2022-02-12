import cv2
cascade_src='haarcascade_car.xml'
video_src='Traffic.avi'

cap=cv2.VideoCapture(video_src) 
cars_cascade=cv2.CascadeClassifier(cascade_src) 

while True:
    _, img=cap.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) 
    cv2.imshow('gri_hal', gray)
    araclar=cars_cascade.detectMultiScale(gray,1.1,4,minSize=(0, 100)) 

    for(x,y,w,h) in araclar:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2)
        cv2.putText(img, "Obje", (x,y-3), cv2.QT_FONT_NORMAL, 0.5, (255, 255, 255), 1)
    cv2.imshow('CIKTI',img)
    if cv2.waitKey(33) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()