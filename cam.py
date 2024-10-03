import ultralytics
import cv2
from ultralytics import YOLO
import pandas as pd
import time

 
cam = cv2.VideoCapture(0) # Open camera
model = YOLO('480.pt')  # Load model
label = open('label.txt', 'r')
labels = label.read().split('\n')

counter, fps = 0,0
fps_avg_frame_count = 10
start_time = time.time()
while True:
    ret, img = cam.read()
    result = model.predict(img, conf=0.3)  # Inference
    a=result[0].boxes.data
    px=pd.DataFrame(a).astype("float")
    #Menghitung FPS
    counter += 1
    if counter % fps_avg_frame_count == 0:
        endtime = time.time()
        fps = fps_avg_frame_count / (endtime - start_time)
        start_time = time.time()    
    fps_text = 'FPS = {:.1f}'.format(fps)
    cv2.putText(img, fps_text, (24,20), cv2.FONT_HERSHEY_PLAIN,1, (0,0,255), 2)
    #Visualisasi Bounding Box dan Label
    for index,row in px.iterrows():
        x1=int(row[0])
        y1=int(row[1])
        x2=int(row[2])
        y2=int(row[3])
        conf_score=(row[4])
        index=int(row[5])
        label_name=labels[index]
        label_text = f'{label_name} {conf_score:.2f}'
        if index == 0:
                cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
                cv2.rectangle(img,(x1,y1),(x2,y1-20),(0,255,0),-2)
                cv2.putText(img,label_text,(x1,y1-5),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2)
                
        else:
                cv2.rectangle(img,(x1,y1),(x2,y2),(0,0,255),2)
                cv2.rectangle(img,(x1,y1),(x2,y1-20),(0,0,255),-2)
                cv2.putText(img,label_text,(x1,y1-5),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2)    
    cv2.imshow('Image', img)    
    if cv2.waitKey(1) == ord('q'):
        break
cam.release()
cv2.destroyAllWindows()
