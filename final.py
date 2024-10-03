import cv2
from ultralytics import YOLO
import pandas as pd
import time
import RPi.GPIO as GPIO
from RPLCD.i2c import CharLCD

# Setup GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setup(26, GPIO.OUT)  # Gunakan pin GPIO 18 sebagai output untuk lampu
# Setup LCD (sesuaikan dengan alamat I2C dan ukuran LCD Anda)
lcd = CharLCD('PCF8574', 0x27)  # Alamat I2C, sesuaikan dengan yang Anda gunakan
lcd.clear()

cam = cv2.VideoCapture(0)  # Open camera
model = YOLO('640_ncnn_model')  # Load model
label = open('label.txt', 'r')
labels = label.read().split('\n')

counter, fps = 0, 0
fps_avg_frame_count = 10
start_time = time.time()

while True:
    ret, img = cam.read()
    result = model.predict(img, conf=0.5)  # Inference
    a = result[0].boxes.data
    px = pd.DataFrame(a).astype("float")

    # Initialize detection flags
    detected_label_0 = False
    detected_label_1 = False

    # Calculate FPS
    counter += 1
    if counter % fps_avg_frame_count == 0:
        endtime = time.time()
        fps = fps_avg_frame_count / (endtime - start_time)
        start_time = time.time()
    
    fps_text = 'FPS = {:.1f}'.format(fps)
    cv2.putText(img, fps_text, (24, 20), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)

    # Visualize Bounding Box, Label, and Confidence Score
    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        conf_score = row[4]  # Get confidence score as float
        label_index = int(row[5])
        label_name = labels[label_index]

        # Set detection flags based on label index
        if label_index == 0:
            detected_label_0 = True
        elif label_index == 1:
            detected_label_1 = True

        # Concatenate label name with confidence score
        label_text = f'{label_name} {conf_score:.2f}'

        # Drawing the bounding box and label with confidence score
        if label_index == 0:
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y1), (x2, y1 - 20), (0, 255, 0), -1)
            cv2.putText(img, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        else:
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.rectangle(img, (x1, y1), (x2, y1 - 20), (0, 0, 255), -1)
            cv2.putText(img, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Control lamp and display message on LCD based on detection flags
    if detected_label_1 and not detected_label_0:
        GPIO.output(26, GPIO.HIGH)  # Turn on the lamp
        lcd.clear()
        lcd.write_string("Pelanggar")  # Display message on LCD
        lcd.cursor_pos = (1,0)
        lcd.write_string("Terdeteksi")
    else:
        GPIO.output(26, GPIO.LOW)   # Turn off the lamp
        lcd.clear()  # Clear the LCD display
        lcd.write_string("Aman")

    cv2.imshow('Image', img)
    if cv2.waitKey(1) == ord('q'):
        break

# Cleanup GPIO and LCD
GPIO.cleanup()
lcd.clear()
cam.release()
cv2.destroyAllWindows()
