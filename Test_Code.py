
from keras.models import load_model
from time import sleep
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
from math import floor
import cv2
import numpy as np
import simpleaudio

def format_duration(seconds):
    if seconds > 3600:
        return f"{int(seconds // 3600)} Hours {int((seconds % 3600) // 60)} Minutes {int(seconds % 60)} Seconds"
    elif seconds > 60:
        return f"{int(seconds // 60)} Minutes {int(seconds % 60)} Seconds"
    else:
        return f"{int(seconds % 60)} Seconds"

face_classifier=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
classifier = load_model('EmotionDetectionModel_v15.h5')

cap=cv2.VideoCapture(0)
FPS = cap.get(cv2.CAP_PROP_FPS)
if cap == None or FPS == 0:
    print("Unable to connect to webcam.")
    exit(1)

song = simpleaudio.WaveObject.from_wave_file('bensound-love.wav')

playback = None

frame_count = 0
anger_inc = 0
anger_frame = 0
total_angry_frames = 0
not_anger_frame =0
while True:
    ret,frame=cap.read()
    frame_count += 1
    labels=[]
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=face_classifier.detectMultiScale(gray,1.3,5)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray=gray[y:y+h,x:x+w]
        roi_gray=cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)

        if np.sum([roi_gray])!=0:
            roi=roi_gray.astype('float')/255.0
            roi=img_to_array(roi)
            roi=np.expand_dims(roi,axis=0)
            
            preds=classifier.predict(roi)[0]
            if preds.argmax() == 0:
                label="Angry"
                label_position=(x,y)
                anger_frame+=1
                total_angry_frames+=1
                if anger_frame == FPS*3:
                    not_anger_frame=0
                    #play music
                    anger_inc+=1
                    playback = song.play()
                #label += " for " + str(anger_frame / FPS) + " seconds"
                cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
            else:
                not_anger_frame+=1
                if not_anger_frame == FPS*3:
                    anger_frame = 0
                    if (playback != None and playback.is_playing()):
                        playback.stop()
    
    cv2.imshow('Emotion Detector',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
if (playback != None and playback.is_playing()):
    playback.stop()
blank = np.zeros((480,640,3), np.uint8)
blank[:] = (255, 255, 255)
stats = f"""Statistics"""
cv2.putText(blank,"Statistics:", (230, 50),cv2.FONT_HERSHEY_TRIPLEX,1,(0,0,0),2)
cv2.putText(blank,f"Anger Count: {anger_inc} Times", (20, 200),cv2.FONT_HERSHEY_TRIPLEX,1,(0,0,0),2)
cv2.putText(blank,f"Total Angry Time: ", (20, 240),cv2.FONT_HERSHEY_TRIPLEX,1,(0,0,0),2)
cv2.putText(blank,format_duration(total_angry_frames / FPS), (40, 280),cv2.FONT_HERSHEY_TRIPLEX,1,(0,0,0),2)
cv2.putText(blank,f"Total Driving Time:", (20, 320),cv2.FONT_HERSHEY_TRIPLEX,1,(0,0,0),2)
cv2.putText(blank,format_duration(frame_count / FPS), (40, 360),cv2.FONT_HERSHEY_TRIPLEX,1,(0,0,0),2)
cv2.imshow('Emotion Detector',blank)
while not (cv2.waitKey(1) & 0xFF == ord('q')):
    pass
cv2.destroyAllWindows()
