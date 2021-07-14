import cv2
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, Activation
from keras.models import Sequential
from keras.models import model_from_json
from keras.utils.generic_utils import get_custom_objects
from keras import backend as K


import tensorflow as tf

def swish_activation(x):
        return (K.sigmoid(x) * x)
get_custom_objects().update({'swish_activation': Activation(swish_activation)})

labels = ['Angry', 'Disgusted', 'Fearful', 'Happy', 'Sad', 'Surprised', 'Neutral']
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

index_to_emotion = {0: 'angry',
                    1: 'fearful',
                    2: 'neutral',
                    3: 'happy',
                    4: 'disgusted',
                    5: 'sad',
                    6: 'surprised'}


json_file = open('jweights.json', 'r')
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)

model.load_weights('hweights.h5')
model1 = tf.keras.models.load_model('my_model.h5')

filepath = sys.argv[1]
print(filepath)
print("at least this is opening")
cap = cv2.VideoCapture(filepath)

while cap.isOpened():
        ret, frame = cap.read()

        if ret == False:
            break
        else:
            gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(gray,1.3,5)

            for (x,y,w,h) in faces:
                # cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
                # roi_gray = gray[y:y+h,x:x+w]
                # roi_gray = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
                # roi_gray = roi_gray/255
                # required_input_shape = roi_gray.reshape((1, 1, 48, 48))
                # yhat= model.predict(required_input_shape)
                # cv2.rectangle(frame, (x, y-25), (x+118, y), (255, 0, 0), -1)
                # detected_label = labels[int(np.argmax(yhat))]
                # cv2.putText(frame, detected_label, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.rectangle(frame, (x-5, y-10), (x+w+5, y + h+10), (255, 0, 0), 2)
                roi_gray = gray[y+25:y-10 + h, x+20:x + w-20]
                roi_color = frame[y+25:y-10 + h, x+20:x + w-20]
                img = np.zeros_like(roi_color)
                img[:, :, 0] = roi_gray
                img[:, :, 1] = roi_gray
                img[:, :, 2] = roi_gray
                # resizing the image
                try:
                    image = cv2.resize(img, (48,48), interpolation=cv2.INTER_AREA)
                except Exception as e:
                    print(str(e))
                img_fed = np.expand_dims(image, axis=0)
                scores = model1.predict(img_fed)
                index = np.argmax(scores)
                cv2.putText(frame, index_to_emotion[index], (x,y-15), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), thickness=2)

            cv2.rectangle(frame, (0, 0), (155, 20), (0, 0, 0), -1)
            cv2.putText(frame, 'Press ESC to exit', (2, 16), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0,  255), 1)
            cv2.imshow('Emotion Detector',frame)

            if cv2.waitKey(4) & 0xFF == 27:
                break

cap.release()
cv2.destroyAllWindows()