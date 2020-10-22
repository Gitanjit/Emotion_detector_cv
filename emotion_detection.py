import streamlit as st
import cv2
import os
import time
import numpy as np
from PIL import Image
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, Activation
from keras.models import Sequential
from keras.models import model_from_json
from keras.utils.generic_utils import get_custom_objects
from keras import backend as K
from playsound import playsound
import threading

def swish_activation(x):
        return (K.sigmoid(x) * x)
get_custom_objects().update({'swish_activation': Activation(swish_activation)})

st.set_option('deprecation.showfileUploaderEncoding', False)
labels = ['Angry', 'Disgusted', 'Fearful', 'Happy', 'Sad', 'Surprised', 'Neutral']
face_detector = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')

json_file = open('models/jweights.json', 'r')
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)

model.load_weights('models/hweights.h5')

def play_sound_thread(file_name):
    thread = threading.Thread(target=playsound, args=(file_name,))
    thread.start()

def image_emotion_detection(img):
    np_img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    faces = face_detector.detectMultiScale(np_img_gray, 1.3, 10)

    for (x, y, w, h) in faces:
        roi = np_img_gray[y:y + h, x:x + w]
        roi = np.expand_dims(np.expand_dims(cv2.resize(roi, (48, 48)), -1), 0)
        roi = roi/255
        required_input_shape = roi.reshape((1, 1, 48, 48))
        yhat= model.predict(required_input_shape)
        cv2.rectangle(img, (x, y-25), (x+118, y), (255, 0, 0), -1)
        detected_label = labels[int(np.argmax(yhat))]
        cv2.putText(img, detected_label, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 3)

        filepath = os.path.join('audio_files', detected_label + '.mp3')
        play_sound_thread(filepath)

    if len(faces) == 0:
        filepath = os.path.join('audio_files', 'nofaces.mp3')
        play_sound_thread(filepath)

    st.image(img, caption="Processed Image", width=500)

def webcam_img():
    captured_img = None
    cap = cv2.VideoCapture(0)

    while cap.isOpened() and captured_img is None:
        ret, frame = cap.read()

        if ret == False:
            break
        else:
            img = frame.copy()
            cv2.rectangle(frame, (0, 0), (222, 40), (0, 0, 0), -1)
            cv2.putText(frame, 'Press ESC to exit', (2, 16), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0,  255), 1)
            cv2.putText(frame, 'Press Q to Capture Image', (2, 36), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0,  255), 1)

            cv2.imshow('WebCam', frame)

            if cv2.waitKey(4) & 0xFF == ord('q'):
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                st.image(img, caption='Captured Image', width=800)
                captured_img = img
                break

            elif cv2.waitKey(4) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()
    if captured_img is not None:
        st.success("Processing the Image....")
        time.sleep(5)
        np_img_gray = cv2.cvtColor(captured_img, cv2.COLOR_RGB2GRAY)

        faces = face_detector.detectMultiScale(np_img_gray, 1.3, 10)

        for (x, y, w, h) in faces:
            roi = np_img_gray[y:y + h, x:x + w]
            roi = np.expand_dims(np.expand_dims(cv2.resize(roi, (48, 48)), -1), 0)
            roi = roi/255
            required_input_shape = roi.reshape((1, 1, 48, 48))
            yhat= model.predict(required_input_shape)
            cv2.rectangle(img, (x, y-25), (x+118, y), (255, 0, 0), -1)
            detected_label = labels[int(np.argmax(yhat))]
            cv2.putText(captured_img, detected_label, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.rectangle(captured_img, (x, y), (x + w, y + h), (255, 0, 0), 3)

            filepath = os.path.join('audio_files', detected_label + '.mp3')
            play_sound_thread(filepath)

        if len(faces) == 0:
            filepath = os.path.join('audio_files', 'nofaces.mp3')
            play_sound_thread(filepath)

        st.image(captured_img, caption="Processed Image", width=800)

def webcam_video():
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()

        if ret == False:
            break
        else:
            gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(gray,1.3,5)

            for (x,y,w,h) in faces:
                cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
                roi_gray = gray[y:y+h,x:x+w]
                roi_gray = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
                roi_gray = roi_gray/255
                required_input_shape = roi_gray.reshape((1, 1, 48, 48))
                yhat= model.predict(required_input_shape)
                cv2.rectangle(frame, (x, y-25), (x+118, y), (255, 0, 0), -1)
                detected_label = labels[int(np.argmax(yhat))]
                cv2.putText(frame, detected_label, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1, cv2.LINE_AA)

            cv2.rectangle(frame, (0, 0), (155, 20), (0, 0, 0), -1)
            cv2.putText(frame, 'Press ESC to exit', (2, 16), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0,  255), 1)
            cv2.imshow('Emotion Detector',frame)

            if cv2.waitKey(4) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()


st.title("Emotion Detection App")
st.text("An app that detects Facial Emotions")

sidebar = st.sidebar.selectbox("", ['Documentation', 'Detection'])
if sidebar == 'Documentation':
    st.image('Images/first.png', width=800)
    st.markdown(
    '''
    **Emotion Detection App** helps to detect emotions of the faces. This app uses
    Covolution Neutral Network for prediction of emotion and HaaR CascadeClassifier
    for detecting the faces.

    We have worked in this project to come up with an optimum classifier so that we can get a good result.
    The Model does a pretty good work to in predicting and provides us an Test accuracy of 86%.

    *Below we have described the model.*

    The summary of the model which we have used is :
    '''
    )
    st.image('Images/model-summary.png')
    st.markdown(
    '''
    From the summary we can see that Convolutional Layer, MaxPooling Layer, swish activation function
    relu activation etc. are used.

    Using this model we got a accuracy graph:
    '''
    )
    st.image('Images/accuracy_graph.png', width=500)
    st.write("And the Loss graph is ")
    st.image('Images/download.png', width=500)
    if st.button("Frequently Asked Questions on Model"):
        st.markdown(
        '''
        ### 1.   Which Neural Network and why?
**Convolutional Neural Network** is used in our Project.
CNN is very much used in analysing visual imagery. It also helps use to get more accurate value and it is also very much proved that CNNs is very useful in image recognition and classification each image processes result in vote here after featuring the pixel with every weight connected to every convolutional layer  to get an accurate value with the model.

### 2.    Which optimizer and why?

**Adam –** Adaptive Moment Estimation.it requires very little memory and it is the one of the most used in gradient descent optimization algorithms.it also update parameters with individual learning rate . and also it is best optimizer and fast then SGD.

**SGD –** stochastic gradient descent. we have also used SGD optimizer in your model it is used for optimizing an objective function. In this few samples are selected and randomly instead of the whole dataset set for each iteration and with help of this it update the weight and parameter values.

### **3.    Which accuracy metric and why?**

Accuracy metric is used in your model . It is used for evaluating classification models it creates two local variables total and count that are used to compute the frequency. The best score for classification problem is 100% and we have achieved 86% with your model .
And also we have confusion matric it is one of the easiest way to measure the performance of a classification problem in this the matrix compare the actual target values with those the predicted by the machine learning model and recall in confusion matrix is the ratio of the relevant result returned by the search engine to the total number of relevant result that could be returned.

### **4.    Which loss function and why.**

We used categorical crossentropy. it is used to train the CNN  to output probability over the classes  for each image .the calculate the error between 0 and 1.It is also used for multi-class classification. It is a function that maps values of one or more variables onto a real from the loss function we learn to reduce the error in prediction it is used in classification problem and works on the whole dataset with the prediction and absolute values of dataset.

### **5.    Brief information on how cleaning was done.**
Having clean data will ultimately increase overall   productivity and allow for the highest quality information and with clean dataset we can get a highest quality information. And your dataset have the quality and mostly no required for cleaning in this because everything is as important as it is to predict the dataset.

### **6.    How data was got into the right shape.**

In the training part all images are 1x 48x48 and  real images are different in size. You must also check the size of the pictures in the dataset and one of them with different size and also you should resize all the images if they are not in the same size to get shape of images  In  this dataset data cleaning was not done because every data is useful and have a certain level of quality and importance so data drop or any other cleaning task are not performed . The data was already a right one for the model so we proceed for the further task on training the model and more.

### **7.    Which functions / features of OpenCV are used?**

1- The functions of OpenCV used in Image Detection Part are-
-	cv2.imread
-	cv2.cvtColor
-	cv2.CascadeClassifier → detectMultiscale
-	cv2.normalize
-	cv2.rectangle
-	cv2.putText
-	cv2.imshow
-	cv2.waitKey

2- The functions of OpenCV used in Video Detection Part are-
-	cv2.CascadeClassifier → detectMultiscale
-	cv2.VideoCapture
-	cv2.cvtColor
-	cv2.rectangle
-	cv2.resize
-	cv2.putText
-	cv2.imshow
-	cv2.waitKey
- cv2.VideoCapture.release
-	cv2.destroyAllWindows


### **8.    Which dataset have you used?**
***fer2013.csv[287.13mb]***

With 0 to 6 emotions and 34034 unique values- emotion detection dataset is used here (https://drive.google.com/uc?id=1X60B-uR3NtqPd4oosdotpbDgy8KOfUdr&export=download)

This drive link consists of 28710 train set .png files and 3590 test set .png files of seven different emotions.

This fer2013.csv file contains the train and test images of the google drive folder above in the form of a single .csv file as pixels matrices.

**Emotions are:-**
 0 - denotes angry,
 1 - denotes disgusted,
 2 - denotes fearful,
 3 - happy,
 4 - sad,
 5 - surprised,
 6 - Neutral
        '''
        )
    st.markdown(
    '''
    For the working of this code we have used few **Python Modules**:

    - keras
    - numpy
    - pandas
    - pillow
    - opencv
    - streamlit
    - os
    - threading
    - playsound
    - gTTS
    - time

    To run this script we need to have all these modules to be installed in your system.

    '''
    )
elif sidebar == 'Detection':
    selection_tuple = ('Uploading Image', 'Uploading Video', 'Captured Image by Webcam', 'Realtime Video by Webcam')
    selection = st.sidebar.radio('Detection by', selection_tuple)
    if selection == selection_tuple[0]:
        st.subheader('Emotion Detection via Image')
        file = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])
        st.markdown(
        '''
         Upload an Image that contains an image of a Face and when
         **Process** button is pressed then it will **detect the face** and
         tell **its emotion**. It will detect the emotion and
         **print it on the image** and also **play a audio**
         that tells about the emotion present on the image.
        '''
        )
        if file is not None:
            st.image(file, caption="Original Image", width=500)
            if st.button("Process"):
                img = Image.open(file)
                np_img = np.array(img.convert('RGB'))
                image_emotion_detection(np_img)
    elif selection == selection_tuple[1]:
        st.subheader('Emotion Detection via Video')
        st.markdown(
        '''
        Provide the **file path** of the video file on which you want to run our Emotion
        Detection App. Our app will tell u the emotion of the faces present in the video.
        It will make a box around the face and show the emotion of the face above it.

        If you want to exit the video press `ESC` key.
        '''
        )
        filepath = st.text_input("Video File Path: ")
        if filepath.endswith('.mp4'):
            if st.button("Start Detecting"):
                os.system('python video_detection.py ' + str(filepath))
    elif selection == selection_tuple[2]:
        st.subheader('Emotion Detection with a Realtime Image captured using Webcam')
        st.markdown(
        '''
        Press the button **`Start WebCam`** to start the webcam.

        Press **`Q`** to capture the image and detect emotion.

        Press **`ESC`** to turn off the webcam.

        As soon as you capture a image it will detect the face and emotion and
        also play the emotion related audio. It will make a box around the face and
        print the emotion on the image.
        '''
        )
        if st.button('Start WebCam'):
            webcam_img()

    elif selection == selection_tuple[3]:
        st.subheader('Emotion Detection on Realtime Video captured using Webcam')
        st.markdown(
        '''
        Press the button **`Start WebCam`** to start the webcam.

        Press **`ESC`** to turn off the webcam.

        It will detect the face on the video and also detect the emotion on the face
        and will show the emotion on the video and play a audio related to the emotion.
        '''
        )
        if st.button('Start WebCam'):
            webcam_video()
    else:
        st.text('Unknown Selection')
