# Facial Emotion Detection.
 
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

<p>We used  categorical cross entropy . it is used to train the CNN  to output probability over the classes  for each image .the calculate the error between 0 and 1.It is also used for multi-class classification. It is a function that maps values of one or more variables onto a real from the loss function we learn to reduce the error in prediction it is used in classification problem and works on the whole dataset with the prediction and absolute values of dataset.

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
 6 - Neutral,


# **Prerequisites For The Project :**
To understand this thoroughly you should have some basic knowledge of :
- Python
- OpenCV
- Convolution Neural Network (CNN) 
- Numpy
- Tensorflow

Liberaries of python used in this project:
- Numpy
- Matplotlib
- Pandas
- OS
- Sys
- Gtts
- PlaySound
- Streamlit
- Threading
- OpenCV
- Tensorflow / Keras

          
First - Import all libraries and modules that are needed in this project and describe all              
        the values and variables which you have in dataset, i.e. -
- number of classes
- size of the image
- batch size and more.

Second - Take the dataset which we have selected is fer2013 in kaggle with 7 classes namely -  Angry, Disgust, Fear, Happy, Sad, Surprise and Neutral.
     The training set consists of a total of 28,709 examples. Now import the validation and training data . This model is trained on the training dataset and validation dataset.
          
Third - Now that we have completed the dataset modification, it's time to implement CNN network on a sequential model. We have used some of layers from keras -
- Conv2D()
- Activation(activation_type)
- BatchNormalization()
- MaxPooling2D(pool_size, strides, padding, data_format, ****kwargs**)
- Dropout(dropout_value)
- Flatten()
- Dense() 
- Activation layer
 
Fourth - Compile and train, now only left to compile and train the model .
