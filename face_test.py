import face_recognition
from sklearn import neighbors
import os
import cv2
import pickle
import numpy as np
from PIL import Image
#from yolo.yolo import YOLO
import time 
from keras.models import load_model
from keras.models import model_from_json
import numpy as np
import sklearn
import pandas as pd
import warnings
from detect import Yolo

from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
warnings.filterwarnings("ignore")

COLS = ['Male', 'Asian', 'White', 'Black','Indian','0-5','10-15','20-35','45-65','65+','No Eyewear']
confidence = 0.64

def draw_rectangle(frame, rectangle, color, thickness, label=None):
    if rectangle is not None:

        bot = (rectangle[1],rectangle[2])
        top = (rectangle[3],rectangle[0])
        
        cv2.rectangle(frame, top, bot, color, thickness)

        if label is not None:
            cv2.putText(frame, label, top, cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0), 2)

    return frame

def apply_offsets(rectangle, offsets):
    x, y, width, height = rectangle[3], rectangle[0], rectangle[1], rectangle[2]
    x_off, y_off = offsets
    return (x - x_off, x + width + x_off, y - y_off, y + height + y_off)

def preprocess_input(x, v2=True):
    x = x.astype('float32')
    x = x / 255.0
    if v2:
        x = x - 0.5
        x = x * 2.0
    return x


with open('facesDatabase.pickle', 'rb') as f:
     faceDatabase = pickle.load(f)

encodings = faceDatabase[0]
names = faceDatabase[1]
knn_clf = faceDatabase[2]

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
sess = tf.Session(config=config)
set_session(sess)
emotion_model_path = 'fer2013_mini_XCEPTION.46-0.82.hdf5'
emotion_labels = {0: 'angry', 1: 'happy', 2: 'surprise', 3: 'neutral'}
emo_labels = ['angry', 'happy', 'surprise', 'neutral']
emotion_classifier = load_model(emotion_model_path, compile=False)
emotion_target_size = emotion_classifier.input_shape[1:3]
emotion_offsets = (20, 40)


with open('face_model.pkl', 'rb') as f:
    clf, labels = pickle.load(f, encoding='latin1')
del faceDatabase
# Init a class for face detection
cap = cv2.VideoCapture('/home/spectra/Downloads/test_walk.MOV')
yolo = Yolo(0.5,0.5, False)


counter = 0
while(True):

    ret, frame = cap.read()
    frameShow = frame
    start = int(round(time.time() * 1000))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame =cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame)

    face_locations = yolo.detect(frame)
    # Find all the faces in the test image using SSD networks(Tensorflow)
    #face_locations = faceDetector.run(frame)

    # Predict all the faces in the test image using the trained classifier
    
    for i in range(len(face_locations)):

        x1, x2, y1, y2 = apply_offsets(face_locations[i], emotion_offsets)
        gray_face = gray[y1:y2, x1:x2]
        try:
            gray_face = cv2.resize(gray_face, (emotion_target_size))
            gray_face = preprocess_input(gray_face, False)
            gray_face = np.expand_dims(gray_face, 0)
            gray_face = np.expand_dims(gray_face, -1)
            emotion_label_arg = emotion_classifier.predict(gray_face)
            print(emotion_label_arg)
        except:
            pass
        for index, emotion in enumerate(emo_labels):
            print(emotion_label_arg[0])
            if index == 0:
                cv2.putText(frameShow, emotion, (10, index * 70+40), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)
                cv2.rectangle(frameShow, (250, index * 60 + 10), (250 + int(emotion_label_arg[0][index] * 500), (index + 1) * 60 + 4),(255, 0, 0), -1)
            if index == 1:
                cv2.putText(frameShow, emotion, (10, index * 70+30), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)
                cv2.rectangle(frameShow, (250, index * 60 + 10), (250 + int(emotion_label_arg[0][index] * 500), (index + 1) * 60 + 4),(255, 0, 0), -1)
            if index == 2:
                cv2.putText(frameShow, emotion, (10, index * 70+20), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)
                cv2.rectangle(frameShow, (250, index * 60 + 10), (250 + int(emotion_label_arg[0][index] * 500), (index + 1) * 60 + 4),(255, 0, 0), -1)
            if index == 3:
                cv2.putText(frameShow, emotion, (10, index * 70+20), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)
                cv2.rectangle(frameShow, (250, index * 60 + 10), (250 + int(emotion_label_arg[0][index] * 500), (index + 1) * 60 + 4),(255, 0, 0), -1)

        
        draw_rectangle(frameShow,face_locations[i],(0,0,255),2,'')

    cv2.imshow('Face',frameShow)
    cv2.waitKey(1)
