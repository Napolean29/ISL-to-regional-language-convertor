# 1. New detection variables
# TEST_VIDEO_PATH = '/content/drive/MyDrive/realtime_test_oct.avi'# insert video path here
import cv2
import os, sys, gc
import time
import numpy as np
import mediapipe as mp
from tqdm.auto import tqdm
import multiprocessing
from joblib import Parallel, delayed
from natsort import natsorted
from glob import glob
import math 
import pickle
import pandas as pd
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout,Flatten
from tensorflow.keras.callbacks import TensorBoard
from keras.layers.advanced_activations import LeakyReLU
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import TimeDistributed
from keras.preprocessing.sequence import pad_sequences
from google.colab.patches import cv2_imshow
from scipy import stats
import matplotlib.pyplot as plt

def keypoints_for_frame(frame):
  results = holistic.process(frame)

  body_data, body_conf = process_body_landmarks(
      results.pose_landmarks, N_BODY_LANDMARKS
  )
  face_data, face_conf = process_other_landmarks(
      results.face_landmarks, N_FACE_LANDMARKS
  )
  lh_data, lh_conf = process_other_landmarks(
      results.left_hand_landmarks, N_HAND_LANDMARKS
  )
  rh_data, rh_conf = process_other_landmarks(
      results.right_hand_landmarks, N_HAND_LANDMARKS
  )

  data = np.concatenate([body_data, lh_data, rh_data])
  conf = np.concatenate([body_conf, lh_conf, rh_conf])

  pose_kps, pose_confs = data, conf

  return data, conf

mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities

N_FACE_LANDMARKS = 468
N_BODY_LANDMARKS = 33
N_HAND_LANDMARKS = 21

MODEL_DIR = '/content/drive/MyDrive/SLRLC.h5'
model = keras.models.load_model(MODEL_DIR)
max_len = # max length of video in model training (padded length)
sequence = []
sentence = []
predictions = []
threshold = 0.5
startTime = time.time()
# used to record the time when we processed last frame
prev_frame_time = 0

# used to record the time at which we processed current frame
new_frame_time = 0

cap = cv2.VideoCapture(0)
# Set mediapipe model 
with mp_holistic.Holistic(static_image_mode=False, model_complexity=2) as holistic:
    while cap.isOpened():

        # Read feed
        ret, frame = cap.read()
        # time.sleep(1)

        # font which we will be using to display FPS
        if not ret:
          break
        frame = cv2.resize(frame, (1000, 600))
        # Make detections
        image, results = mediapipe_detection(frame, holistic)
        # print(results)
        
        # Draw landmarks
        draw_styled_landmarks(image, results)
        
        # 2. Prediction logic
        keypoints, _ = keypoints_for_frame(frame)
        sequence.append(keypoints)
        sequence = sequence[-(max_len):]
        res = 0
        if len(sequence) == max_len:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            print(max(res))
            print(labels[np.argmax(res)])
            predictions.append(np.argmax(res))
            
            
        #3. Viz logic
            if np.unique(predictions[-10:])[0]==np.argmax(res): 
                if res[np.argmax(res)] > threshold: 
                    
                    if len(sentence) > 0: 
                        if labels[np.argmax(res)] != sentence[-1]:
                            sentence.append(labels[np.argmax(res)])
                    else:
                        sentence.append(labels[np.argmax(res)])

            if len(sentence) > 1: 
                sentence = sentence[-1:]

            # Viz probabilities
            # image = prob_viz(res, labels, image, colors)
            
        cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
        cv2.putText(image, ' '.join(sentence), (3,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 2, cv2.LINE_AA)
        # cv2.putText(image, str(max(res)), (4,45), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 0, 255), 2, cv2.LINE_AA)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        # time when we finish processing for this frame
        new_frame_time = time.time()

        # Calculating the fps

        # fps will be number of frame processed in given time frame
        # since their will be most of time error of 0.001 second
        # we will be subtracting it to get more accurate result
        fps = (1/(new_frame_time-prev_frame_time)) + 1
        prev_frame_time = new_frame_time

        # converting the fps into integer
        fps = int(fps)

        # converting the fps to string so that we can display it on frame
        # by using putText function
        fps = str(fps)

        # putting the FPS count on the frame
        # cv2.putText(gray, f'FPS = {fps}', (7, 70), font, 1, (100, 255, 0), 3, cv2.LINE_AA)
        cv2.putText(image, fps, (3, 30),  font, 1, (100, 255, 0), 1, cv2.LINE_AA)
        

        
        # Show to screen
        cv2_imshow(image) # while using colab instance
        # cv2.imshow('OpenCV Feed', image) # while using local instance

        # Break gracefully
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    endTime = time.time()
    elapsedTime = endTime - startTime
    print("Elapsed Time = %s" % elapsedTime)
    cap.release()
    cv2.destroyAllWindows()