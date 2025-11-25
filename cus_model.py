import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import sys
import os
import seaborn as sns
import cv2
from log_code import setup_logging
logger=setup_logging('custom')
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import warnings
warnings.filterwarnings('ignore')
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Activation,Conv2D,MaxPooling2D,Flatten
from tensorflow.keras.activations import relu,sigmoid,softmax

def modell(train,valid):
    try:
        logger.info(f'training:{os.listdir(train+'/burger')}')
        # c_img=cv2.imread('C:\\Users\\sravs\\Downloads\\project2\\training_dataset\\cheesecake\\53831.jpg')
        # logger.info(f'shape:{c_img.shape}')
        # cv2.imshow('image_c',c_img)
        # cv2.waitKey()
        # cv2.destroyAllWindows()
        training_set=ImageDataGenerator(rescale=1/255,rotation_range=0.2,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)
        validation_set=ImageDataGenerator(rescale=1/255,rotation_range=0.2,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)
        labels=os.listdir(train)
        train_input_data=training_set.flow_from_directory(train,classes=labels,target_size=(256,256),class_mode='categorical',batch_size=100)
        valid_input_data=validation_set.flow_from_directory(valid,classes=labels,target_size=(256,256),class_mode='categorical',batch_size=100)
        logger.info(f'_____________________')
        model=Sequential()
        model.add(Conv2D(128,kernel_size=(3,3),kernel_initializer='he_uniform',padding='valid',strides=1,input_shape=(256,256,3)))
        model.add(MaxPooling2D(pool_size=()))
    except Exception as e:
        e_type, e_msg, e_linno = sys.exc_info()
        logger.info(f'Issue is:{e_linno.tb_lineno} due to {e_msg}')