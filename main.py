import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import random
import sys
import os
import shutil
import cv2
import seaborn as sns
from matplotlib.pyplot import imshow
import tensorflow
from tensorflow import keras
from tensorflow.keras.models import load_model
from model import Performance
from log_code import setup_logging
logger=setup_logging('main')

class FOOD:
    def __init__(self,path):
        logger.info(f'Constructor started')
        try:
            self.path=path
            column_names=[]
            for i in os.listdir(self.path):
                column_names.append(i)
            logger.info(f'len:{len(column_names)}')
            logger.info(f'columns:{column_names}')
            self.training_dir='./training_dataset'
            self.validation_dir='./validation_dataset'
            self.testing_dir='./testing_dataset'
            # self.training_path='C:\\Users\\sravs\\Downloads\\project2\\training_dataset'
            # self.testing_path='C:\\Users\\sravs\\Downloads\\project2\\testing_dataset'
            # self.validation_path='C:\\Users\\sravs\\Downloads\\project2\\validation_dataset'
            # logger.info(f'training:{os.listdir(self.training_path)}')
            # logger.info(f'validate:{os.listdir(self.testing_path)}')
        except Exception as e:
            e_type, e_msg, e_linno = sys.exc_info()
            logger.info(f'Issue is:{e_linno.tb_lineno} due to {e_msg}')

    def train_test_split_images(self):
        try:
            images_per_class = 400
            split_ratio = 0.8
            self.source_path=self.path
            if os.path.exists(self.source_path):
                os.makedirs(self.training_dir, exist_ok=True)
                os.makedirs(self.testing_dir, exist_ok=True)
                os.makedirs(self.validation_dir, exist_ok=True)
                for i in os.listdir(self.source_path):
                    source = os.path.join(self.source_path, i)
                    if not os.path.isdir(source):
                        continue
                    os.makedirs(os.path.join(self.training_dir, i), exist_ok=True)
                    os.makedirs(os.path.join(self.testing_dir, i), exist_ok=True)
                    os.makedirs(os.path.join(self.validation_dir, i), exist_ok=True)
                    images = [f for f in os.listdir(source) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                    if not images:
                        print(f" No images found in class '{i}'")
                        continue
                    final_images = random.sample(images, min(len(images), images_per_class))
                    random.shuffle(final_images)
                    train_index = int(len(final_images) * split_ratio)
                    train_images = final_images[:train_index]
                    test_images = final_images[train_index:]
                    validation_images = train_images[:20]
                    train_images = train_images[20:]
                    for img in train_images:
                        source_path_train = os.path.join(source, img)
                        target_path_train = os.path.join(self.training_dir, i, img)
                        shutil.copy2(source_path_train, target_path_train)
                    for img1 in test_images:
                        source_path_test = os.path.join(source, img1)
                        target_path_test = os.path.join(self.testing_dir, i, img1)
                        shutil.copy2(source_path_test, target_path_test)
                    for img2 in validation_images:
                        source_path_valid = os.path.join(source, img2)
                        target_path_valid = os.path.join(self.validation_dir, i, img2)
                        shutil.copy2(source_path_valid, target_path_valid)
        except Exception as e:
            e_type, e_msg, e_linno = sys.exc_info()
            logger.info(f'Issue is:{e_linno.tb_lineno} due to {e_msg}')
    def custom_model(self):
        try:
            model=load_model('./custom_model.h5', compile=False)
            model1=load_model('./vgg16_model.h5', compile=False)
            test_path='./testing_dataset'
            Performance.custom(model,test_path)
            Performance.vgg16(model1,test_path)
        except Exception as e:
            e_type, e_msg, e_linno = sys.exc_info()
            logger.info(f'Issue is:{e_linno.tb_lineno} due to {e_msg}')

    def prediction(self,img_path):
        try:
            model=load_model('./resnet_model_1.h5',compile=False)
            image=cv2.imread(img_path,1)
            labels = ['chole_bhature', 'chicken_curry', 'cheesecake']
            logger.info(f'Original Image shape  : {image.shape}')
            resize_imag=cv2.resize(image,(256,256))
            logger.info(f'Resized Image shape : {resize_imag.shape}')
            input_img=np.expand_dims(resize_imag,axis=0)
            logger.info(f'Perfect Model Input : {input_img.shape}')
            res=model.predict(input_img)
            r = labels[np.argmax(res)]
            print(f'Predicted Class was : {r}')
            plt.imshow(image[:, :, ::-1])
            plt.title(f'Predicted: {r}')
            plt.axis('off')
            plt.show()
        except Exception as e:
            e_type, e_msg, e_linno = sys.exc_info()
            logger.info(f'Issue is:{e_linno.tb_lineno} due to {e_msg}')


if __name__=="__main__":
    try:
        path='C:\\Users\\sravs\\Downloads\\project2\\archive\\Food Classification dataset'
        obj=FOOD(path)
        #obj.train_test_split_images()
        #obj.custom_model()
        image=r'C:\Users\sravs\Downloads\project2\_Delicious Chole Bhature_ A Taste of India_.jpg'
        obj.prediction(image)
    except Exception as e:
        e_type, e_msg, e_linno = sys.exc_info()
        logger.info(f'Issue is:{e_linno.tb_lineno} due to {e_msg}')
