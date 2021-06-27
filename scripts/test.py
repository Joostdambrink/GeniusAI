import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
from tensorflow.keras.layers import Lambda
from PIL import Image
from PIL import ImageEnhance
from Utils import Utils
from Models import SuperResModels
import os

class TestModels:
    def __init__(self):
        self.Utils = Utils()
    """Loads keras model from path
    
    Args:
        model_path (str) : path of model to load
        compile (bool, optional) : compile model automatically True or False. defaults to True
    
    Returns:
        (keras model) : the loaded model.
    """
    def loadModel(self,model_path, is_custom = True):
        if is_custom:
            return tf.keras.models.load_model(model_path, custom_objects = {'L1Loss' : self.L1Loss})
        else:
            return tf.keras.models.load_model(model_path, custom_objects={'normalize': Lambda(self.Utils.normalize), 'denormalize' : Lambda(self.Utils.denormalize)})

    def L1Loss(self, y_true,y_pred):
        return abs(y_true - y_pred)
    
    def resolve_single(self,model, lr):
        return self.resolve(model, tf.expand_dims(lr, axis=0))[0]

    def resolve(self,model, lr_batch):
        lr_batch = tf.cast(lr_batch, tf.float32)
        sr_batch = model(lr_batch)
        sr_batch *= 255
        sr_batch = tf.clip_by_value(sr_batch, 0, 255)
        sr_batch = tf.round(sr_batch)
        sr_batch = tf.cast(sr_batch, tf.uint8)
        return sr_batch

    def plot_sample(self,lr, sr, sr_2, hr):
        plt.figure(figsize=(20, 10))

        images = [lr, sr, sr_2, hr]
        titles = ['LR', f'SR (x{sr.shape[0] // lr.shape[0]})', f'SR_enhanced (x{sr.shape[0] // lr.shape[0]})', f'HR']

        for i, (img, title) in enumerate(zip(images, titles)):
            plt.subplot(1, len(images), i+1)
            plt.title(title)
            plt.xticks([])
            plt.yticks([])
            plt.imshow(img)
        plt.show()
        
    def PlotAndSave(self,model,lr_path, hr_path, save_path, save_output = True):
        hr = cv2.cvtColor(np.array(Image.open(hr_path)), cv2.COLOR_BGRA2BGR)
        lr = cv2.cvtColor(np.array(Image.open(lr_path)), cv2.COLOR_BGRA2BGR)
        sr = self.resolve_single(model, lr / 255)
        img = Image.fromarray(np.array(sr))
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.25)
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(1.3)
        if save_output:
            Image.fromarray(np.array(sr)).save(save_path)
        self.plot_sample(lr,sr, img, hr)