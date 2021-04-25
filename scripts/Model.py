import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Activation, UpSampling2D,PReLU,Input,Conv2DTranspose,GaussianNoise,Conv2DTranspose,Concatenate,LeakyReLU
from tensorflow.keras import Sequential
from tensorflow.keras.initializers import Constant
from time import time
from tensorflow.keras.callbacks import TensorBoard
import pickle
import numpy as np
import cv2
from ResidualBlock import ResidualBlock
from CustomSchedular import CustomLearningRateScheduler
from Utils import Utils
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class SuperResModel:
    def __init__(self):
        self.Utils = Utils()

    """
        LoadModel:
            takes in the path of a saved model and a boolean value for compile (use compile = False if the model uses a custom loss function).
            loads and returns the model
    """
    def loadModel(self,model_path, compile = True):
        model = tf.keras.models.load_model('saved_model/my_model', compile = compile)
        if(not compile):
            model.compile(optimizer = tf.keras.optimizers.SGD(learning_rate = 0.01,momentum=0.9), loss = L1Loss, metrics = ['acc'])
        return model


    """
        PredictAndShowImage:
            takes in a ML model and the path of the input data
            loads the data and feeds the first 2 images to the model to predict.
            shows the prediction in a window using cv2
    """
    def PredictAndShowImage(self, model, data_path):
        test_lr = pickle.load(open(data_path,"rb"))[:2]
        test_lr = test_lr.reshape(-1,96,96,3)
        test_lr = test_lr/255

        pred = model.predict(test_lr)
        cv2.imshow("image",pred[0])
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    """
        Model:
            this is the convolutional model used for the super resolution task.
            the model contains the following layers:
                - Convolutional layer
                - followed by a PReLU activation layer
                - a residual block of 24 residual units
                - two convolutional layers
                - a PReLU activation layer
                - two convolutional layers
                -PReLU activation layer
                -upsampling layer
                -four convolutional layers
                - one PReLU activation layer
                - one last convolutional layer

            returns a Keras model with all these layers.
    """
    def Model(self):
        conv_args = {
            "activation": "relu",
            "kernel_initializer": "Orthogonal",
            "padding": "same",
        }
        rb = ResidualBlock()
        inputs = Input(shape=[None,None,3])
        c1 = Conv2D(64,5,input_shape = (None,None,3),**conv_args)(inputs)
        pr1 = PReLU(alpha_initializer="zeros",shared_axes=[1,2])(c1)

        resUnit = rb.ResBlock(pr1,num_of_units = 24,num_of_filters = 64)
        
        concat = Concatenate(axis=-1)([pr1,resUnit])
        
        c2 = Conv2D(64,4,**conv_args)(concat)

        c3 = Conv2D(64,4,**conv_args)(c2)
        pr2 = PReLU(alpha_initializer="zeros",shared_axes=[1,2])(c3)
        c4 = Conv2D(64,4,**conv_args)(pr2)

        c5 = Conv2D(64,3,**conv_args)(c4)
        pr3 = PReLU(alpha_initializer="zeros",shared_axes=[1,2])(c5)

        up1 = UpSampling2D(size = (2,2),interpolation = "nearest")(pr3)
        c6 = Conv2D(64,2,**conv_args)(up1)
        leaky = LeakyReLU()(c6)
        up2 = UpSampling2D(size = (2,2), interpolation="nearest")(leaky)
        c7 = Conv2D(64,2,**conv_args)(up2)
        leaky2 = LeakyReLU()(c7)

        c8 = Conv2D(64,2,**conv_args)(leaky2)
        c9 = Conv2D(64,2,**conv_args)(c8)
        pr4 = PReLU(alpha_initializer="zeros",shared_axes=[1,2])(c9)
        c10 = Conv2D(3,2,**conv_args)(pr4)

        return tf.keras.Model(inputs = inputs, outputs = c10)

    """
        schedular:
            custom learning rate schedular takes in current epoch index and the current learning rate.
            halves the learning rate eacht 20 epochs.
            returns the new learning rate.
    """
    def schedular(self, epochIndex, lr):
        if epochIndex % 20 == 0 and epochIndex > 0:
            return lr * 0.5
        return lr

    """
        L1Loss:
            L1 loss function takes in the true value of y and predicted value.
            substracts them from eachother.
            returns the absolute value of the substraction.
    """
    def L1Loss(self, y_true,y_pred):
        return abs(y_true - y_pred)

    """
        TrainModel:
            takes in training data paths (lowres and highres).
            loads the data and normalizes it
            build a model and fit the data to it.
            saves the model after training is finished.
    """
    def TrainModel(self,training_lr_path , training_hr_path, num_of_epochs = 100):
        checkpoint_filepath = 'saved_model/my_model'
        tensorboard = TensorBoard(log_dir = "logs/{}".format(time()))
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=False,
        monitor='loss',
        mode='min',
        save_best_only=False)
        training_lr = self.Utils.LoadH5File(training_lr_path)
        training_hr = self.Utils.LoadH5File(training_hr_path)
        training_lr = training_lr/255
        training_hr = training_hr/255

        model = self.Model()
        model.summary()
        model.compile(optimizer = tf.keras.optimizers.SGD(learning_rate = 0.01,momentum=0.9), loss = self.L1Loss)
        model.fit(training_lr, training_hr, batch_size = 1, epochs= num_of_epochs ,callbacks=[CustomLearningRateScheduler(self.schedular), early_stop, tensorboard, model_checkpoint_callback])

        model.save(checkpoint_filepath)

model = SuperResModel()
#model.PredictAndShowImage(loadModel('saved_model/my_model', compile = False), data_path=r"Data\train_lr.pickle")

model.TrainModel( r"D:\HBO\MinorAi\PickleFiles\train_lr.h5", r"D:\HBO\MinorAi\PickleFiles\train_hr.h5", num_of_epochs = 1)