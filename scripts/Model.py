import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Activation, UpSampling2D,PReLU,Input,Conv2DTranspose,GaussianNoise,Conv2DTranspose,Concatenate,LeakyReLU,Add
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


    """Loads keras model from path
    
    Args:
        model_path (str) : path of model to load
        compile (bool, optional) : compile model automatically True or False. defaults to True
    
    Returns:
        (keras model) : the loaded model.
    """
    def loadModel(self,model_path, compile = True):
        model = tf.keras.models.load_model(model_path, compile = compile)
        if(not compile):
            model.compile(optimizer = tf.keras.optimizers.SGD(learning_rate = 0.1,momentum=0.9), loss = self.L1Loss)
        return model


    """Predicts and shows input and prediction images

    Args:
        model (keras model) : keras model that does the prediction
        data_path (str) : path of the input data
        read_from_directory (bool, optional) : whether to read images from a directory or from a file. defaults to False
    """
    def PredictAndShowImage(self, model, data_path, read_from_directory = False):
        data = []
        if read_from_directory:
            data = self.Utils.ReadImages(data_path)
        else:
            data = self.Utils.LoadH5File(data_path)
        data = data/255
        image = np.expand_dims(data[1], axis = 0)
        pred = model.predict(image)
        cv2.imshow("prediction",pred[0])
        cv2.imshow("input",image[0])
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
        
        concat = Add()([pr1,resUnit])
        
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


    """drops learning rate by 50% each 20 epochs

    Args:
        epoch_index (int) : index of current epoch
        lr (float) : current learning rate

    Returns:
        (float) : modified learning rate
    """
    def schedular(self, epoch_index, lr):

        if epoch_index % 20 == 0 and epoch_index > 0:
            return lr * 0.5
        return lr


    """L1 loss function

    Args:
        y_true (numpy array / tf tensor) : the observed value of y
        y_pred (numpy array / tf tensor) : the predicted value of y

    Returns:
        (float) : the absolute value of the difference between y_true and y_pred
    """
    def L1Loss(self, y_true,y_pred):

        return abs(y_true - y_pred)


    """Trains keras model on input data

    Args:
        training_lr_path (str) : path of low res training data
        training_hr_path (str) : path of high res training data
        num_of_epochs (Int, optional) : number of training epochs
    """
    def TrainModel(self,training_lr_path , training_hr_path, num_of_epochs = 100):

        checkpoint_filepath = 'super_res.h5'
        tensorboard = TensorBoard(log_dir = "logs/latest_model")
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=False,
        monitor='loss',
        mode='min',
        save_best_only=False)

        model = self.Model()
        model.summary()
        model.compile(optimizer = tf.keras.optimizers.SGD(learning_rate = 0.1,momentum=0.9), loss = self.L1Loss)
        model.fit(self.Utils.LoadH5File(training_lr_path)/255,
        self.Utils.LoadH5File(training_hr_path)/255,
        batch_size = 1,
        epochs= num_of_epochs,
        callbacks=[CustomLearningRateScheduler(self.schedular), early_stop, tensorboard, model_checkpoint_callback])

model = SuperResModel()
model.PredictAndShowImage(model.loadModel('super_res.h5', compile = False), data_path=r"D:\HBO\MinorAi\Div2kx4\valid_lr", read_from_directory = True)

#model.TrainModel( r"D:\HBO\MinorAi\PickleFiles\X_train_1.h5", r"D:\HBO\MinorAi\PickleFiles\y_train_1.h5", num_of_epochs = 1)