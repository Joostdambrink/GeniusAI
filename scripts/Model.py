import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Activation, UpSampling2D,PReLU,Input,LeakyReLU,Add,Dropout
from tensorflow.keras import Sequential
from tensorflow.keras.initializers import Constant
from time import time
from time import gmtime, strftime
from tensorflow.keras.callbacks import TensorBoard
import pickle
import numpy as np
import cv2
from ResidualBlock import ResidualBlock
from CustomSchedular import CustomLearningRateScheduler
from Utils import Utils
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import keras.backend as K

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
    def loadModel(self,model_path):
        model = tf.keras.models.load_model(model_path, custom_objects = {'L1Loss' : self.L1Loss})
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
        image = np.expand_dims(data[0], axis = 0)
        pred = model.predict(image)
        cv2.imshow("prediction",pred[0])
        cv2.imshow("input",cv2.resize(image[0], (pred[0].shape[1],pred[0].shape[0]), interpolation=cv2.INTER_CUBIC))
        cv2.imwrite(r"D:\HBO\MinorAi\test\test2.jpg", pred[0]*255)
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
        c1 = Conv2D(64,3,input_shape = (None,None,3),**conv_args)(inputs)
        pr1 = PReLU(alpha_initializer="zeros",shared_axes=[1,2])(c1)

        resUnit = rb.ResBlock(pr1,num_of_units = 24,num_of_filters = 64)
        
        concat = Add()([pr1,resUnit])
        
        c2 = Conv2D(64,3,**conv_args)(concat)

        c3 = Conv2D(64,3,**conv_args)(c2)
        pr2 = PReLU(alpha_initializer="zeros",shared_axes=[1,2])(c3)
        c4 = Conv2D(64,3,**conv_args)(pr2)
        d1 = Dropout(0.3)(c4)
        c5 = Conv2D(64,3,**conv_args)(d1)
        pr3 = PReLU(alpha_initializer="zeros",shared_axes=[1,2])(c5)

        up1 = UpSampling2D(size = (2,2),interpolation = "nearest")(pr3)
        c6 = Conv2D(64,5,**conv_args)(up1)
        leaky = LeakyReLU()(c6)
        up2 = UpSampling2D(size = (2,2), interpolation="nearest")(leaky)
        c7 = Conv2D(64,5,**conv_args)(up2)
        leaky2 = LeakyReLU()(c7)
        d2 = Dropout(0.3)(leaky2)
        c9 = Conv2D(64,5,**conv_args)(d2)
        pr4 = PReLU(alpha_initializer="zeros",shared_axes=[1,2])(c9)
        c10 = Conv2D(3,5,**conv_args)(pr4)

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

    """loads a model and gets its weights

    Args:
        model_path (str) : path of saved model
        compile (bool) : whether to compile the model or not
    
    Returns:
        (numpy array) : weights of the loaded model
    """
    def GetModelWeights(self,model_path):
        model = self.loadModel(model_path)
        return model.get_weights()


    """Trains keras model on input data

    Args:
        X_train_path (str) : path of low res training data
        y_train_path (str) : path of high res training data
        num_of_epochs (Int, optional) : number of training epochs
    """
    def TrainModel(self,model,X_train_path , y_train_path, num_of_epochs = 100, checkpoint_filepath = None,existing_weights = None, load_weights = False):
        tensorboard = TensorBoard(log_dir = "logs/latest_model_{}".format(strftime("%d_%m_%Y", gmtime())))
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=False,
        monitor='loss',
        mode='min',
        save_best_only=False)
        if load_weights and not existing_weights == None:
            model.set_weights(existing_weights)
        model.summary()
        model.compile(optimizer = tf.keras.optimizers.SGD(learning_rate = 0.1,momentum=0.9), loss = self.L1Loss)
        model.fit(self.Utils.LoadH5File(X_train_path)/255,
        self.Utils.LoadH5File(y_train_path)/255,
        batch_size = 1,
        validation_split = 0.1,
        epochs= num_of_epochs,
        callbacks=[CustomLearningRateScheduler(self.schedular), early_stop, tensorboard, model_checkpoint_callback])
    

    """Resumes training after sudden stop of a model
    
    Args:
        X_train_path (str) : X training data path
        y_train_path (str) : y training data path
        model_path (str) : path of model to load
        checkpoint_filepath (str) : path to save the model
        num_of_epochs (int) : number of training epochs
    """
    def ResumeTraining(self,X_train_path, y_train_path,model_path,checkpoint_filepath = None, num_of_epochs = 100):
        tensorboard = TensorBoard(log_dir = "logs/latest_model_{}".format(strftime("%d_%m_%Y", gmtime())))
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=False,
        monitor='loss',
        mode='min',
        save_best_only=False)
        model = self.loadModel(model_path)
        latest_epoch = self.GetEpochFromLearningRate(K.eval(model.optimizer.lr))
        model.fit(self.Utils.LoadH5File(X_train_path)/255,
        self.Utils.LoadH5File(y_train_path)/255,
        batch_size = 1,
        epochs= num_of_epochs,
        initial_epoch = latest_epoch,
        callbacks=[CustomLearningRateScheduler(self.schedular), early_stop, tensorboard, model_checkpoint_callback])
        


    """Calculates epoch index based on learning rate

    Args:
        lr (float) : learning rate

    Returns:
        (int) : epoch number
    """
    def GetEpochFromLearningRate(self,lr):
        count = 0
        start_value = 0.10000
        lr_rounded = round(lr,5)
        while start_value > lr_rounded:
            start_value /= 2
            count += 1
        return count * 20

model = SuperResModel()
model.PredictAndShowImage(model.loadModel('super_res.h5'), data_path=r"D:\HBO\MinorAi\test", read_from_directory = True)