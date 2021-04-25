import tensorflow as tf
from tensorflow.keras.layers import Conv2D,PReLU,Concatenate
from tensorflow.keras.initializers import Constant
from keras.layers.merge import add
class ResidualBlock:
    def __init__(self):
        return

    """
        Residual block:
            it takes the input data in the form of an array/tensor, number of residual units, and number of filters per convolutional layer
            returns the result of all the residual units in the form of a convolutional layer.
    """
    def ResBlock(self,input_data,num_of_units = 3, num_of_filters = 64):
        resunit = []
        for i in range(num_of_units):
            if i == 0:
                resUnit = self.ResidualUnit(input_data,num_of_filters)
            else:
                resUnit = self.ResidualUnit(resUnit, num_of_filters)
        return resUnit

    """
        Residual unit:
            a function that takes in input data as an array/tensor and number of filters for each convolution.
            returns the result of the residual block as a convolutional layer.
    """
    def ResidualUnit(self,inputdata, num_of_filters):
        x = Conv2D(num_of_filters,(2,2),padding="same")(inputdata)
        x = PReLU(alpha_initializer=Constant(value=0.25),shared_axes=[1,2])(x)
        x = Conv2D(num_of_filters,(2,2),padding = "same")(x)
        return x