import tensorflow as tf
from tensorflow.keras.layers import Conv2D,PReLU
from tensorflow.keras.initializers import Constant
class ResidualBlock:
    def __init__(self):
        return

    """
        Residual block:
            it takes the input data in the form of an array/tensor, number of residual units, and number of filters per convolutional layer
            returns the result of all the residual units in the form of a convolutional layer.
    """
    def ResBlock(self,input_data,num_of_units = 3, num_of_filters = 64):
        resUnit = self.ResidualUnit(input_data,num_of_filters)
        for i in range(num_of_units - 1):
            resUnit = self.ResidualUnit(resUnit,num_of_filters)
        return resUnit

    """
        Residual unit:
            a function that takes in input data as an array/tensor and number of filters for each convolution.
            returns the result of the residual block as a convolutional layer.
    """
    def ResidualUnit(self,inputdata, num_of_filters):
        x = Conv2D(num_of_filters,(6,6),padding="same")(inputdata)
        x = PReLU(alpha_initializer=Constant(value=0.25),shared_axes=[1,2])(x)
        x = Conv2D(num_of_filters,(6,6),padding = "same")(x)
        return x