import tensorflow as tf
from tensorflow.keras.layers import Conv2D,PReLU,Add,Input,LeakyReLU, Lambda
from tensorflow.keras.initializers import Constant
class ResidualBlock:
    def __init__(self):
        return

    """Creates residual block from n number of residual units

    Args:
        input_data (numpy array / tf tensor) : input data of the first unit
        num_of_units (Int, optional) : number of residual units
        num_of_filters : number of filters per Conv2D layer

    Returns:
        (keras layer) : the last layer of the residual block
    """
    def ResBlock(self,input_data, num_of_blocks = 6, num_of_filters = 16, kernel_size = (3,3), residual_scaling = None):
        
        res_unit = input_data
        for i in range(num_of_blocks):
            res_unit = self.ResidualUnit(res_unit, num_of_filters, residual_scaling, kernel_size = kernel_size)

        
        return res_unit


    """Residual unit that consists of 2 Conv2D layers and one PReLU layer
    
    Args:
        input_data (numpy array / tf tensor) : input data of this unit
        num_of_filters (Int) : number of filters per Conv2D layer
    
    Returns:
        (keras layer) : result of the last layer in the unit
    """
    def ResidualUnit(self,input_data, num_of_filters, residual_scaling, kernel_size = (3,3), strides = (1,1)):
        x = Conv2D(num_of_filters,kernel_size,padding="same", strides= strides, activation = LeakyReLU())(input_data)
        x = Conv2D(num_of_filters,kernel_size,padding = "same",strides= strides)(x)
        if residual_scaling:
            x = Lambda(lambda t: t * residual_scaling) (x)
        x = Add()([input_data, x])
        return x