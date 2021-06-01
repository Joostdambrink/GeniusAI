import tensorflow as tf
from tensorflow.keras.layers import Conv2D,PReLU,Add,Input,LeakyReLU
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
    def ResBlock(self,input_data,num_of_units = 3, num_of_filters = 16, kernel_size = (3,3), inc_filters = False):
        
        res_unit = input_data
        block_size = num_of_units // 3
        for i in range(num_of_units):
            res_input = res_unit
            if i == 0:
                res_unit = self.ResidualUnit(input_data,num_of_filters, kernel_size = kernel_size)
            else:
                res_unit = self.ResidualUnit(res_unit, num_of_filters, kernel_size = kernel_size)
                res_unit = Add()([res_input, res_unit])
                res_unit = LeakyReLU()(res_unit)
            if i % block_size == 0 and inc_filters:
                num_of_filters *= 2
        
        return res_unit


    """Residual unit that consists of 2 Conv2D layers and one PReLU layer
    
    Args:
        input_data (numpy array / tf tensor) : input data of this unit
        num_of_filters (Int) : number of filters per Conv2D layer
    
    Returns:
        (keras layer) : result of the last layer in the unit
    """
    def ResidualUnit(self,input_data, num_of_filters, kernel_size = (3,3)):
        x = Conv2D(num_of_filters,kernel_size,padding="same")(input_data)
        x = LeakyReLU()(x)
        x = Conv2D(num_of_filters,kernel_size,padding = "same")(x)
        return x