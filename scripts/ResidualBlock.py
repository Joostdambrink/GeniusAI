import tensorflow as tf
from tensorflow.keras.layers import Conv2D,PReLU,Concatenate
from tensorflow.keras.initializers import Constant
from keras.layers.merge import add
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
    def ResBlock(self,input_data,num_of_units = 3, num_of_filters = 64):

        resunit = []
        for i in range(num_of_units):
            if i == 0:
                resUnit = self.ResidualUnit(input_data,num_of_filters)
            else:
                resUnit = self.ResidualUnit(resUnit, num_of_filters)
        return resUnit


    """Residual unit that consists of 2 Conv2D layers and one PReLU layer
    
    Args:
        input_data (numpy array / tf tensor) : input data of this unit
        num_of_filters (Int) : number of filters per Conv2D layer
    
    Returns:
        (keras layer) : result of the last layer in the unit
    """
    def ResidualUnit(self,input_data, num_of_filters):
        identity = input_data
        x = Conv2D(num_of_filters,(2,2),padding="same")(identity)
        x = PReLU(alpha_initializer=Constant(value=0.25),shared_axes=[1,2])(x)
        x = Conv2D(num_of_filters,(2,2),padding = "same")(x)
        x += identity
        x = PReLU(alpha_initializer=Constant(value=0.25),shared_axes=[1,2])(x)
        return x