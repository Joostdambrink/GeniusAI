import tensorflow as tf
from tensorflow.keras.layers import Permute, Conv2D
import numpy as np
class PixelUnshuffle(tf.keras.layers.Layer):
    def __init__(self, scale, array_shape = (1,96,96,32), parent_name = ""):
        super(PixelUnshuffle, self).__init__()
        self.scale = scale
        self.array_shape = array_shape
        self.parent_name = parent_name
        self.b , h , w , c = self.array_shape
        self.output_channels = c * self.scale ** 2
        self.output_height = int(h / self.scale)
        self.output_width = int(w / self.scale)
        self.reshape_1 = lambda x :  tf.reshape(x,[self.b ,c,self.output_height,self.scale, self.output_width, self.scale], name = self.parent_name + "reshape_1")

    def call(self, inputs):

        # c = inputs.shape[-1]
        # self.b = 16


        output_reshaped = self.reshape_1(inputs)
        output_permuted = Permute((1,2,4,6,3,5), name = self.parent_name + "permute_1")((tf.expand_dims(output_reshaped, 0)))
        output= tf.reshape(output_permuted, [self.b ,self.output_channels,self.output_height, self.output_width], name = self.parent_name + "reshape_2")

        output = Permute((1,3,4,2), name = self.parent_name + "permute_2")((tf.expand_dims(output, 0)))
        output= tf.reshape(output, [self.b ,self.output_height, self.output_width, self.output_channels], name = self.parent_name + "reshape_3")

        return output
    def get_config(self):
        return {"scale": self.scale, "array_shape" : self.array_shape}

class DownSample(tf.keras.layers.Layer):
    def __init__(self, scale, filters, ksize = 1, array_shape = (1,96,96,32), parent_name = ""):
        super(DownSample, self).__init__()
        self.downsample = tf.keras.Sequential(
            [
                PixelUnshuffle(scale, array_shape = array_shape, parent_name = parent_name),
                Conv2D(filters, ksize, strides = 1, padding = "same", name = parent_name + "conv_1")
            ]
        )
    
    def call(self,inputs):
        return self.downsample(inputs)

class UpSample(tf.keras.layers.Layer):
    def __init__(self, factor, num_of_filters):
        super(UpSample, self).__init__()
        self.factor = factor
        self.num_of_filters = num_of_filters
        self.conv = Conv2D(self.num_of_filters * (self.factor ** 2), 3, padding = "same")

    
    def call(self, inputs):
        return tf.nn.depth_to_space(self.conv(inputs), self.factor)

    def get_config(self):
        return {"factor": self.factor, "num_of_filters" : self.num_of_filters}
