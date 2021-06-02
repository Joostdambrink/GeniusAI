import tensorflow as tf
from tensorflow.keras.layers import Permute, Conv2D
import numpy as np
class PixelUnshuffle(tf.keras.layers.Layer):
    def __init__(self, scale, array_shape = (1,96,96,32)):
        super(PixelUnshuffle, self).__init__()
        self.scale = scale
        self.array_shape = array_shape

    def call(self, inputs):
        b,h,w,c = self.array_shape
        output_channels = c * self.scale ** 2
        output_height = int(h / self.scale)
        output_width = int(w / self.scale)

        output_reshaped = tf.reshape(inputs,[b,c,output_height,self.scale, output_width, self.scale], name = "reshape_1")
        output_permuted = Permute((1,2,4,6,3,5), name = "permute_1")((tf.expand_dims(output_reshaped, 0)))
        output= tf.reshape(output_permuted, [b,output_channels,output_height, output_width], name = "reshape_2")

        output = Permute((1,3,4,2))((tf.expand_dims(output, 0)))
        output= tf.reshape(output, [b,output_height, output_width, output_channels], name = "reshape_2")

        return output

class DownSample(tf.keras.layers.Layer):
    def __init__(self, scale, filters, ksize = 1, array_shape = (1,96,96,32)):
        super(DownSample, self).__init__()
        self.downsample = tf.keras.Sequential(
            [
                PixelUnshuffle(scale, array_shape = array_shape),
                Conv2D(filters, ksize, strides = 1, padding = "same")
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

