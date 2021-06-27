import tensorflow as tf
from tensorflow.keras.layers import Conv2D, UpSampling2D,Input,LeakyReLU,Add,BatchNormalization,MaxPool2D, Lambda, PReLU
from ResidualBlock import ResidualBlock
from Utils import Utils
from CustomLayers import DownSample, UpSample


class SuperResModels:
    def __init__(self):
        self.Utils = Utils()

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
            "activation" : LeakyReLU(alpha = 0.2),
            "kernel_initializer": "Orthogonal",
            "padding": "same",
        }
        rb = ResidualBlock()
        inputs = Input(shape=[None,None,3])
        c1 = Conv2D(64,5,input_shape = (None,None,3),**conv_args)(inputs)
        pr1 = res = PReLU(alpha_initializer=tf.keras.initializers.Constant(value=0.25),shared_axes=[1,2])(c1)
        for _ in range(24):
            res = Conv2D(64,(2,2),padding="same")(res)
            res = PReLU(alpha_initializer=tf.keras.initializers.Constant(value=0.25),shared_axes=[1,2])(res)
            res = Conv2D(64,(2,2),padding = "same")(res)

        add1 = Add()([pr1,res])
        c2 = Conv2D(64,4,**conv_args)(add1)
        
        c3 = Conv2D(64,4,**conv_args)(c2)
        act = PReLU(alpha_initializer=tf.keras.initializers.Constant(value=0.25),shared_axes=[1,2])(c3)
        c4 = Conv2D(64,4,**conv_args)(act)

        c5 = Conv2D(64,3,**conv_args)(c4)
        act = PReLU(alpha_initializer=tf.keras.initializers.Constant(value=0.25),shared_axes=[1,2])(c5)

        up1 = UpSampling2D(size = (2,2),interpolation = "nearest")(act)
        c6 = Conv2D(64,2, padding = "same")(up1)
        act = LeakyReLU(alpha = 0.2)(c6)
        up2 = UpSampling2D(size = (2,2), interpolation="nearest")(act)
        c7 = Conv2D(64,2,**conv_args)(up2)
        act = LeakyReLU(alpha = 0.2)(c7)

        c8 = Conv2D(64,2,**conv_args)(act)
        c9 = Conv2D(64,2,**conv_args)(c8)
        act = PReLU(alpha_initializer=tf.keras.initializers.Constant(value=0.25),shared_axes=[1,2])(c9)
        c10 = Conv2D(3,2,**conv_args)(act)

        return tf.keras.Model(inputs = inputs, outputs = c10)

    """
        Denoising model:
            this is the convolutional model used for the denoising task.
            the model contains the following layers:
                - Convolutional layer (relu)
                - Batch normalization
                - Max pooling layer
                - Convolutional layer (relu)
                - Batch normalization
                - Max pooling layer
                
                - Convolutional layer (relu)
                - Batch normalization 
                - Upsampling layer
                - Convolutional layer (relu)
                - Batch normalization
                - Upsampling layer
                - Last convolutional layer (sigmoid)

            returns a Keras model with all these layers.
    """
    def DenoisingModel(self):
        rb = ResidualBlock()
        inputs = Input(shape =[None,None,3])
        c1 = Conv2D(64, (7,7), activation = 'relu', padding = 'same')(inputs)
        bn1 = BatchNormalization()(c1)
        p2 = MaxPool2D(pool_size = (2,2), padding = 'same')(bn1)
        c2 = Conv2D(64, (5,5), activation = 'relu', padding = 'same')(p2)
        bn2 = BatchNormalization()(c2)
        res_net = rb.ResBlock(bn2, num_of_blocks= 5, num_of_filters = 64)
        encoded = MaxPool2D(pool_size = (2,2), padding = 'same')(res_net)

        c3 = Conv2D(64, (3,3), activation = 'relu', padding = 'same')(encoded)
        bn3 = BatchNormalization()(c3)
        up1 = UpSampling2D()(bn3)
        c4 = Conv2D(64, (3,3), activation = 'relu', padding = 'same')(up1)
        bn4 = BatchNormalization()(c4)
        up2 = UpSampling2D()(bn4)
        decoded = Conv2D(3, (3,3), activation = 'sigmoid', padding = 'same')(up2)
        output = LeakyReLU()(decoded)

        return tf.keras.Model(inputs = inputs, outputs = output, name = "denoiser_autoenc" )
    
    def Resampler(self, batch_size):
        rb = ResidualBlock()
        inputs = Input(shape =[192,192,3], batch_size= batch_size)
        c1 = Conv2D(3, 3, padding = "same", activation = LeakyReLU())(inputs)
        down1 = DownSample(2,64,array_shape = c1.shape, parent_name = "down_1")(c1)
        down2 = DownSample(2,128,array_shape = down1.shape, parent_name = "down_1")(down1)
        res_net = rb.ResBlock(down2,num_of_blocks = 1, num_of_filters=128)
        down2 = DownSample(2,256,array_shape = res_net.shape, parent_name = "down_1")(res_net)
        c5 = Conv2D(256, 3, padding = "same", activation = LeakyReLU())(down2)
        c6 = Conv2D(256, 3, padding = "same", activation = LeakyReLU())(c5)
        c7 = Conv2D(256, 3, padding = "same", activation = LeakyReLU())(c6)
        upsample = UpSample(2,256)(c7)
        c3 = Conv2D(3, 1,padding = "same", activation = LeakyReLU())(upsample)
        model = tf.keras.Model(inputs = inputs, outputs = c3)
        model.summary()
        return model
    
    def EDSR(self):
        rb = ResidualBlock()
        inputs = Input(shape=[None,None,3], batch_size=None)
        norm = Lambda(self.Utils.normalize)(inputs)

        c1  = res_block = Conv2D(64, 3, padding = "same")(norm)
        res_block = rb.ResBlock(res_block,num_of_blocks = 16, num_of_filters=64, residual_scaling = 0.1)
        res_block = Conv2D(64, 3, padding='same')(res_block)
        added = Add()([c1, res_block])

        upsample = UpSample(2,64)(added)
        upsample = UpSample(2,64)(upsample)
        c3 = Conv2D(3, 3,padding = "same")(upsample)
        
        outputs = Lambda(self.Utils.denormalize)(c3)
        model = tf.keras.Model(inputs = inputs, outputs = outputs)
        return model


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

