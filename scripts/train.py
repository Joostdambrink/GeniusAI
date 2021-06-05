import datetime as datetime
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
from CustomSchedular import CustomLearningRateScheduler
from Models import SuperResModels

class TrainModels:
    
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


    """Trains keras model on input data

    Args:
        X_train_path (str) : path of low res training data
        y_train_path (str) : path of high res training data
        num_of_epochs (Int, optional) : number of training epochs
    """
    def TrainModel(self,model = None,X_train_path = None , y_train_path = None, num_of_epochs = 100, checkpoint_filepath = None,existing_weights = None, load_weights = False, optimizer = None,logsdir = None, batch_size = 5, callbacks_extra = [], **args):
        log_dir = logsdir + "/" +  datetime.now().strftime("%d_%m_%y_%H%M%S")
        tensorboard = TensorBoard(log_dir = log_dir)
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
        model.compile(optimizer = optimizer, loss = tf.keras.losses.MeanAbsoluteError())
        X_train = self.Utils.LoadH5File(X_train_path)
        X_train /= 255
        data_gen = tf.keras.preprocessing.image.ImageDataGenerator(
            horizontal_flip= True,
            vertical_flip = True,
            validation_split= 0.1,
            dtype = tf.float32
        )
        data_gen.fit(X_train)
        self.Utils.LoadH5File(y_train_path)/255,
        model.fit(data_gen.fit(X_train, self.Utils.LoadH5File(y_train_path)/255, batch_size = batch_size),
        epochs= num_of_epochs,
        callbacks= [tensorboard, model_checkpoint_callback] + callbacks_extra)
    
    
    def TrainCAR(self,model = None,batch_size = 1,
    X_train_path = None , y_train_path = None,
    num_of_epochs = 100, checkpoint_filepath = None,
    existing_weights = None, load_weights = False,
    optimizer = None,logsdir = None, **args):
        tensorboard = TensorBoard(log_dir = logsdir)
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
        model.compile(optimizer = optimizer, loss = tf.keras.losses.MeanAbsoluteError())
        data = self.Utils.LoadH5File(X_train_path)
        data = data / 255
        data_gen = tf.keras.preprocessing.image.ImageDataGenerator(
            horizontal_flip = True,
            vertical_flip = True,
        )
        data_gen.fit(data)
        model.fit(data_gen.flow(data,data, batch_size = batch_size),
        steps_per_epoch = len(data) / batch_size,
        epochs= num_of_epochs,
        callbacks=[tensorboard, model_checkpoint_callback])

    

if __name__ == '__main__':
    model = SuperResModels()
    trainer = TrainModels()
    super_res = model.loadModel("super_res.h5")

    for layer in super_res.layers :
        layer.trainable = False
    batch_size = 16
    denoiser_model = tf.keras.Sequential([super_res, model.DenoisingModel()], name ="super_res_denoise")
    denoiser_model.load_weights("super_res_denoiser_weights.h5")
    optim_args = {
        "learning_rate" : 1e-5,
        "beta_1" : 0.9,
        "beta_2" : 0.999,
        "epsilon" : 1e-8
    }
    model_args = {
        "model" : denoiser_model,
        "X_train_path" : r"/home/genai-admin/Data/X_train_48.h5",
        "y_train_path" : r"/home/genai-admin/Data/y_train_192.h5",
        "num_of_epochs" : 100,
        "logsdir" : "denoiser_logs",
        "checkpoint_filepath" : "super_res_denoiser_v2",
        "existing_weights" : None,
        "load_weights" : False,
        "optimizer" : tf.keras.optimizers.Adam(**optim_args),
        "batch_size" : batch_size,
        "callbacks_extra" : [CustomLearningRateScheduler(trainer.schedular)]
    }

    trainer.TrainModel(**model_args)