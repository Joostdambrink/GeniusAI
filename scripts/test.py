import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.layers import Lambda
from PIL import Image
from PIL import ImageEnhance
from Utils import Utils
from Models import SuperResModels
class TestModels:
    def __init__(self):
        self.Utils = Utils()
    """Loads keras model from path
    
    Args:
        model_path (str) : path of model to load
        compile (bool, optional) : compile model automatically True or False. defaults to True
    
    Returns:
        (keras model) : the loaded model.
    """
    def loadModel(self,model_path, is_custom = True):
        if is_custom:
            return tf.keras.models.load_model(model_path, custom_objects = {'L1Loss' : self.L1Loss})
        else:
            return tf.keras.models.load_model(model_path, custom_objects={'normalize': Lambda(self.Utils.normalize), 'denormalize' : Lambda(self.Utils.denormalize)})


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
        denoiser = self.loadModel('denoiser', is_custom = False)
        denoised = denoiser.predict(pred)
        original_denoise = denoiser.predict(image)
        # cv2.imwrite(r"D:\HBO\MinorAi\test\prediction.png", pred[0]/255)
        # cv2.imwrite(r"D:\HBO\MinorAi\test\denoised.png", denoised[0]*255)
        # cv2.imwrite(r"D:\HBO\MinorAi\test\denoised_original.png", original_denoise[0]*255)
        # cv2.imwrite(r"D:\HBO\MinorAi\test\bicubic.png", cv2.resize(image[0], (pred[0].shape[1],pred[0].shape[0]), interpolation=cv2.INTER_CUBIC)*255)
        # cv2.imwrite(r"D:\HBO\MinorAi\test\scaling_2x.png", cv2.resize(pred[0], (pred[0].shape[1]//2,pred[0].shape[0]//2), interpolation=cv2.INTER_CUBIC)*255)
        # cv2.imwrite(r"D:\HBO\MinorAi\test\sharp.png", np.float32(self.Utils.SharpenImage(pred[0])))
        # cv2.imwrite(r"D:\HBO\MinorAi\test\sharp_denoise.png", np.float32(self.Utils.SharpenImage(denoised[0])))
        # cv2.imwrite(r"D:\HBO\MinorAi\test\smooth.png", np.float32(self.Utils.SmoothImage(pred[0])))
        images = [ ("prediction", self.Utils.ReverseColors(pred[0] / 255)), ("after sharpening", self.Utils.SharpenImage(self.Utils.ReverseColors(pred[0]))), ("denoised", self.Utils.ReverseColors(denoised[0])), ("upscaled bicubic", self.Utils.ReverseColors(cv2.resize(image[0], (pred[0].shape[1],pred[0].shape[0]), interpolation=cv2.INTER_CUBIC)))]
        rows = 1
        cols = 4
        axes=[]
        fig=plt.figure()
        for a in range(rows*cols):
            axes.append( fig.add_subplot(rows, cols, a+1) )
            axes[-1].set_title(images[a][0]) 
            plt.imshow(images[a][1])

        fig.tight_layout()    
        plt.show()
    
    def resolve_single(self,model, lr):
        return self.resolve(model, tf.expand_dims(lr, axis=0))[0]

    def resolve(self,model, lr_batch):
        lr_batch = tf.cast(lr_batch, tf.float32)
        sr_batch = model(lr_batch)
        sr_batch *= 255
        sr_batch = tf.clip_by_value(sr_batch, 0, 255)
        sr_batch = tf.round(sr_batch)
        sr_batch = tf.cast(sr_batch, tf.uint8)
        return sr_batch

    def plot_sample(self,lr, sr, sr_2, hr):
        plt.figure(figsize=(20, 10))

        images = [lr, sr, sr_2, hr]
        titles = ['LR', f'SR (x{sr.shape[0] // lr.shape[0]})', f'SR_enhanced (x{sr.shape[0] // lr.shape[0]})', f'HR']

        for i, (img, title) in enumerate(zip(images, titles)):
            plt.subplot(1, len(images), i+1)
            plt.title(title)
            plt.xticks([])
            plt.yticks([])
            plt.imshow(img)
        plt.show()


if __name__ == '__main__':
    test_model = TestModels()
    # x = test_model.loadModel("super_res_denoiser", is_custom = False)
    # hr = cv2.cvtColor(np.array(Image.open(r"D:\HBO\MinorAi\test\Capture_hr.png")), cv2.COLOR_BGRA2BGR)
    # lr = cv2.cvtColor(np.array(Image.open(r"D:\HBO\MinorAi\test\Capture.png")), cv2.COLOR_BGRA2BGR)
    # sr = test_model.resolve_single(x, lr / 255)
    # sr_2 = test_model.resolve_single(x.layers[1], sr/255)
    # sr_2 = test_model.resolve_single(x.layers[1], sr_2/255)
    # img = Image.fromarray(np.array(sr))
    # enhancer = ImageEnhance.Contrast(img)
    # img = enhancer.enhance(1.3)
    # img.save(r"D:\HBO\MinorAi\test\Capture_4x_enhanced.png")
    # test_model.plot_sample(lr,sr, img, hr)
    model = test_model.loadModel("super_res_denoiser", is_custom = False)
    model.save_weights("super_res_denoiser_weights.h5")

