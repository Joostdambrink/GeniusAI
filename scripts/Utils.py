import os
import cv2
import pickle
import numpy as np
import h5py
from PIL import Image, ImageEnhance
from PIL import ImageFilter
class Utils:
    def __init__(self):
        self.train_lr = []
        self.train_hr = []
        self.test_lr = []
        self.test_hr = []
    
    """Reads images from a directory

    Args:
        path (str) : directory of images

    Returns:
        (numpy array) : array containing all the images in the directory
    """
    def ReadImages(self,path):
        images = []
        for filename in os.listdir(path):
            img = cv2.imread(os.path.join(path,filename))
            if img is not None:
                images.append(np.array(img))
        return np.array(images)


    """crops images from an array

    Args:
        images_array (numpy array) : array of images to crop
        crop_size (Int) : size of cropped piece i.e crop_size = 96 ===> array of images with shape (num_of_images, 96, 96, 3)

    Returns:
        (numpy array) : numpy array of cropped images
    """
    def GetCroppedImages(self,images_array, crop_size):
        cropped_images = []
        for i in images_array:
            for a in range(0,i.shape[0],crop_size):
                for b in range(0,i.shape[1],crop_size):
                    image = i[a:a+crop_size,b:b+crop_size]
                    if image.shape == (crop_size,crop_size,3): 
                        cropped_images.append(image)

        return (np.array(cropped_images))


    """rotates given images to given rotation

    Args:
        images (numpy array) : array of images to rotate
        rotation (cv2 rotation, optional) : degrees of rotation. defaults to cv2.ROTATE_90_CLOCKWISE

    Returns:
        (generator) : generator with rotated images
    """
    def ImageAugmentations(self,images, rotation = cv2.ROTATE_90_CLOCKWISE):
        for i in range(len(images)):
            self.printProgressBar(i,len(images),prefix="Augmenting", suffix = "Complete")
            yield cv2.rotate(images[i],rotation)


    """Saves data to pickle binary file
    
    Args:
        data (any python type) : data to save into file
        path (str) : path of the file
    """
    def SaveDataToFile(self,data,path):
        with open(path, 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    

    """loads data from pickle binary file

    Args:
        path (str) : path of file to load data from

    Returns:
        (any python type) : whatever data the file contains
    """
    def LoadDataFromFile(self,path):
        return pickle.load( open( path, "rb" ) )


    """saves input array to h5 file

    Args:
        data (numpy array) : data you want to save
        directory (str) : directory of where you want to save the data
        chunk_shape (tuple, optional) : shape of data chunk. Defaults to (200,96,96,3).
        dtype (h5py type, optional) : type of data to save. Defaults to h5py.h5t.STD_I32LE.
    """
    def SaveAsH5File(self,data, directory,chunk_shape = (200,96,96,3), dtype = h5py.h5t.STD_I32LE):

        file = h5py.File(directory, "w",rdcc_nbytes =1024**2*1024,rdcc_w0 = 1)
        dataset = file.create_dataset("images", np.shape(data), dtype, data= data, chunks = chunk_shape, compression="lzf")
        file.close()


    """Loads data from h5 file

    Args:
        path (str) : path of the file that contains the data

    Returns:
        (numpy array) : the data in the file as a numpy array
    """
    def LoadH5File(self,path):
        file = h5py.File(path, "r+")
        return np.array(file["images"]).astype(np.float32)


    """Call in a loop to create terminal progress bar

    Args:
        iteration (integer) : current iteration
        total (integer) : total iterations
        prefix (str, optional) : prefix string. Defaults to ''.
        suffix (str, optional) : suffix string. Defaults to ''.
        decimals (int, optional) : positive number of decimals in percent complete. Defaults to 1.
        length (int, optional) : character length of bar. Defaults to 100.
        fill (str, optional) : bar fill character. Defaults to '█'.
        printEnd (str, optional) : end character. Defaults to "\r".
    """
    def printProgressBar (self,iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
        percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
        filledLength = int(length * iteration // total)
        bar = fill * filledLength + '-' * (length - filledLength)
        print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
        # Print New Line on Complete
        if iteration == total: 
            print()

    """Saves data in multiple files

    Args:
        data (numpy array) : data to save
        directory (str) : saving directory
        prefix (str, optional) : name of file to save to. Defaults to "images".
        suffix (str, optional) : type of file to save to. Defaults to ".h5".
        num_of_splits (int, optional) : the number of splits of data. Defaults to 5.
    """
    def SaveInBatches(self, data, directory,prefix = "images", suffix = ".h5", num_of_splits = 5):
        step_size = len(data) // num_of_splits
        current_split = 1
        for i in range(0,len(data), step_size):
            data_sliced = data[i : i+step_size]
            self.SaveAsH5File(data_sliced, directory + "\\" + prefix + "_" + str(current_split) + suffix )
            current_split += 1


    """shows the first images of x and y

    Args:
        x (numpy array) : array of images
        y (numpy array) : array of images
    """
    def DisplayImages(self, x, y):
        cv2.imshow("image_1", x[0]/255)
        cv2.imshow("image_2", y[0]/255)
        cv2.waitKey()
        cv2.destroyAllWindows()


    """ downscales an image

    Args:
        images : array of images
        factor : the factor for the downscaling

    Returns:
        resized image
    """
    def DownscaleImage(self, img,  factor):
        new_shape = (img.shape[1] // factor, img.shape[0] // factor)
        
        return cv2.resize(img, new_shape, interpolation= cv2.INTER_AREA)


    """ downscales multiple images from a path

    Args:
        path: string where the images are
        factor: factor for the downscaling

    Returns:
        array with resized images
    """
    def DownscaleImages(self, path, factor):
        images = self.ReadImages(path)
        downscaled_images = [self.DownscaleImage(image, factor) for image in images]

        return np.array(downscaled_images)
    
    """Adds sharpen filter to given images

    Args:
        image (np array) : input image as array

    Returns:
        (array) : enhanced image in an array
    """
    def SharpenImage(self, image):
        x = (image * 255).astype(np.uint8)
        x = Image.fromarray(x)
        enhancer = ImageEnhance.Sharpness(x)
        return enhancer.enhance(1)
    

    """Reverses the colors of given images

    Args:
        image (np array) : array of input image

    Returns:
        (array) : image with reversed colors
    """
    def ReverseColors(self,image):
        return np.array(cv2.cvtColor(np.float32(image), cv2.COLOR_BGR2RGB))


    """Adds noise to an array of images

    Args:
        images (array) : array of images
        noise (float) : noise factor

    Returns:
        (array) : images with added noise
    """    
    def AddNoise(self, images, noise = 0.1):
        noisyImages = []
        for x in images:
            x = x + noise + np.random.normal(loc = 0., scale = 1., size = None)
            noisyImages.append(x)

        return noisyImages

if __name__ == "__main__":
    test = Utils()
    test.SaveAsH5File(test.DownscaleImages(r"D:\HBO\MinorAi\1m_faces_00_01_02_03\1m_faces_01", 64), r"D:\HBO\MinorAi\PickleFiles\X_train_faces_2.h5", chunk_shape=(200,16,16,3))