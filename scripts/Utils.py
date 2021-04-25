import os
import cv2
import pickle
import numpy as np
import h5py
class Utils:
    def __init__(self):
        self.train_lr = []
        self.train_hr = []
        self.test_lr = []
        self.test_hr = []
    
    """
        ReadImages:
            Reads images from a given path.
            Returns a numpy array with all the images.
    """
    def ReadImages(self,path):
        images = []
        for filename in os.listdir(path):
            img = cv2.imread(os.path.join(path,filename))
            if img is not None:
                images.append(np.array(img))
        return np.array(images)

    """
        GetCroppedImages
            Gets a input of numpy array images (low-res and high-res)
            Crops the images into multiple smaller images and returns these.
    """
    def GetCroppedImages(self,images_array, crop_size):
        low_res_cropped = []
        for i in images_array:
            for a in range(0,i.shape[0],crop_size):
                for b in range(0,i.shape[1],crop_size):
                    image = i[a:a+crop_size,b:b+crop_size]
                    if image.shape == (crop_size,crop_size,3): 
                        low_res_cropped.append(image)

        return (np.array(low_res_cropped))

    """
    this function takes two image pandas arrays (one low res and the other high res) and does the following transformations:
        - rotations 90,180,270
        - each rotated image gets added to the array.
    returns:
        - a tuple with both arrays (new_array_lr, new_array_hr)
    """
    def ImageAugmentations(self,images, rotation = cv2.ROTATE_90_CLOCKWISE):
        for i in range(len(images)):
            self.printProgressBar(i,len(images),prefix="Augmenting", suffix = "Complete")
            yield cv2.rotate(images[i],rotation)


    """
        SaveDataToFile:
            takes in the data to save and the save path and saves the data to the path with pickle.
            returns void
    """
    def SaveDataToFile(self,data,path):
        with open(path, 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    """
        LoadDataFromFile:
            takes in path of the data as input.
            returns the python data structure equivalent of the data in the file.
    """
    def LoadDataFromFile(self,path):
        return pickle.load( open( path, "rb" ) )

    def SaveAsH5File(self,data, directory,chunck_shape = (200,96,96,3), dtype = h5py.h5t.STD_I32LE):
        chunck_shape = chunck_shape
        file = h5py.File(directory, "w",rdcc_nbytes =1024**2*1024,rdcc_w0 = 1)
        dataset = file.create_dataset("images", np.shape(data), dtype, data= data, chunks = chunck_shape, compression="lzf")
        file.close()

    def LoadH5File(self,path):
        file = h5py.File(path, "r+")
        return np.array(file["images"]).astype(np.float32)

    def Run(self):
        self.train_lr = self.ReadImages(r"D:\HBO\MinorAi\Div2kx4\train_lr")
        self.train_hr = self.ReadImages(r"D:\HBO\MinorAi\Div2kx4\train_hr")
        self.train_lr, self.train_hr = self.GetCroppedImages(self.train_lr, self.train_hr)
        self.SaveAsH5File(self.train_lr,r"D:\HBO\MinorAi\PickleFiles\train_lr.h5")
        self.SaveAsH5File(self.train_hr,r"D:\HBO\MinorAi\PickleFiles\train_hr.h5")

    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    def printProgressBar (self,iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
        percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
        filledLength = int(length * iteration // total)
        bar = fill * filledLength + '-' * (length - filledLength)
        print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
        # Print New Line on Complete
        if iteration == total: 
            print()
    
    def SaveAugmentedImages(self,images_path, save_path):
        print("loading images from file....")
        images = self.LoadDataFromFile(images_path)

        print("starting lr images augmentation...")
        images_rot_90_lr = []
        for item in self.ImageAugmentations(images):
            images_rot_90_lr.append(item)

        return np.array(images_rot_90_lr)

    def SaveInBatches(self, data, directory,prefix = "images", suffix = ".h5", num_of_splits = 5):
        step_size = len(data) // num_of_splits
        current_split = 1
        for i in range(0,len(data), step_size):
            data_sliced = data[i : i+step_size]
            self.SaveAsH5File(data_sliced, directory + "\\" + prefix + "_" + str(current_split) + suffix )
            current_split += 1

    def DisplayImages(self, x, y):
        cv2.imshow("image_1", x[0]/255)
        cv2.imshow("image_2", y[0]/255)
        cv2.waitKey()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    test = Utils()