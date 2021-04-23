import os
import cv2
import pickle
import numpy as np
import h5py
class Utils:
    def __init__(self,train_lr_path = r"D:\HBO\MinorAi\data\LR",train_hr_path = r"D:\HBO\MinorAi\data\HR",test_lr_path = r"D:\HBO\MinorAi\data\LR",test_hr_path = r"D:\HBO\MinorAi\data\HR"):
        self.train_lr_path = train_lr_path
        self.train_hr_path = train_hr_path
        self.test_lr_path = test_lr_path
        self.test_hr_path = test_hr_path
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
    def GetCroppedImages(self,low_res_array,high_res_array):
        lowpix = 96
        highpix = 384
        low_res_cropped = []
        high_res_cropped = []
        for i in low_res_array:
            for a in range(0,i.shape[0],lowpix):
                for b in range(0,i.shape[1],lowpix):
                    image = i[a:a+lowpix,b:b+lowpix]
                    if image.shape == (lowpix,lowpix,3): 
                        low_res_cropped.append(image)

        for i in high_res_array:
            for a in range(0,i.shape[0],highpix):
                for b in range(0,i.shape[1],highpix):
                    image = i[a:a+highpix,b:b+highpix]
                    if image.shape == (highpix,highpix,3):
                        high_res_cropped.append(image)

        return (np.array(low_res_cropped),np.array(high_res_cropped))

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

    def SaveAsH5File(self,data, directory, dtype = h5py.h5t.STD_I32LE):
        file = h5py.File(directory, "w")
        dataset = file.create_dataset("images", np.shape(data), dtype, data= data)
        file.close()

    def LoadH5File(self,path):
        file = h5py.File(path, "r+")
        return np.array(file["images"]).astype(np.float32)

    def Run(self):
        self.train_lr = self.ReadImages(self.train_lr_path)
        self.train_hr = self.ReadImages(self.train_hr_path)
        # self.test_lr = self.ReadImages(self.test_lr_path)
        # self.test_hr = self.ReadImages(self.test_hr_path)
        self.train_lr, self.train_hr = self.GetCroppedImages(self.train_lr, self.train_hr)
        self.SaveDataToFile(self.train_lr, r"D:\HBO\MinorAi\PickleFiles\train_lr.pickle")
        self.SaveDataToFile(self.train_hr, r"D:\HBO\MinorAi\PickleFiles\train_hr.pickle")

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

test = Utils(train_lr_path=r"D:\HBO\MinorAi\Div2kx4\train_lr", train_hr_path=r"D:\HBO\MinorAi\Div2kx4\train_hr", test_lr_path=r"D:\HBO\MinorAi\Div2kx4\valid_lr", test_hr_path = r"D:\HBO\MinorAi\Div2kx4\valid_lr")
#test.SaveAugmentedImages(r"D:\HBO\MinorAi\PickleFiles\train_hr.pickle", r"D:\HBO\MinorAi\PickleFiles\train_hr_rot_90.pickle")
#test.SaveAsH5File(test.LoadDataFromFile(r"D:\HBO\MinorAi\PickleFiles\train_hr.pickle"),r"D:\HBO\MinorAi\PickleFiles\train_hr.h5")
#test.SaveAsH5File(test.LoadDataFromFile(r"D:\HBO\MinorAi\PickleFiles\train_lr.pickle"),r"D:\HBO\MinorAi\PickleFiles\train_lr.h5")
test.LoadH5File(r"D:\HBO\MinorAi\PickleFiles\train_hr.h5")