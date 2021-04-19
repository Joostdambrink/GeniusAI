import os
import cv2
import pickle
import numpy as np
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
    this function takes two image pandas arrays (one low res and the other high res) and does the following transformations:
        - rotations 90,180,270
        - each rotated image gets added to the array.
    returns:
        - a tuple with both arrays (new_array_lr, new_array_hr)
    """
    def ImageAugmentations(self,images_lr, images_hr):
        return

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

    def Run(self):
        self.train_lr = self.ReadImages(self.train_lr_path)[:80]
        self.train_hr = self.ReadImages(self.train_hr_path)[:80]
        self.test_lr = self.ReadImages(self.test_lr_path)[80::]
        self.test_hr = self.ReadImages(self.test_hr_path)[80::]
        self.SaveDataToFile(self.train_lr,"D:\\HBO\\MinorAi\\GeniusAI\\Data\\train_lr.pickle")
        self.SaveDataToFile(self.train_hr,"D:\\HBO\\MinorAi\\GeniusAI\\Data\\train_hr.pickle")
        self.SaveDataToFile(self.test_lr,"D:\\HBO\\MinorAi\\GeniusAI\\Data\\test_lr.pickle")
        self.SaveDataToFile(self.test_hr,"D:\\HBO\\MinorAi\\GeniusAI\\Data\\test_hr.pickle")

        # for i in range(1,9):
        #     self.SaveDataToFile(self.train_lr[(i-1) * 100 : i * 100], "D:\\HBO\\MinorAi\\GeniusAI\\Data\\train_lr_div2k" + "_" + str(i) + ".pickle")
        #     self.SaveDataToFile(self.train_hr[(i-1) * 100 : i * 100], "D:\\HBO\\MinorAi\\GeniusAI\\Data\\train_hr_div2k" + "_" + str(i) + ".pickle")
        # self.SaveDataToFile(self.test_lr, "D:\\HBO\\MinorAi\\GeniusAI\\Data\\test_lr_div2k.pickle")
        # self.SaveDataToFile(self.test_hr, "D:\\HBO\\MinorAi\\GeniusAI\\Data\\test_hr_div2k.pickle")

test = Utils()
test.Run()