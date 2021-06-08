import os, time, argparse
from Models import SuperResModels
from test import TestModels

parser = argparse.ArgumentParser(description='Residual Dense Network for single-image Super Resulotion')
parser.add_argument('--lr_path', type=str, help='path for the low_res images')
parser.add_argument('--hr_path', type=str, help='path for the high_res images')
parser.add_argument('--save_path', type=str, help='path for saving')
args = parser.parse_args()

if __name__ == '__main__':
    test_model = TestModels()
    models = SuperResModels()
    start = time.time()
    model = test_model.loadModel(str(os.path.abspath("../super_res_denoiser")), is_custom= True)
    lr_path = args.lr_path
    hr_path = args.hr_path
    save_path = args.save_path
    lr_files = os.listdir(lr_path)
    hr_files = os.listdir(hr_path)
    for lr_file,hr_file in zip(lr_files,hr_files):
        test_model.PlotAndSave(model,os.path.join(lr_path,lr_file), os.path.join(hr_path,hr_file), os.path.join(save_path,lr_file), save_output= True)
