# GeniusAI
## _Single Image Super Resolution Model_

An CNN Model used for enhancing the resolution on images.

- [Features](#Features)
- [Installation](#Installation)
- [References](#References)

## Features

- Enhance a single image resolution 4x
- Load websites faster by loading low res images and using model to enhance

## Installation

Our Model requires Python to run

Download the dependencies with:
```sh
python setup.py
```

To run our program open the terminal use:
```sh
python run.py --lr_path <path> --hr_path <path> --save_path <path>
```
| Parameter | Description |
| ------ | ------ |
| --lr_path | Path where the low-res images are stored |
| --hr_path | Path where the high-res images are stored |
| --save_path | Path where the results are stored |

### EDSR
#### Architecture
![arch_edsr](images/super_res_edsr.png)
#### EDSR Results
![result_edsr](images/results/edsr.PNG)


### Our model
#### Architecture
![arch_model](images/super_res_model_scheme.png)
#### Out model Results
![result_model](images/results/model_resultPNG.PNG)


### Our model + denoiser
#### Architecture
![arch_edsr](images/model_denoiser.png)
#### Our model + denoiser Results
![result_edsr](images/results/model_denoiser.PNG)


# Refereces
https://arxiv.org/pdf/1707.02921.pdf
https://arxiv.org/pdf/1907.12904v2.pdf
https://github.com/krasserm/super-resolution
