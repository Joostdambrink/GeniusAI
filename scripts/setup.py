import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

"""
Setup function for installing all the needed packages
"""
def setup():
    packages = ["tensorflow","numpy","argparse","matplotlib","opencv-python","Pillow","h5py"]
    for package in packages :
        try:
            install(package)
        except:
            print(f"{package} could not be installed")

if __name__ == '__main__' :
    setup()
    
