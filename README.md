# bacteria-SANet
Source code and demos for "Scale-adaptive Deep Model for Bacterial Raman Spectra Identification"

# Requirements
* Python 3.7.7
* Pytorch 1.7.0
* torchvision 0.8.1
* Numpy 1.18.5
* Scikit-learn 0.23.1
* Matplotlib 3.2.2
* Jupyter 6.2.0
* Seaborn 0.10.1
* tqdm 4.46.1

# Model
![Scale-adaptive-model](https://github.com/DenglinGo/bacteria-SANet/blob/main/model.png)  
The detail of the model can be found in [model.py](https://github.com/DenglinGo/bacteria-SANet/blob/main/model.py)

# Data
The data for the demos can be downloaded [here](https://www.dropbox.com/sh/gmgduvzyl5tken6/AABtSWXWPjoUBkKyC2e7Ag6Da?dl=0) and should be saved in the data directory.
* wavenumbers.npy : The wavenumbers of all spectra.
* X_reference.npy,y_refence.npy : reference spectra and their isolate-level labels.
* X_finetune.npy,y_finetune.npy : spectra used for finetuning and their isolate-level labels.
* X_test.npy,y_test.npy : spectra used for testing and their isolate-level labels.
* X_2018clinical.npy,y_2018clinical.npy,X_2019clinical.npy,y_2019clinical.npy : clinical spectra and their antibiotic-level labels.
