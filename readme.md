# PyTorch Projects










### Step by Step setup environment with Cloud9

##### Install MiniConda
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod a+x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh

##### Create virtual environment
conda create -n py3 python=3 ipython
source activate py3
conda install pip

##### Packages
pip install pandas
pip install numpy

##### Pytorch
conda install pytorch-cpu torchvision-cpu -c pytorch


##### Create requirements.txt (venv)
pip freeze --local > requirements.txt