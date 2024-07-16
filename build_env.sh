# create new python 3.11 conda env
conda create --prefix ./tf_osx_2.15.0 python==3.11

# activate
conda activate ./tf_osx_2.15.0

# update pip in env
pip install --upgrade pip

# install tensorflow 2.15.0 via pip
python -m pip install tensorflow==2.15.0 

# install further dependencies as needed
python -m pip install tensorflow-datasets pandas tensorflow-probability
