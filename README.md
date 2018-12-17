# Project Overview
A data science project that classifies images of dogs according to their breed using deep learning and convolutional neural networks algorithms and tools.

The project includes a web app that accepts any user-supplied image as input. If a dog is detected in the image, it will provide an estimate of the dog's breed. If a human is detected, it will provide an estimate of the dog breed that is most resembling. . 

<img src="/data/snapshot.png" alt="screenshot"/>

# Instructions
1. Download the dog dataset (https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip). Unzip and save it to folder "data/dogImages" 

2. Download the human dataset (https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/lfw.zip). Unzip and save it to folder "data/lfw"

3. Donwload the bottleneck features for the dog dataset. Place it to folder "data/bottleneck_features".
- VGG16: https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogVGG16Data.npz
- VGG19: https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogVGG19Data.npz
- Resnet50: https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/ResNET50Data.npz


1. Run the following commands in the directory  "models" to train model.
    - `python train.py`

2. Run the following command in the  directory "app' to run your web app.
    - `python run.py`

3. Go to http://0.0.0.0:3001/

# Important Files:

Requirement.txt: List of Python packages required. Note Python 3.6.5 is used as Tensorflow does not support Pyhton3.7 yet as of the time of this project (12/2018).

models/train.py: The Machine Learning pipeline used to fit, tune, evaluate, and export the model. The trained model will be saved as "model_full.h5".

app/templates/*.html: HTML templates for the web app.

app/run.py: Start the Python server for the web app.

data/dogImages: Data used for model training and evaluation

data/bottlenect_features/DogResnet50Data.npz: ResNet50 bottleneck features

data/haarcascade_frontalface_alt.xml: pre-trained face detectors by OpenCV stored as XML

notebook/dog_app.ipynb: Jupyter notebook of this project
