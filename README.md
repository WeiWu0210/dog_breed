# Project Overview
A data science project that classifies images of dogs according to their breed using deep learning and convolutional neural networks algorithms and tools.

The project includes a web app that accepts any user-supplied image as input. If a dog is detected in the image, it will provide an estimate of the dog's breed. If a human is detected, it will provide an estimate of the dog breed that is most resembling. . 

<img src="/data/snapshot.png" alt="screenshot"/>

# Instructions
0. Download the dog dataset. Unzip the folder and save it to folder /dogImages 

1. Run the following commands in the models directory to train model.
    - `python train.py`

2. Run the following command in the app's directory to run your web app.
    - `python run.py`

3. Go to http://0.0.0.0:3001/

# Important Files:

models/train.py: The Machine Learning pipeline used to fit, tune, evaluate, and export the model. The trained model will be saved as "model_full.h5"

app/templates/*.html: HTML templates for the web app.

app/run.py: Start the Python server for the web app.

data/dogImages: Data used for model training and evaluation

data/bottlenect_features/DogResnet50Data.npz: ResNet50 bottleneck features

data/haarcascade_frontalface_alt.xml: pre-trained face detectors by OpenCV stored as XML

notebook/dog_app.ipynb: Jupyter notebook of this project
