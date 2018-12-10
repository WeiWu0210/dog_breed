import json
import os
import numpy as np
from tqdm import tqdm
import tensorflow as tf
import cv2
from glob import glob

from flask import Flask
from flask import render_template, request, jsonify
from keras.models import model_from_json
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.preprocessing import image
from keras.models import load_model

app = Flask(__name__)

def path_to_tensor(img_path):
    """
    convert image file to a 4D tensor
    Args:
        img_path: path to image file
    Returns:
        4D tensor of the image file
    """
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

# returns "True" if face is detected in image stored at img_path
def face_detector(img_path):
    """
    Human face detector
    Args:
        img_path: path to image file
    Returns:
        True if any human face detected
        False if no human face detected
    """
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # extract pre-trained face detector
    face_cascade = cv2.CascadeClassifier('../data/haarcascade_frontalface_alt.xml')
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0


def ResNet50_predict_labels(img_path):
    """
    Predict dog breed using Resnet50 pre-trained model
    Args:
        img_path: path to image file
    Returns:
        god breed index
    """
    # returns prediction vector for image located at img_path
    img = preprocess_input(path_to_tensor(img_path))
    return np.argmax(ResNet50_model.predict(img))

### returns "True" if a dog is detected in the image stored at img_path
def dog_detector(img_path):
    """
    dog detector
    Args:
        img_path: path to image file
    Returns:
        True if any dog detected
        False if no dog detected
    """
    prediction = ResNet50_predict_labels(img_path)
    return ((prediction <= 268) & (prediction >= 151))

# Extract bottlenect features
def extract_Resnet50(tensor):
    return ResNet50(weights='imagenet', include_top=False).predict(preprocess_input(tensor))

# Predict breed
def predict_breed(img_path):
    """
    Predict dog breed using Resnet50 pre-trained model
    Args:
        img_path: path to image file
    Returns:
        god breed index
    """
    bottleneck_feature = extract_Resnet50(path_to_tensor(img_path))
    # obtain predicted vector
    #predicted_vector = loaded_model.predict(bottleneck_feature)
    predicted_vector = model_full.predict(bottleneck_feature)
    #print(predicted_vector) 
    # return dog breed that is predicted by the model
    dog_name= dog_names[np.argmax(predicted_vector)]
    return  dog_name
    #return predicted_vector

# Load trained model
model_full = load_model('../models/model_full.h5')
print("Loaded model")
#set folder path for image uploade
UPLOAD_FOLDER = "../uploads"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
#save graph info
graph = tf.get_default_graph()
# define ResNet50 model
ResNet50_model = ResNet50(weights='imagenet')
# load list of dog names
dog_names = [item[20:-1] for item in sorted(glob("../data/dogImages/train/*/"))]

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    # This will render the master.html  
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    #Get the image file uploaed by a user
    file = request.files['image']
    f = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    #remove previous uploaded imgage
    dirPath = app.config['UPLOAD_FOLDER']
    fileList = os.listdir(dirPath)
    for fileName in fileList:
         os.remove(dirPath+"/"+fileName)
    #save the file to the folder for uploads
    file.save(f)
    #Need to run the model within saved graph
    global graph
    with graph.as_default():
        if face_detector(f):
            result = 'This person looks like '+predict_breed(f).split('.')[1]
        elif dog_detector(f):
            result = 'The breed of this dog is '+predict_breed(f).split('.')[1]
        else:
            result ='invalid image input! '
        print(result)
    
    return render_template('index.html',result=result)



def main():
    app.run(host='0.0.0.0', port=3001, debug=True)
   
if __name__ == '__main__':
    main()
