import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.models import load_model
from sklearn.datasets import load_files       
from keras.utils import np_utils
from glob import glob
#load data and define model architecture

# define function to load train, test, and validation datasets
def load_dataset(path):
    """
    Load data of dog images
    Args:
        path: path to files of dog images
    Returns:
        dog_files: files of dog images
        dog_targets: targets of the specified dog_files
    """
    data = load_files(path)
    dog_files = np.array(data['filenames'])
    dog_targets = np_utils.to_categorical(np.array(data['target']), 133)
    return dog_files, dog_targets

# load train, test, and validation datasets
train_files, train_targets = load_dataset('../data/dogImages/train')
valid_files, valid_targets = load_dataset('../data/dogImages/valid')
test_files, test_targets = load_dataset('../data/dogImages/test')

# load list of dog names
dog_names = [item[20:-1] for item in sorted(glob("../data/dogImages/train/*/"))]
### Obtain bottleneck features from another pre-trained CNN.
bottleneck_features = np.load('../data/bottleneck_features/DogResnet50Data.npz')
train_Resnet50= bottleneck_features['train']
valid_Resnet50 = bottleneck_features['valid']
test_Resnet50 = bottleneck_features['test']

### Define your architecture.
Resnet50_model = Sequential()
Resnet50_model.add(GlobalAveragePooling2D(input_shape=train_Resnet50.shape[1:]))
Resnet50_model.add(Dense(133, activation='softmax'))

Resnet50_model.summary()

### Compile the model.
Resnet50_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])


### Train the model.
checkpointer = ModelCheckpoint(filepath='../models/weights.best.Resnet50.hdf5', 
                               verbose=1, save_best_only=True)

Resnet50_model.fit(train_Resnet50, train_targets, 
          validation_data=(valid_Resnet50, valid_targets),
          epochs=100, batch_size=20, callbacks=[checkpointer], verbose=1)

### TODO: Load the model weights with the best validation loss.
Resnet50_model.load_weights('../models/weights.best.Resnet50.hdf5')


### TODO: Calculate classification accuracy on the test dataset.
# get index of predicted dog breed for each image in test set
Resnet50_predictions = [np.argmax(Resnet50_model.predict(np.expand_dims(feature, axis=0))) for feature in test_Resnet50]

# report test accuracy
test_accuracy = 100*np.sum(np.array(Resnet50_predictions)==np.argmax(test_targets, axis=1))/len(Resnet50_predictions)
print('Test accuracy: %.4f%%' % test_accuracy)


#save full model
Resnet50_model.save('../models/model_full.h5')
