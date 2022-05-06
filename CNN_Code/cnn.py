import numpy as np
import time
from helpers import progressbar
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.feature import hog
from skimage.transform import resize
from scipy.spatial.distance import cdist
from scipy.stats import mode
from sklearn.cluster import KMeans
from sklearn import svm
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
#import tensorflow_addons as tfa
import pickle5 as pickle
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report

from sklearn.model_selection import GridSearchCV
#from keras.wrappers.scikit_learn import KerasClassifier
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers import Dense,Dropout,MaxPooling2D, Flatten,BatchNormalization, GaussianNoise,Conv2D
import os
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dropout, Flatten



def get_images(image_paths):

    images = []
    for im_path in image_paths:
        im = imread(im_path)
        if len(im.shape) == 3:
            im = rgb2gray(im)
        im = resize(im, (70,70), anti_aliasing=True)
        im = np.asarray(im)
        images.append(im)
        
    images = np.expand_dims(np.asarray(images), axis=-1)


    return images




def cnn_classify(train_image_feats, train_labels, test_image_feats):
    
    train_labels = np.expand_dims(np.asarray(train_labels), axis=1)

    model = Sequential()
    model.add(Conv2D(100, 3, 1,  activation='relu', padding="same", input_shape = (70,70, 1)))
    model.add(MaxPooling2D(2, padding="same"))

    model.add(Dropout(0.5))
    model.add(Conv2D(100, (3, 3), activation='relu'))
    #model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(50, 3, 1, activation='relu'))
    model.add(MaxPooling2D(2, padding="same"))

    model.add(Dropout(0.3))
    model.add(Conv2D(32, 3, 1, activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    #model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(2, activation = "softmax"))
    
    
    model.compile(Adam(learning_rate = 0.0005), "sparse_categorical_crossentropy", metrics = ["sparse_categorical_accuracy"])
    
    model.summary()
    

    history = model.fit(train_image_feats, train_labels, epochs=50, batch_size=32, validation_split=0.2, verbose=1) #validation_split=0.1 

    #model.save('cnn_model.h5')
    return history, model.predict(test_image_feats)

