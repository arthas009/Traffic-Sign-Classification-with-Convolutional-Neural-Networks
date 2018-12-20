#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 04:18:20 2018

@author: yusuf
"""
import random
import keras
import os
import numpy as np  
import pandas as pd
from skimage import transform
from skimage import data
import skimage as sm
from skimage import color
from skimage.color import rgb2gray
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.utils import Sequence
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.models import load_model
import h5py
import tensorflow as tf
from PIL import Image

#THOSE WHO WANTS TO WORK WITH GPU
#DELETE THIS SECTION IF YOU DON'T WANT TO WORK WITH GPU
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
config = tf.ConfigProto(gpu_options = gpu_options,log_device_placement=True)
config.gpu_options.allow_growth=True
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
keras.backend.set_session(sess)

#LOADING IMAGE DATAS AND THEIR LABELS
def load_data(data_directory):
    directories = [d for d in os.listdir(data_directory) 
                   if os.path.isdir(os.path.join(data_directory, d))]
    labels = []
    images = []
    for d in directories:
        label_directory = os.path.join(data_directory, d)
        file_names = [os.path.join(label_directory, f) 
                      for f in os.listdir(label_directory) 
                      if f.endswith(".ppm")]
        for f in file_names:
            images.append(sm.data.imread(f))
            labels.append(int(d))
    return images, labels 

#PREPARE A CONVOLUTIONAL NEURAL NETWORKS TAKES 64X64X3 (3 FOR RGB) AS INPUT 
def prepareClassifier():
    # ilkleme
    classifier = Sequential()

    # STEP 1 - Convolution
    # 64,64,3 = 64X64 IMAGE DIMENSIONS, 3 FOR RGB
    # 32,3 = 32 SUBSAMPLES, 3X3 CONVOLUTON MATRIX DIMENSIONS, 1X1 STRIDE DIMENSIONS, PADDING IS VALID

    classifier.add(Convolution2D(32,3, strides=(1,1),input_shape = (64, 64, 3),padding='valid', activation = 'relu'))

    # STEP 2 - Pooling
    # 2,2 = POOL MATRIX DIMENSIONS
    classifier.add(MaxPooling2D(pool_size = (2,2)))

    # STEP 3 - 2. Convolution
    # 64,64,3 = 64X64 IMAGE DIMENSIONS, 3 FOR RGB
    # 32,3 = 32 SUBSAMPLES, 3X3 CONVOLUTON MATRIX DIMENSIONS, 1X1 STRIDE DIMENSIONS, PADDING IS VALID
    classifier.add(Convolution2D(32, 3,strides=(1,1),activation = 'relu', padding='valid'))
	# STEP 4 - Convolution
     # 2,2 = POOL MATRIX DIMENSIONS
    classifier.add(MaxPooling2D(pool_size = (2,2)))
    # Adım 5 - Flattening
    # FLATTENING FOR THE FULLY CONNECTED LAYER
    classifier.add(Flatten())

    # Adım 5 - NEURAL NETWORK
    # 128 = HIDDEN LAYER BITS
    classifier.add(Dense(output_dim = 128, activation = 'relu'))
    #62 - OUTPUT BITS FOR 62 CLASSES. SOFTMAX IS SUGGESTED FOR OUTPUT
    classifier.add(Dense(output_dim = 62, activation = 'softmax'))

    # CNN
    # OPTIMIZE ACCURACY WITH ADAM OPTIMIZER AND CATEGORICAL CROSSENTROPY LOSS FUNCTION
    classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    return classifier

# OUR CURRENT MODEL
classifier = prepareClassifier()

#CNN AND IMAGES WITH IMAGEDATA GENERATOR WITH DATA AUGMENTATION FOR INCREASING THE SUCCESS

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./150,shear_range = 0.2, zoom_range = 0.2)

test_datagen = ImageDataGenerator(rescale=1./150,shear_range = 0.2, zoom_range = 0.2)

training_set = train_datagen.flow_from_directory("TrafficSigns/Training", 
                                                 target_size = (64, 64), batch_size = 8, class_mode = 'categorical')

test_set = test_datagen.flow_from_directory("TrafficSigns/Testing",
                                            target_size = (64, 64), batch_size = 8, class_mode = 'categorical') 
#4552 = TRAINING IMAGES, 2520 = TEST IMAGES
classifier.fit_generator(training_set, samples_per_epoch = 4552, nb_epoch =50, 
                         validation_steps=2520, validation_data = test_set )

classifier.save('./savedmodel/my_model.h5')  # creates a HDF5 file 'my_model.h5'
# returns a compiled model
# identical to the previous one
#model = load_model('./savedmodel/my_model.h5')

import matplotlib.pyplot as plt

# YOU MUST SET ROOT_PATH AND OTHER IMAGE DATA GENERATOR'S PATH AS YOUR OWN DATA SET'S LOCATION
ROOT_PATH = "C:/Users/Yusuf/Documents/makineogrenmesi/tensorflowdeneme3"

test_data_directory = os.path.join(ROOT_PATH, "TrafficSigns/Testing")

#LOAD IMAGES AND LABELS
imagesTest,labelsTest = load_data(test_data_directory)
unique_labels = set(labelsTest)

# RESIZE ALL IMAGES TO 64X64
images28 = [transform.resize(image, (64, 64)) for image in imagesTest] 

#CONVERT IT TO ARRAY
images28 = np.array(images28)

# Pick 10 random images
sample_indexes = random.sample(range(len(images28)), 10)
# 10 RANDOM IMAGE
sample_images = [images28[i] for i in sample_indexes]
# 10 RANDOM IMAGES'S LABELS
sample_labels = [labelsTest[i] for i in sample_indexes]

#CONVER THEM TO ARRAY
sample_images_array=np.array(sample_images)

predicted = classifier.predict(sample_images_array)

'''
#SINGLE IMAGE TESTING
testimage = load_img("./BelgiumTSC_Testing/Testing/00008/00587_00001.ppm",target_size=(64,64))
testimage = img_to_array(testimage)
testimage = np.expand_dims(testimage, axis=0)

singletest = classifier.predict(testimage)

print("--Single test--")
print(np.argmax(singletest[0]))
'''

# PRINT 10 RANDOM IMAGES PREDICTIONS AND TRUE VALUES WITH THEIR IMAGE
print("----10 RANDOM TEST SAMPLES---")
#Display the predictions and the ground truth visually.
fig = plt.figure(figsize=(10, 10))
for i in range(len(sample_images)):
    truth = sample_labels[i]
    prediction = np.argmax(predicted[i])
    plt.subplot(5, 2,1+i)
    plt.axis('off')
    color='green' if truth == prediction else 'red'
    plt.text(70, 20, "Truth:        {0}\nPrediction: {1}".format(truth, prediction), 
             fontsize=12, color=color)
    plt.imshow(sample_images[i],  cmap="gray")

plt.show()
# TEST WHOLE TESTING DATASET
array = classifier.predict(images28)

# COUNT THE TRUE PREDICTIONS
result = []
for i in range (0,len(array)):
    result.append(np.argmax(array[i]))             
result2 = []
true=0
for i in range (0,len(labelsTest)):
    if labelsTest[i] == int(result[i]):
        true = true+1
        result2.append(1)
    else:
        result2.append(0)
    
print("----ALL TEST SAMPLES---")
print("True predicted:"+str(true)+" / Total: "+str(len(labelsTest))+" = "+str(true/len(labelsTest)))

print("done")