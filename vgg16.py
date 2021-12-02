# -*- coding: utf-8 -*-
"""
@author: Krish.Naik
"""

from keras.callbacks import ModelCheckpoint
from keras.models import load_model
import tensorflow as tf
from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
import numpy as np
from glob import glob
import os
import matplotlib.pyplot as plt
import pickle
import cv2  # opencv
# To make one-hot vectors
from keras.utils import np_utils

# re-size all the images to this
IMAGE_SIZE = [224, 224]

train_path = 'skin/train'
test_path = 'skin/test'
num_classes = 3


# add preprocessing layer to the front of VGG
vgg = VGG16(input_shape=IMAGE_SIZE + [3],
            weights='imagenet', include_top=False)

# don't train existing weights
for layer in vgg.layers:
    layer.trainable = False

# useful for getting number of classes
folders = glob('skin/train/*')
class_test = []
for imgfolder in os.listdir(test_path):
    for filename in os.listdir(test_path + '/' + imgfolder):
        filename = test_path + '/' + imgfolder + '/' + filename
        # print(filename)
        img = cv2.imread(filename, 0)
        class_test.append(img)
class_test = np.asarray(class_test)

# our layers - you can add more if you want
x = Flatten()(vgg.output)
# x = Dense(1000, activation='relu')(x)
prediction = Dense(len(folders), activation='softmax')(x)

# create a model object
model = Model(inputs=vgg.input, outputs=prediction)

# view the structure of the model
model.summary()

# tell the model what cost and optimization method to use
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)
checkpoint = ModelCheckpoint('vgg16_model_improved.h5',  # model filename
                             monitor='val_loss',  # quantity to monitor
                             verbose=0,  # verbosity - 0 or 1
                             save_best_only=True,  # The latest best model will not be overwritten
                             mode='auto')  # The decision to overwrite model is made
# automatically depending on the quantity to monitor

train_datagen = ImageDataGenerator()

test_datagen = ImageDataGenerator()


training_set = train_datagen.flow_from_directory('skin/train',
                                                 target_size=(224, 224),
                                                 batch_size=32,
                                                 class_mode='categorical')

test_set = test_datagen.flow_from_directory('skin/test',
                                            target_size=(224, 224),
                                            batch_size=32,
                                            class_mode='categorical')

'''r=model.fit_generator(training_set,
                         samples_per_epoch = 8000,
                         nb_epoch = 5,
                         validation_data = test_set,
                         nb_val_samples = 2000)'''

# fit the model
r = model.fit_generator(
    training_set,
    validation_data=test_set,
    epochs=50,
    steps_per_epoch=len(training_set),
    callbacks=[checkpoint],
    validation_steps=len(test_set)
)
# loss
plt.plot(r.history['loss'], label='vgg train loss')
plt.plot(r.history['val_loss'], label='vgg val loss')
plt.legend()
plt.show()
plt.savefig('LossVal_loss')

# accuracies
plt.plot(r.history['accuracy'], label='vgg train acc')
plt.plot(r.history['val_accuracy'], label='vgg val acc')
plt.legend()
plt.show()
plt.savefig('AccVal_acc')

scores = model.evaluate(training_set)
print("Accuracy: %.2f%%" % (scores[1]*100))
scorest = model.evaluate(test_set)
print("Accuracy: %.2f%%" % (scorest[1]*100))
model.save('vgg16_model')
