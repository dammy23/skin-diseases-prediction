# -*- coding: utf-8 -*-
"""
@author: Krish.Naik
"""

from keras.callbacks import ModelCheckpoint
from keras.models import load_model
import tensorflow as tf
from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model
from keras.applications.resnet_v2 import ResNet152V2
from keras.applications.resnet_v2 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
import numpy as np
from glob import glob
import matplotlib.pyplot as plt

# re-size all the images to this
IMAGE_SIZE = [224, 224]

train_path = 'skin/train'
valid_path = 'skin/test'

# add preprocessing layer to the front of VGG
resnet = ResNet152V2(input_shape=IMAGE_SIZE + [3],
                     weights='imagenet', include_top=False)

# don't train existing weights
for layer in resnet.layers:
    layer.trainable = False

    # useful for getting number of classes
folders = glob('skin/train/*')
class_test = glob('skin/test/*')

# our layers - you can add more if you want
x = Flatten()(resnet.output)
# x = Dense(1000, activation='relu')(x)
prediction = Dense(len(folders), activation='softmax')(x)

# create a model object
model = Model(inputs=resnet.input, outputs=prediction)

# view the structure of the model
model.summary()

# tell the model what cost and optimization method to use
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)
checkpoint = ModelCheckpoint('resnett_model_improved.h5',  # model filename
                             monitor='val_loss',  # quantity to monitor
                             verbose=0,  # verbosity - 0 or 1
                             save_best_only=True,  # The latest best model will not be overwritten
                             mode='auto')  # The decision to overwrite model is made
# automatically depending on the quantity to monitor

train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

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
plt.plot(r.history['loss'], label='resnet train loss')
plt.plot(r.history['val_loss'], label='resnet val loss')
plt.legend()
plt.show()
plt.savefig('LossVal_loss')

# accuracies
plt.plot(r.history['accuracy'], label='resnet train acc')
plt.plot(r.history['val_accuracy'], label='resnet val acc')
plt.legend()
plt.show()
plt.savefig('AccVal_acc')

scores = model.evaluate(training_set)
print("Accuracy: %.2f%%" % (scores[1]*100))
scorest = model.evaluate(test_set)
print("Accuracy: %.2f%%" % (scorest[1]*100))
model.save('resnett_model')
