import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input
from tensorflow.keras.callbacks import * #as ModelCheckpoint#, ReduceLROnPlateau
#from livelossplot import PlotLossesKeras
#from livelossplot.keras import PlotLossesCallback
#from keras.layers import Conv2D
#from keras.layers import MaxPooling2D, Input, add, GlobalAveragePooling2D
#from keras.layers.core import Dense, Flatten, Activation, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
#from keras.layers.normalization import BatchNormalization
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.metrics import confusion_matrix
from tensorflow.keras.applications import Xception, VGG16, MobileNetV2, MobileNet, ResNet50, InceptionV3
import os
#from keras.models import model_from_json





new_input = Input(shape=(224, 224, 3))

def genData(train, val):
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train,
        target_size=(224, 224),
        batch_size=64,
        class_mode='categorical')
    validation_generator = test_datagen.flow_from_directory(
        val,
        target_size=(224, 224),
        batch_size=1024,
        class_mode='categorical')
    os.system("rm -rf ./valSet/.ipynb_checkpoints")
    os.system("rm -rf ./dataset/.ipynb_checkpoints")
    return train_generator,validation_generator

def trainModel(model,aug_train,aug_valid,weights = None,epochs= None):
    if epochs == None : epochs = 30
    if weights != None : weights = 'models/'+weights
    if model.lower() == 'vgg16':
        #model = VGG16(include_top=False, input_tensor=new_input)
        checkpoint = ModelCheckpoint("models/model_Vgg16.h5", monitor='val_accuracy',
                             save_weights_only=True, mode='max', verbose=1)
        callbacks = [checkpoint]
        model = VGG16(
            include_top=True,
            weights=weights,
            input_tensor=new_input,
            input_shape=(224,224,3),
            pooling='avg',
            classes=8,
            classifier_activation="softmax",
         )
        save_model_json = model.to_json()
        with open("models/modelVgg16.json", "w") as json_file:
            json_file.write(save_model_json )
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics='accuracy')
        model.fit(aug_train,
                  validation_data=aug_valid,
                  steps_per_epoch = (287654//1024),
                  epochs = epochs,
                  validation_steps = (3999//1024),
                  callbacks = callbacks
                 )
    elif model.lower() == 'xception':
        #model = VGG16(include_top=False, input_tensor=new_input)
        checkpoint = ModelCheckpoint("models/model_Xcep.h5", monitor='val_accuracy',
                             save_weights_only=True, mode='max', verbose=1)
        callbacks = [checkpoint]
        model = Xception(
            include_top=True,
            weights=weights,
            input_tensor=new_input,
            input_shape=(224,224,3),
            pooling='avg',
            classes=8,
            classifier_activation="softmax",
         )
        save_model_json = model.to_json()
        with open("models/modelXcep.json", "w") as json_file:
            json_file.write(save_model_json )
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics='accuracy')
        model.fit(aug_train,
                  validation_data=aug_valid,
                  steps_per_epoch = (287654//1024),
                  epochs = epochs,
                  validation_steps = (3999//1024),
                  callbacks = callbacks
                 )
    elif model.lower() == 'resnet50':
        #model = VGG16(include_top=False, input_tensor=new_input)
        checkpoint = ModelCheckpoint("models/model_Resnet50.h5", monitor='val_accuracy',
                             save_weights_only=True, mode='max', verbose=1)
        callbacks =  [checkpoint]
        model = ResNet50(
            include_top=True,
            weights=weights,
            input_tensor=new_input,
            input_shape=(224,224,3),
            pooling='avg',
            classes=8,
            classifier_activation="softmax",
         )
        save_model_json = model.to_json()
        with open("models/modelResN50.json", "w") as json_file:
            json_file.write(save_model_json )
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics='accuracy')
        model.fit(aug_train,
                  validation_data=aug_valid,
                  steps_per_epoch = (287654//1024),
                  epochs = epochs,
                  validation_steps = (3999//1024),
                  callbacks = callbacks
                 )
    elif model.lower() == 'mobilenetv2':
        #model = VGG16(include_top=False, input_tensor=new_input)
        checkpoint = ModelCheckpoint("models/model_Mobnetv2.h5", monitor='val_accuracy',
                             save_weights_only=True, mode='max', verbose=1)
        callbacks = [checkpoint]
        model = MobileNetV2(
            include_top=True,
            weights=weights,
            input_tensor=new_input,
            input_shape=(224,224,3),
            pooling='avg',
            classes=8,
            classifier_activation="softmax",
         )
        save_model_json = model.to_json()
        with open("models/modelMobNet2.json", "w") as json_file:
            json_file.write(save_model_json )
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics='accuracy')
        model.fit(aug_train,
                  validation_data=aug_valid,
                  steps_per_epoch = (287654//1024),
                  epochs = epochs,
                  validation_steps = (3999//1024),
                  callbacks = callbacks
                 )
    elif model.lower() == 'mobilenet':
        #model = VGG16(include_top=False, input_tensor=new_input)
        checkpoint = ModelCheckpoint("models/model_Mobnet.h5", monitor='val_accuracy',
                             save_weights_only=True, mode='max', verbose=1)
        callbacks = [checkpoint]
        model = MobileNet(
            include_top=True,
            weights=weights,
            input_tensor=new_input,
            input_shape=(224,224,3),
            pooling='avg',
            classes=8,
            classifier_activation="softmax",
         )
        save_model_json = model.to_json()
        with open("models/modelMobNet.json", "w") as json_file:
            json_file.write(save_model_json )
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics='accuracy')
        model.fit(aug_train,
                  validation_data=aug_valid,
                  steps_per_epoch = (287654//1024),
                  epochs = epochs,
                  validation_steps = (3999//1024),
                  callbacks = callbacks,
                 )
    elif model.lower() == 'inceptionv3':
        #model = VGG16(include_top=False, input_tensor=new_input)
        checkpoint = ModelCheckpoint("models/model_Incep.h5", monitor='val_accuracy',
                             save_weights_only=True, mode='max', verbose=1)
        callbacks = [ checkpoint]
        model = InceptionV3(
            include_top=True,
            weights=weights,
            input_tensor=new_input,
            input_shape=(224,224,3),
            pooling='avg',
            classes=8,
            classifier_activation="softmax",
         )
        save_model_json = model.to_json()
        with open("models/modelIncep3.json", "w") as json_file:
            json_file.write(save_model_json )
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics='accuracy')
        model.fit(aug_train,
                  validation_data=aug_valid,
                  steps_per_epoch = (287654//1024),
                  epochs = epochs,
                  validation_steps = (3999//1024),
                  callbacks = callbacks,
                 )
    else :
        print('Models you can choose are : VGG16, Resnet50, Xception, InceptionV3, MobileNetV2, MobileNet')