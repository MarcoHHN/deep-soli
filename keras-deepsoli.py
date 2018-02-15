# -*- coding: utf-8 -*-
"""
Keras/Tensorflow Implementation of the Deep-Soli end-to-end model
https://ait.ethz.ch/projects/2016/deep-soli/
https://github.com/simonwsw/deep-soli

SGDM: 86,91% per frame accuracy (like in the paper)
ADAM: 92,14% per frame accuracy (used ADAM optimizer, batch_size=64)

"""

import keras
from keras.models import Sequential
from keras.layers import Conv2D, LSTM
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers.wrappers import TimeDistributed
from keras.layers.normalization import BatchNormalization
from keras.utils import to_categorical
from keras.callbacks import LearningRateScheduler
from keras import backend as K
import os
import json
import h5py
import numpy as np
import math
import matplotlib.pyplot as plt

dir_path = 'C:/Users/User/Downloads/deep-soli'
os.chdir(dir_path)

#Parameters
initial_learnrate = 1e-3
momentum = 0.9
batch_size = 64
epochs = 50
timesteps = 40 # sequence length
im_height, im_width, channels = 32, 32, 1
n_classes = 11 # 11 Gesten

#optimizer=keras.optimizers.SGD(lr=initial_learnrate, momentum=momentum)
optimizer=keras.optimizers.Adam(lr=initial_learnrate)

#%% Model mit TimeDistributed() wrapper
# This wrapper applies a layer to every temporal slice of an input.
# You can then use TimeDistributed to apply a layer to each of the timesteps, independently:
# input:0' shape=(None, None, 32, 32, 1)
#        batch_size, timesteps, height, width, rgb-channels
#                    timesteps=None for variable length
model = Sequential()
model.add(TimeDistributed(Conv2D(32, (3, 3), strides=2, padding='valid'), input_shape=(None, im_height, im_width, channels)))
model.add(TimeDistributed(BatchNormalization()))
model.add(TimeDistributed(Activation('relu')))
model.add(TimeDistributed(Conv2D(64, (3, 3), strides=2, padding='valid')))
model.add(TimeDistributed(BatchNormalization()))
model.add(TimeDistributed(Activation('relu')))
model.add(TimeDistributed(Dropout(0.4)))
model.add(TimeDistributed(Conv2D(128, (3, 3), strides=2, padding='valid')))
model.add(TimeDistributed(BatchNormalization()))
model.add(TimeDistributed(Activation('relu')))
model.add(TimeDistributed(Dropout(0.4)))
model.add(TimeDistributed(Flatten()))  # this converts our 3D feature maps to 1D feature vectors
model.add(TimeDistributed(Dense(512)))
model.add(TimeDistributed(BatchNormalization()))
model.add(TimeDistributed(Activation('relu')))
model.add(TimeDistributed(Dropout(0.5)))
model.add(TimeDistributed(Dense(512)))
model.add(LSTM(512, return_sequences=True, dropout=0.0, recurrent_dropout=0.0))
model.add(TimeDistributed(Dropout(0.5)))
model.add(TimeDistributed(Dense(n_classes)))
model.add(TimeDistributed(Activation('softmax')))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=optimizer,
              metrics=['accuracy'])
model.summary()

#%% Load soli data with subsampling/adding frames, load all channels
#json_data=open('config/file_half.json').read()
# Filename:     [gesture ID]_[session ID]_[instance ID].h5
#                11 Gesten  10 Probanden    je 25 mal

dataset = json.load(open('config/file_half.json'))
train_list = dataset["train"]
test_list = dataset["eval"]

train_samples = np.size(train_list)
test_samples = np.size(test_list)

use_channel = [0, 1, 2, 3]

train_data_all = []
train_label_all = []
test_data_all = []
test_label_all = []

for ch in use_channel:
    #Trainset
    train_data= np.zeros((train_samples,timesteps,im_height,im_width,channels))
    train_label = np.zeros((train_samples,timesteps,n_classes))
    
    for i, file_id in enumerate(train_list):
        filename = 'dsp/%s.h5' %file_id
        with h5py.File(filename, 'r') as f:
        # Data and label are numpy arrays
            data = f['ch{}'.format(ch)][()]
            label = f['label'][()]
        num_frames = np.size(data,0)
        
        #Reshape Range-Doppler maps
        rds = np.zeros([num_frames,im_height,im_width,channels])
        for j in range(num_frames): 
            rd = data[j,:]
            rd = np.reshape(rd,(im_height,im_width,channels))
            rds[j,:,:,:] = rd
        
        #subsample/add num_frames to be length 40 (skip some frames or repeat some)
        subsample_steps = np.linspace(0,num_frames-1,timesteps,dtype='int')
        rds_subsample = np.zeros([timesteps,32,32,1])
        for idx, rd_nr in enumerate(subsample_steps):
            rds_subsample[idx,:,:,:] = rds[rd_nr,:,:,:] 
        
        # add Range-Doppler sequence to dataset
        train_data[i,:,:,:,:] = rds_subsample
        
        #labels
        label_categorical = to_categorical(label, num_classes=n_classes)
        label_categorical = np.vstack((label_categorical,label_categorical)) #verdopplen, damit ausreichend lang
        train_label[i,:,:] = label_categorical[0:timesteps]
    
    train_data_all.append(train_data)
    train_label_all.append(train_label)
    
    #Testset ----------------------------------------------------------------------
    test_data= np.zeros((test_samples,timesteps,im_height,im_width,channels))
    test_label = np.zeros((test_samples,timesteps,n_classes))
    
    for i, file_id in enumerate(test_list):
        filename = 'dsp/%s.h5' %file_id
        with h5py.File(filename, 'r') as f:
        # Data and label are numpy arrays
            data = f['ch{}'.format(ch)][()]
            label = f['label'][()]
        num_frames = np.size(data,0)
            
        #Reshape Range-Doppler maps
        rds = np.zeros([num_frames,im_height,im_width,channels])
        for j in range(num_frames): 
            rd = data[j,:]
            rd = np.reshape(rd,(im_height,im_width,channels))
            rds[j,:,:,:] = rd
            
        #subsample/add num_frames to be length 40 (skip some frames or repeat some)
        subsample_steps = np.linspace(0,num_frames-1,timesteps,dtype='int')
        rds_subsample = np.zeros([timesteps,32,32,1])
        for idx, rd_nr in enumerate(subsample_steps):
            rds_subsample[idx,:,:,:] = rds[rd_nr,:,:,:] 
            
        # add Range-Doppler sequence to dataset
        test_data[i,:,:,:,:] = rds_subsample
        
        #labels
        label_categorical = to_categorical(label, num_classes=n_classes)
        label_categorical = np.vstack((label_categorical,label_categorical)) #verdopplen, damit ausreichend lang
        test_label[i,:,:] = label_categorical[0:timesteps] 
        
    test_data_all.append(train_data)
    test_label_all.append(train_label)

#concatenate lists
train_data = np.concatenate((train_data_all[0], train_data_all[1],train_data_all[2], train_data_all[3]),axis=0)
train_label = np.concatenate((train_label_all[0], train_label_all[1],train_label_all[2], train_label_all[3]),axis=0)
test_data = np.concatenate((test_data_all[0], test_data_all[1],test_data_all[2], test_data_all[3]),axis=0)
test_label = np.concatenate((test_label_all[0], test_label_all[1],test_label_all[2], test_label_all[3]),axis=0)

print('Data loaded.')

#%% Training callbacks

def step_decay(epoch):
   initial_lrate = initial_learnrate
   drop = 0.1
   epochs_drop = 20.0
   lrate = initial_lrate * math.pow(drop,  
           math.floor((1+epoch)/epochs_drop))
   return lrate

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
       self.losses = []
       self.lr = []
 
    def on_epoch_end(self, batch, logs={}):
       self.losses.append(logs.get('loss'))
       self.lr.append(step_decay(len(self.losses)))
       
      
loss_history = LossHistory()
lrate = LearningRateScheduler(step_decay)
callbacks_list = [loss_history, lrate]

#%% Training

history = model.fit(train_data, train_label,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(test_data, test_label),
          callbacks=callbacks_list,
          shuffle=True)

model.save_weights('temp.h5')  # save weights after training

#%% Evaluate

#model.load_weights('temp.h5')
score = model.evaluate(test_data, test_label, batch_size=256)
print('Loss:', score[0])
print('Accuracy:', score[1])

#%% Load soli data (one file) and predict
# Filename:     [gesture ID]_[session ID]_[instance ID].h5
#                11 Gesten  10 Probanden    je 25 mal

filename = 'dsp/7_0_18.h5'
use_channel = 0
with h5py.File(filename, 'r') as f:
    # Data and label are numpy arrays
    data = f['ch{}'.format(use_channel)][()]
    label = f['label'][()]
num_frames = np.size(data,0)
#rds = []
rds = np.zeros([num_frames,32,32,1])
for i in range(num_frames): 
    rd = data[i,:]
    rd = np.reshape(rd,(32,32,1))
    #rds.append([rd])
    rds[i,:,:,:] = rd
    
x = np.zeros((1,num_frames,im_height,im_width,channels))
x[0,:,:,:,:] = rds
classes = model.predict(x)

#%% Plots

#plot accuracy
fig = plt.figure()
plt.plot(range(1,epochs+1),history.history['val_acc'],label='validation')
plt.plot(range(1,epochs+1),history.history['acc'],label='training')
plt.legend(loc=0)
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.xlim([1,epochs])
plt.ylim([0,1])
plt.grid(True)
plt.title("Model Accuracy")
plt.show()
#fig.savefig('accuracy.jpg')
#plt.close(fig)

# plot loss
fig = plt.figure()
plt.plot(range(1,epochs+1),history.history['val_loss'],label='validation')
plt.plot(range(1,epochs+1),history.history['loss'],label='training')
plt.legend(loc=0)
plt.xlabel('epochs')
plt.ylabel('loss')
plt.xlim([1,epochs])
#plt.ylim([0,1])
plt.grid(True)
plt.title("Model Loss")
plt.show()

# plot learning rate
fig = plt.figure()
plt.plot(range(1,epochs+1),loss_history.lr,label='learning rate')
plt.xlabel("epoch")
plt.xlim([1,epochs+1])
plt.ylabel("learning rate")
plt.legend(loc=0)
plt.grid(True)
plt.title("Learning rate")
plt.show()