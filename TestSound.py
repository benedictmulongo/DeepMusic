import IPython
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Activation, Dense, LSTM, Dropout, Flatten
import mido
import numpy as np 
import pandas as pd
import music21
import pickle
import mingus.core.notes as notes
from playsound import playsound
from mido import MidiFile, MidiTrack, Message
from music21 import converter, instrument, note, chord, stream
from pypianoroll import *
from matplotlib import pyplot as plt

def extract_pianoroll(song, index = 1 ) :
    
    files = Multitrack(song)
    track = files.tracks[index]
    track_pianoroll = track.pianoroll
    fig, axs = track.plot()
    plt.show()
    
    return track, track_pianoroll 

def create_dataset(dataset, look_back=1):
  dataX, dataY = [], []
  
  for i in range(len(dataset)-look_back+1):
    a = dataset[i:(i+look_back), :]
    dataX.append(a)
    dataY.append(dataset[i + look_back - 1, :])
    
  return np.array(dataX), np.array(dataY)
 
def create_network(X): 
    """Create the model architecture"""
    model = Sequential()
    print(np.shape(X))
    print(X.shape[1:])
    model.add(LSTM(200, input_shape=X.shape[1:], return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(200, return_sequences=True))
    model.add(Flatten())
    model.add(Dense(X.shape[2]))
    model.compile(loss='mse', optimizer='adam',metrics=['accuracy'])
    # # model.compile(loss='categorical_crossentropy', optimizer='adam')
    # model.compile(loss='categorical_crossentropy', optimizer='rmsprop',metrics=['accuracy'])

    return model

def sample_music(song, ind) :
        
    track, track_pianoroll = extract_pianoroll(song,ind) 
    X,Y = create_dataset(track_pianoroll, 5)
    model = create_network(X)
    history = model.fit(X, Y, validation_split=0.20, epochs=2, batch_size=100)

    test = X[200]
    preds = [] 
    
    for i in range(2):
        
    
        a, b = np.shape(test)
        test = np.reshape(test, (1, a, b))
        
        prediction = model.predict(test, verbose=0)
        preds.append(prediction[0].astype(int).tolist() )
        print("sum ( ", i , " ) : ", np.sum(prediction[0].astype(int)))
    
        test1 = np.concatenate((test[0], prediction))
        test = test1[1:]
    
    preds = np.array(preds)
    track = Track(pianoroll=preds, name='mpiano')
    
    return track
    
song = 'DiamondHead.mid'

track1 = sample_music(song, 0)
track2 = sample_music(song, 1)
multitrack = Multitrack(tracks=[track1, track2])

# playsound(song)



# test = X[200]
# preds = [] 
#     
# print("np.shape(test) : ",np.shape(test) )
# a, b = np.shape(test)
# test = np.reshape(test, (1, a, b))
# prediction = model.predict(test, verbose=0)
# preds = preds.append(prediction.astype(int).tolist() )
# print("Pred shape 0 : ", np.shape(prediction))
# print("Pred shape : ", np.shape(preds))
# # 
# test1 = np.concatenate((test[0], prediction))
# test = test1[1:]