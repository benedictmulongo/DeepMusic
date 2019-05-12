import sys
import re 
import numpy as np 
import pandas as pd
import music21
import pickle
import mingus.core.notes as notes
from playsound import playsound
import mido
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
  
  

song = 'DiamondHead.mid'
# playsound(song)