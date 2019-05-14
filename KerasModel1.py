import sys
import re 
import numpy as np 
import pandas as pd
import music21
from glob import glob
import IPython
from tqdm import tqdm
import pickle
from keras.utils import np_utils
import mingus.core.notes as notes
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Activation, Dense, LSTM, Dropout, Flatten
from playsound import playsound
import mido
from mido import MidiFile, MidiTrack, Message
from music21 import converter, instrument, note, chord, stream
from pypianoroll import *
from matplotlib import pyplot as plt


class KerasModel1 :
    
    def __init__(self, path, epochs = 10, sequenceLength= 100, nLayer = 2, sampleLength = 500):
        
        # Path should gives the map of where the midi file is lcoated 
        self.Music_data =  glob( path  + '/*.midi')
        self.path = path
        self.sequenceLength = sequenceLength
        self.nLayer = nLayer
        self.sampleLength = sampleLength
        self.epochs = epochs
        self.all_notes = []
        self.n_vocab = []
        self.note_to_int = []
        self.int_to_note = []
        self.pitchnames = [] 
    
    def convert_midi_to_Notes_chords(self, song):
        print("A")
        notes = []
        # Conversion the Midi file to a stream object
        midi = converter.parse(song)
        notes_to_parse = []
        try:
            # Given a single stream, partition into a part for each unique instrument
            parts = instrument.partitionByInstrument(midi)
        except:
            pass
        if parts: 
            notes_to_parse = parts.parts[0].recurse()
        else:
            notes_to_parse = midi.flat.notes
    
        for element in notes_to_parse: 
            if isinstance(element, note.Note):
                # if element is a note, extract pitch
                notes.append(str(element.pitch))
            elif(isinstance(element, chord.Chord)):
                # if element is a chord, append the normal form of the 
                # chord (a list of integers) to the list of notes. 
                notes.append('.'.join(str(n) for n in element.normalOrder))
    
        return notes
        
    def Data_to_Notes_chords(self) :
        
        Notes = []
        for data in self.Music_data :
            Notes.extend(KerasModel1.convert_midi_to_Notes_chords(self, data))
            
        self.all_notes = Notes
        
        return Notes
        
    def create_dictionary(self, notes) :
        
        # Extract the unique pitches in the list of notes.
        pitchnames = sorted(set(item for item in notes))
        n_vocab = len(pitchnames)
    
        # Create a dictionary to map pitches to integers
        note_to_int = dict((note, number) for number, note in enumerate(pitchnames))
        int_to_note = dict((number, note) for number, note in enumerate(pitchnames))
        
        self.n_vocab = n_vocab
        self.note_to_int = note_to_int
        self.int_to_note = int_to_note
        self.pitchnames = pitchnames
        
        return n_vocab, note_to_int , int_to_note
            
    def prepare_sequences(self, notes): 
        sequence_length = self.sequenceLength
        n_vocab, note_to_int , int_to_note = KerasModel1.create_dictionary(self, notes)
    
        network_input = []
        network_output = []
    
        # create input sequences and the corresponding outputs
        for i in range(0, len(notes) - sequence_length, 1):
            sequence_in = notes[i: i + sequence_length]
            sequence_out = notes[i + sequence_length]
            network_input.append([note_to_int[char] for char in sequence_in])
            network_output.append(note_to_int[sequence_out])
        
        n_patterns = len(network_input)
        
        # Reshape the input into a format compatible with the LSTM layers 
        network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))
        
        # Normalization of the input
        network_input = network_input / float(n_vocab)
        
        # One-hot encodeíng the output vectors
        network_output = np_utils.to_categorical(network_output)
    
        
        return network_input, network_output
        
    def create_network(self, network_in): 
        """
        Create the model architecture
        
        """
        model = Sequential()
        model.add(LSTM(128, input_shape=network_in.shape[1:], return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(128, return_sequences=True))
        model.add(Flatten())
        model.add(Dense(256))
        model.add(Dropout(0.3))
        model.add(Dense(self.n_vocab))
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop',metrics=['accuracy'])
        
        self.models = model
    
        return model
        
    def train(self) :
        
        # Pre processing of data
        notes = KerasModel1.Data_to_Notes_chords(self)
        
        # Data formation
        network_input, network_output = KerasModel1.prepare_sequences(self, notes)
        
        # Model architecture
        model = KerasModel1.create_network(self, network_input)
        
        # Training
        history = model.fit(network_input, network_output, validation_split=0.20, epochs=self.epochs, batch_size=64)
        
        
        # Generation of music
        KerasModel1.notes_sampling(self, notes, model, network_input)
        KerasModel1.plotHistory(history)
        
    def plotHistory(history) :
        plt.figure()
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()
        
        # Plot training & validation loss values
        plt.figure()
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()
        
    def notes_sampling(self, notes, model, network_input):
        """ Generate notes from the neural network based on a sequence of notes """
        # Pick a random integer
        start = np.random.randint(0, len(network_input)-1)
        
        # n_vocab, note_to_int , int_to_note = create_dictionary(notes)
    
        # int_to_note = dict((number, note) for number, note in enumerate(pitchnames))
        # pick a random sequence from the input as a starting point for the prediction
        pattern = network_input[start]
        # print("The pattern", pattern )
        # print("PAttern shape -> ", np.shape(pattern) )
        prediction_output = []
        
    
        # generate #(self.sampleLength) notes
        for note_index in range(self.sampleLength):
            prediction_input = np.reshape(pattern, (1, len(pattern), 1))
            # prediction_input = prediction_input / float(n_vocab)
            
            # print(" prediction_input : ", prediction_input)
    
            prediction = model.predict(prediction_input, verbose=0)
            
            # Predicted output is the argmax(P(h|D))
            index = np.argmax(prediction)
            # print("Pred : ", index )
            # Mapping the predicted interger back to the corresponding note
            result = self.int_to_note[index]
            
            # Storing the predicted output
            prediction_output.append(result)
            
            # Reshape pattern for prediction
            pattern = np.concatenate((pattern, [[index/float(self.n_vocab)]]))
            pattern = pattern[1:len(pattern)+1]

    
        print('Notes Generated...')
        KerasModel1.create_midi(self, prediction_output)
        
        return prediction_output
        
    def create_midi(self, prediction_output):
        """ convert the output from the prediction to notes and create a midi file
            from the notes """
        offset = 0
        output_notes = []
    
        # create note and chord objects based on the values generated by the model
        for pattern in prediction_output:
            # pattern is a chord
            if ('.' in pattern) or pattern.isdigit():
                notes_in_chord = pattern.split('.')
                notes = []
                for current_note in notes_in_chord:
                    new_note = note.Note(int(current_note))
                    # new_note.storedInstrument = instrument.Piano()
                    # new_note.storedInstrument = instrument.Saxophone()
                    new_note.storedInstrument = instrument.Trumpet()
                    notes.append(new_note)
                new_chord = chord.Chord(notes)
                new_chord.offset = offset
                output_notes.append(new_chord)
            # pattern is a note
            else:
                new_note = note.Note(pattern)
                new_note.offset = offset
                # new_note.storedInstrument = instrument.Piano()
                # new_note.storedInstrument = instrument.Saxophone()
                # new_note.storedInstrument = instrument.SopranoSaxophone()
                new_note.storedInstrument = instrument.Trumpet()
                output_notes.append(new_note)
    
            # increase offset each iteration so that notes do not stack
            offset += 0.5
    
        midi_stream = stream.Stream(output_notes)
        
        print('Saving Output file as midi....')
    
        midi_stream.write('midi', fp='test_output5.mid')

# Model1 = KerasModel1(path = 'midi1')
Model1 = KerasModel1(path = 'Maestro2017')
s = Model1.Data_to_Notes_chords() 
print(len(s))
print("S : ", s)
Model1.train()
