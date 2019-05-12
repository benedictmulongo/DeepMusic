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
###-----------------------------------------------------------
from plot import plot_history
###-----------------------------------------------------------

songs = glob('Jazz/*.mid')
songs = songs[:3]

def element_of_Midifile(song):
    
    
    midi = converter.parse(song)
    print(midi)
    parts = instrument.partitionByInstrument(midi)
    print(parts)
    notes_to_parse = parts.parts[0].recurse()
    print(notes_to_parse)
    print()
    print("--------------------------------------------")
    for element in notes_to_parse: 
        print(element)
        
    # notes_to_parse.show()
    
    print("--------------------------------------------")
    
def convert_midi_to_Notes_chords(song):
    
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


def create_dictionary(notes) :
    
    # Extract the unique pitches in the list of notes.
    pitchnames = sorted(set(item for item in notes))
    n_vocab = len(pitchnames)

    # Create a dictionary to map pitches to integers
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))
    int_to_note = dict((number, note) for number, note in enumerate(pitchnames))
    
    return n_vocab, note_to_int , int_to_note
    
    
def prepare_sequences(notes): 
    sequence_length = 100
    n_vocab, note_to_int , int_to_note = create_dictionary(notes)

    network_input = []
    network_output = []

    # create input sequences and the corresponding outputs
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i: i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])
        network_output.append(note_to_int[sequence_out])
    
    n_patterns = len(network_input)
    
    # Reshape the input into a format comatible with LSTM layers 
    network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))
    
    # Normalization of the input
    network_input = network_input / float(n_vocab)
    
    # one hot encode the output vectors
    network_output = np_utils.to_categorical(network_output)

    
    return network_input, network_output
   
def create_network(network_in, n_vocab): 
    """Create the model architecture"""
    model = Sequential()
    model.add(LSTM(128, input_shape=network_in.shape[1:], return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(128, return_sequences=True))
    model.add(Flatten())
    model.add(Dense(256))
    model.add(Dropout(0.3))
    model.add(Dense(n_vocab))
    model.add(Activation('softmax'))
    # model.compile(loss='categorical_crossentropy', optimizer='adam')
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop',metrics=['accuracy'])

    return model
    
    
# def create_network(network_input, n_vocab):
#     """ create the structure of the neural network """
#     model = Sequential()
#     model.add(LSTM(
#         512,
#         input_shape=(network_input.shape[1], network_input.shape[2]),
#         return_sequences=True
#     ))
#     model.add(Dropout(0.3))
#     model.add(LSTM(512, return_sequences=True))
#     model.add(Dropout(0.3))
#     model.add(LSTM(512))
#     model.add(Dense(256))
#     model.add(Dropout(0.3))
#     model.add(Dense(n_vocab))
#     model.add(Activation('softmax'))
#     model.compile(loss='categorical_crossentropy', 
#                   optimizer='rmsprop', 
#                   metrics=['accuracy'])
# 
# return model


def train(thesong) :
    
    # Pre processing of data
    notes = convert_midi_to_Notes_chords(thesong)
    
    # Data formation
    
    network_input, network_output = prepare_sequences(notes)
    # Model architecture
    model = create_network(network_input, n_vocab)
    # Treaining
    filepath = "weights-{epoch:02d}-{loss:.4f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=0,save_best_only=True,mode='min')
    callbacks_list = [checkpoint]
    # history = model.fit(network_input, network_output, epochs=20, batch_size=64, callbacks=callbacks_list)
    
    history = model.fit(network_input, network_output, validation_split=0.20, epochs=100, batch_size=64, callbacks=callbacks_list)
    # history = model.fit(x, y, validation_split=0.25, epochs=50, batch_size=16, verbose=1)
    
    # Ploting
    plot_history(history)  
    
    # Generation of music
    generate_notes(notes, model, network_input)
    

def generate_notes(notes, model, network_input):
    """ Generate notes from the neural network based on a sequence of notes """
    # Pick a random integer
    start = np.random.randint(0, len(network_input)-1)
    
    n_vocab, note_to_int , int_to_note = create_dictionary(notes)

    # int_to_note = dict((number, note) for number, note in enumerate(pitchnames))
    # pick a random sequence from the input as a starting point for the prediction
    pattern = network_input[start]
    # print("The pattern", pattern )
    # print("PAttern shape -> ", np.shape(pattern) )
    prediction_output = []
    
    print('Generating notes........')

    # generate 500 notes
    for note_index in range(100):
        prediction_input = np.reshape(pattern, (1, len(pattern), 1))
        # prediction_input = prediction_input / float(n_vocab)
        
        # print(" prediction_input : ", prediction_input)

        prediction = model.predict(prediction_input, verbose=0)
        
        # Predicted output is the argmax(P(h|D))
        index = np.argmax(prediction)
        print("Pred : ", index )
        # Mapping the predicted interger back to the corresponding note
        result = int_to_note[index]
        # Storing the predicted output
        prediction_output.append(result)

        # pattern.append(index)
        # np.append(pattern, index)
        pattern = np.concatenate((pattern, [[index/float(n_vocab)]]))
        # Next input to the model
        # print("PAttern shape -> ", np.shape(pattern) )
        pattern = pattern[1:len(pattern)+1]
        # print("index : ", note_index ,"PAttern shape -> ", np.shape(pattern) )

    print('Notes Generated...')
    create_midi(prediction_output)
    
    return prediction_output
    
def create_midi(prediction_output):
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

    midi_stream.write('midi', fp='test_output4.mid')

song = 'Berckman-Berckman.mid'
# playsound(song)
element_of_Midifile(song)
print()
print()
print(convert_midi_to_Notes_chords(song))
print()
print()

notes =  convert_midi_to_Notes_chords(song)
pitchnames = sorted(set(item for item in notes))
note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

print()
print(pitchnames)
print()
print(note_to_int)

n_vocab = len(set(notes))
print('N_Vocab : ', n_vocab) 
print('N_Vocab 2 : ', len(pitchnames))

prepare_sequences(notes)
train('Berckman-Berckman.mid')


# >>> np.shape(ultros.tracks[0].pianoroll)
# (6168, 128)