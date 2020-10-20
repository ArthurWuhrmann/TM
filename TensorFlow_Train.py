import glob
import pickle
import numpy
from music21 import converter, instrument, note, chord
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Activation
from keras.layers import BatchNormalization as BatchNorm
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
import os


sequences_length = 100
foldername = "midi_songs"
notes = []

dic = {}


#charge les fichiers midi dans le dossier 'foldername' et renvoie une liste de notes
def load_midis(foldername):

    notes = []

    for file in glob.glob("jazz_midis/*.mid"):
        midi = converter.parse(file)

        notes_to_parse = None

        try: # file has instrument parts
            s2 = instrument.partitionByInstrument(midi)
            notes_to_parse = s2.parts[0].recurse() 
        except: # file has notes in a flat structure
            notes_to_parse = midi.flat.notes

        notes_to_p = []
        for index, elem in enumerate(notes_to_parse):
            if not isinstance(elem, note.Note) and not isinstance(elem, chord.Chord):
                pass
            else:
                notes_to_p.append(elem)
        notes_to_parse = notes_to_p

        lastOffset = 0
        for index, element in enumerate(notes_to_parse):
            if isinstance(element, note.Note):
                if element.offset == notes_to_parse[(index + 1)%len(notes_to_parse)].offset:
                    theChord = [element]
                    theOffset = element.offset
                    i = 1
                    while(theOffset == notes_to_parse[(index + i)%len(notes_to_parse)].offset):
                        theChord.append(notes_to_parse[(index + i)%len(notes_to_parse)])
                        notes_to_parse.pop((index + i)%len(notes_to_parse))
                        i+=1
                    litteral_Chord = ""
                    for k in theChord:
                        if isinstance(k, note.Note):
                            litteral_Chord += str(k.pitch) + "."
                        elif isinstance(k, chord.Chord):
                            litteral_Chord += '.'.join(str(n.pitch) for n in k)
                    if litteral_Chord[:-1] == '.':
                        litteral_Chord = litteral_Chord[:-1]
                else:
                    notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.pitches))
            
            # if element.offset - lastOffset > .5:
            #     notes.append(' ')

            if element.offset == lastOffset and ' ' not in notes[-2]:
                lastn = notes.pop()
                notes[-1] = notes[-1] + '.' + lastn

            lastOffset = element.offset

        print("Fichier %s déchiffré" % file)

        for i in range(3):
            notes.append(' ')
    print(os.listdir(), os.getcwd())

    with open('datas/notes', 'wb') as filepath:
         pickle.dump(notes, filepath)


    print()
    print("\t-----")
    print("   Chargement de fichier terminé")
    print("\t-----")
    print()

    return notes
    
#Renvoie le dictionnaire (qui marche dans les deux sens)
def get_dic(notes):
    dic = {}
    for index, pitch in enumerate(sorted(set(pitch for pitch in notes))):
        dic[index] = pitch
        dic[pitch] = index
    return dic

#Partitionne les données
def get_parts(notes, dic, len_vocab):

    inputs, targets = [], []

    for i in range(len(notes) - sequences_length):
        input_ = notes[i:i+sequences_length] # de i (compris) à i+n (non compris)
        target = notes[i + sequences_length] # seulement i+n

        inputs.append([dic[item] for item in input_])
        targets.append(dic[target])

    inputs = numpy.reshape(inputs, (len(inputs), sequences_length, 1))
    #On rajoute juste une troisième dimension égale à 1 (la forme ne change pas réellement,
    #c'est pour le rendre compatible avec le réseau...
    
    inputs = inputs / float(len_vocab) #Normalisation

    targets = np_utils.to_categorical(targets) #A checker

    return inputs, targets

def build_model(inputs, len_vocab):

    model = Sequential()
    model.add(LSTM(
        300,
        input_shape=(inputs.shape[1], inputs.shape[2]),
        return_sequences=False))
    model.add(Dropout(.5))
    model.add(Activation('relu'))
    model.add(BatchNorm())
    model.add(Dropout(.5))
    model.add(Dense(len_vocab))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    return model

def get_items():


    notes = load_midis(foldername)
    
    dic = get_dic(notes)
    inputs, targets = get_parts(notes, dic, len(set(notes)))

    return notes, dic, inputs, targets, len(set(notes)), len(inputs)


def train_model():

    _, dic, inputs, targets, len_vocab, n_sequences = get_items()

    print("inputs", len(inputs), len(inputs[0]))
    print("outputs", len(targets), targets.shape)


    model = build_model(inputs, len_vocab)

    filepath = "weigths-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"

    checkpoint = ModelCheckpoint(
        filepath,
        monitor="loss",
        verbose=0,
        save_best_only=True,
        mode='min')
    callbacks_list = [checkpoint]

    model.fit(inputs, targets, epochs=1000, batch_size=128, callbacks=callbacks_list)

if __name__ == "__main__":
    train_model()
    
