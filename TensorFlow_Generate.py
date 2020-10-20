import glob
import pickle
import numpy as np
from music21 import converter, instrument, note, chord, stream
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Activation
from keras.layers import BatchNormalization as BatchNorm
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
import tf_train

sequences_length = tf_train.sequences_length

prediction_length = 500

notes, dic, inputs, targets, len_vocab, n_sequences = tf_train.get_items(False)

print(notes[:50])

model = tf_train.build_model(inputs, len_vocab)
model.load_weights("weigths-improvement-498-1.6059-bigger.hdf5")

pred_inputs = np.reshape(inputs, (inputs.shape[0], inputs.shape[1]))
pred_inputs = pred_inputs.tolist()

rdm = np.random.randint(0, len(pred_inputs)-1)
rdm_song = pred_inputs[rdm]
predictions = []

for note_index in range(prediction_length):

    pred_in = np.reshape(rdm_song, (1, len(rdm_song), 1))
    pred_in = pred_in / float(len_vocab)

    pred = model.predict(pred_in) #Prédit la prochaine note

    pred_note = dic[np.argmax(pred)]
    predictions.append(pred_note)

    rdm_song.append(np.argmax(pred)) #On ajoute la note devinée
    rdm_song = rdm_song[1:len(rdm_song)] #On prend du 2eme élément pour qu'il ait la même taille

print(predictions[:30])


offset = 0
out_notes = []


for item in predictions:

    if('.' in item) or item.isdigit():

        notes_in_chord = item.split(".")
        notes = []

        for note_ in notes_in_chord:
            new_note = note.Note(note_)
            new_note.storedInstrument = instrument.Piano()
            notes.append(new_note)
        new_chord = chord.Chord(notes)
        new_chord.offset = offset

        out_notes.append(new_chord)
    elif item == ' ':
            out_notes.append(note.Rest(.5))
    else:
        new_note = note.Note(item)
        new_note.offset = offset
        new_note.storedInstrument = instrument.Piano()
        out_notes.append(new_note)

    offset += .5


midi_stream = stream.Stream(out_notes)

midi_stream.write('midi', fp='out.mid')