
import numpy as np
from pathlib import Path
import glob
import librosa

from inharmonic_Analysis import *
import constants

class StringBetas():
    def __init__(self, barray):
        self.betas_array = barray #0->string 1->fret
        self.betas_list_array = [[[] for x in range(0,constants.no_of_frets)] for i in range(0,len(constants.tuning))]
        
    
    def add_to_list(self, note_instance):
        self.betas_list_array[note_instance.string][note_instance.fret].append(note_instance.beta)
        #self.betas_list_array.remove(0)
    
    def input_instance(self, instance_audio, midi_note, string):
        fundamental = librosa.midi_to_hz(midi_note)
        ToolBoxObj = ToolBox(compute_partials, compute_inharmonicity, [constants.NO_OF_PARTIALS, fundamental/2], [])
        note_instance = NoteInstance(fundamental, 0, instance_audio, ToolBoxObj, constants.sampling_rate)
        fundamental = note_instance.recompute_fundamental()
        note_instance = NoteInstance(fundamental, 0, instance_audio, ToolBoxObj, constants.sampling_rate) # compute again with recomputed fundamental
        ToolBoxObj = ToolBox(compute_partials, compute_inharmonicity, [constants.NO_OF_PARTIALS, fundamental/2], [])
        note_instance.string = string
        note_instance.fret = midi_note - constants.tuning[note_instance.string]
        return note_instance
    
    def list_to_medians(self):
        for i, r in enumerate(self.betas_list_array):
            for j, l in enumerate(r):
                self.betas_array[i][j] = np.median(l)

def train_GuitarSet(strBetaObj, train_frets = [0]):
    '''train on selected instances from guitarset saved in specified folder. Fundamental 
    frequency is recomputed (since it is not available otherwise) by computing the expected
     fundamental based on midi note and finding the highest peak in a +-10 hz frame'''
    for index, open_midi in enumerate(constants.tuning):
        for train_fret in train_frets:
            midi_train = open_midi + train_fret
            path_to_train_data = str(Path(constants.TRAINING_PATH + str(midi_train) + "/" +str(index) + "/good/*.wav"))
            list_of_names = glob.glob(path_to_train_data)
            for note_name in list_of_names:
                note_audio, _ = librosa.load(note_name, constants.sampling_rate)
                note_instance = strBetaObj.input_instance(note_audio, midi_train, index)
                strBetaObj.add_to_list(note_instance)
    

def GuitarSetTrainWrapper():

    strBetaObj = StringBetas(np.zeros((len(constants.tuning), constants.no_of_frets)))
    train_GuitarSet(strBetaObj, train_frets = constants.TRAIN_FRETS)
    strBetaObj.list_to_medians()
    print(strBetaObj.betas_array)
    return strBetaObj
GuitarSetTrainWrapper()