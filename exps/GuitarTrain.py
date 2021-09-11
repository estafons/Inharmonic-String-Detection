
import numpy as np
from pathlib import Path
import glob
import librosa
import os, sys

BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cur_path = Path(BASE_PATH + '/src/InharmonicStringDetection')
sys.path.append(str(cur_path))

from inharmonic_Analysis import *
from Inharmonic_Detector import StringBetas
from constants_parser import Constants

class GuitarSetStringBetas(StringBetas):
    def add_to_list(self, note_instance):
        self.betas_list_array[note_instance.string][note_instance.fret].append(note_instance.beta)
        #self.betas_list_array.remove(0)
    
    def input_instance(self, instance_audio, midi_note, string, constants : Constants):
        fundamental = librosa.midi_to_hz(midi_note)
        ToolBoxObj = ToolBox(compute_partials, compute_inharmonicity, [constants.no_of_partials, fundamental/2, constants], [])
        note_instance = NoteInstance(fundamental, 0, instance_audio, ToolBoxObj, constants.sampling_rate, constants)
        fundamental = note_instance.recompute_fundamental(constants)
        note_instance = NoteInstance(fundamental, 0, instance_audio, ToolBoxObj, constants.sampling_rate, constants) # compute again with recomputed fundamental
        ToolBoxObj = ToolBox(compute_partials, compute_inharmonicity, [constants.no_of_partials, fundamental/2, constants], [])
        note_instance.string = string
        note_instance.fret = midi_note - constants.tuning[note_instance.string]
        return note_instance
    
    def list_to_medians(self):
        for i, r in enumerate(self.betas_list_array):
            for j, l in enumerate(r):
                self.betas_array[i][j] = np.median(l)

def train_GuitarSet(strBetaObj, constants, train_frets = [0]):
    '''train on selected instances from guitarset saved in specified folder. Fundamental 
    frequency is recomputed (since it is not available otherwise) by computing the expected
     fundamental based on midi note and finding the highest peak in a +-10 hz frame'''
    for index, open_midi in enumerate(constants.tuning):
        for train_fret in train_frets:
            midi_train = open_midi + train_fret
            path_to_train_data = str(Path(constants.training_path + str(midi_train) + "/" +str(index) + "/good/*.wav"))
            list_of_names = glob.glob(path_to_train_data)
            for note_name in list_of_names:
                note_audio, _ = librosa.load(note_name, constants.sampling_rate)
                note_instance = strBetaObj.input_instance(note_audio, midi_train, index, constants)
                strBetaObj.add_to_list(note_instance)
    

def GuitarSetTrainWrapper(constants):

    strBetaObj = GuitarSetStringBetas(np.zeros((len(constants.tuning), constants.no_of_frets)), constants)
    train_GuitarSet(strBetaObj, train_frets = constants.train_frets)
    strBetaObj.list_to_medians()
    print(strBetaObj.betas_array)
    return strBetaObj