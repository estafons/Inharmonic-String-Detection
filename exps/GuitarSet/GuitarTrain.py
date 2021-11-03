'''
Methods for training on the guitarset dataset on isolated note instances can be found in ```GuitarTrain.py```
script. A folder structure as <#midi_note>──<#string> is expected where cropped note instances are stored for 
the specified midi_note and string number (strings are numbered 0,1,2,3,4,5 as E,A,G,D,B,e). 
Running the ```GuitarSetTrainWrapper``` method will print the betas computed and return a ```StringBetas``` 
object where they are stored. Also the user can specify the frets she wishes to train on the ```constants.ini``` 
file at constant ***train_frets***.
'''
import numpy as np
from pathlib import Path
import glob
import librosa
import os, sys

BASE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
cur_path = Path(BASE_PATH + '/src/')
sys.path.append(str(cur_path))

from helper import printProgressBar
from inharmonic_Analysis import *
# from Inharmonic_Detector import StringBetas
from string_betas import StringBetas
from constants_parser import Constants
from playsound import playsound

def train_GuitarSet(strBetaObj, constants, train_frets = [0]):
    '''train on selected instances from guitarset saved in specified folder. Fundamental 
    frequency is recomputed (since it is not available otherwise) by computing the expected
     fundamental based on midi note and finding the highest peak in a +-10 hz frame'''
    print('Training on the specified subset...')
 
    for index, open_midi in enumerate(constants.tuning):
        printProgressBar(index,len(constants.tuning),decimals=0, length=50)
        for train_fret in train_frets:
            # print(train_fret)
            midi_train = open_midi + train_fret
            path_to_train_data = str(Path(constants.training_path + str(midi_train) + "/" +str(index) + "/*.wav"))
            list_of_names = glob.glob(path_to_train_data)
            for note_name in list_of_names:
                note_audio, _ = librosa.load(note_name, constants.sampling_rate)
                note_instance = strBetaObj.input_instance(note_audio, midi_train, index, constants)         
                strBetaObj.add_to_list(note_instance)
    

def GuitarSetTrainWrapper(constants):

    # strBetaObj = GuitarSetStringBetas(np.zeros((len(constants.tuning), constants.no_of_frets)), constants)
    strBetaObj = StringBetas(np.zeros((len(constants.tuning), constants.no_of_frets)), constants)
    train_GuitarSet(strBetaObj, constants, train_frets = constants.train_frets)
    strBetaObj.list_to_medians()
    strBetaObj.set_limits(constants)
    if hasattr(constants, 'verbose') and constants.verbose:
        print()
        print('Beta estimations:')
        print(strBetaObj.betas_array)
        print()
    print('Acceptable range of beta values:', constants.upper_limit, '-', constants.lower_limit)
    print()
    #print(" Median beta values for nFret methods:")

    return strBetaObj