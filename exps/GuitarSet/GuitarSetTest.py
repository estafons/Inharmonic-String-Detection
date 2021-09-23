from posix import listdir
import jams
from pathlib import Path
import matplotlib.pyplot as plt
import itertools
import numpy as np
import configparser
import os, sys
import argparse
from GuitarTrain import GuitarSetTrainWrapper
import threading

BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cur_path = Path(BASE_PATH + '/src/')
sys.path.append(str(cur_path))

from track_class import *
from Inharmonic_Detector import *
from inharmonic_Analysis import *
from constants_parser import Constants
import genetic
from helper import ConfusionMatrix, compute_partial_orders, printProgressBar

from playsound import playsound
import soundfile as sf

import warnings
warnings.filterwarnings("ignore")

#config_path = Path("C:\\Users/stefa/Documents//Inharmonic String Detection/exps/constants.ini")
parser = argparse.ArgumentParser()
parser.add_argument('config_path', type=str)
parser.add_argument('workspace_folder', type=str)
parser.add_argument('-plot', action='store_true') 
parser.add_argument('-run_genetic_alg', action='store_true') 
# parser.add_argument('--dataset', type=str, default= '')
# parser.add_argument('--train_mode', type=str, default= '')

args = parser.parse_args()

try:
    constants = Constants(args.config_path, args.workspace_folder)
except Exception as e:
    print(e)

# HARDWIRE CONSTANTS
constants.plot = args.plot
constants.run_genetic_alg = args.run_genetic_alg
# if args.dataset:
#     constants.dataset = args.dataset
# if args.train_mode:
#     constants.train_mode = args.train_mode


def read_tablature_from_GuitarSet(jam_name, constants):
    """function to read a jam file and return the annotations needed"""
    string = 0
    with open(jam_name) as fp:
        try:
            jam = jams.load(fp)
        except:
            print('failed again!!!!!!!!!')
    tups = []
    annos = jam.search(namespace='note_midi')
    if len(annos) == 0:
        annos = jam.search(namespace='pitch_midi')
    for string_tran in annos:
        for note in string_tran:
            onset = note[0]
            midi_note = note[2]
            fundamental = Inharmonic_Detector.midi_to_hz(midi_note)
            # fret = int(round(midi_note - constants.tuning[string]))
            tups.append((onset, fundamental, string))
        string += 1
    tups.sort(key=lambda x: x[0])
    return Annotations(tups, constants)

def load_data(track_name, annotation_name, constants : Constants):
    """function that loads annotation and audio file and returns instances"""
    track_name = Path(constants.track_path + track_name)
    annotation_name = Path(constants.annos_path + annotation_name)

    data, _ = librosa.load(track_name, constants.sampling_rate) # _ cause dont need to reassign sampling rate
    annotations = read_tablature_from_GuitarSet(annotation_name, constants)
    tups = [(x.onset,x.fundamental, 6) for x in annotations.tablature.tablature]
    # TODO onsets!!
    tablature = Tablature(tups, data, constants)
    track_instance = TrackInstance(tablature, data, constants)
    return track_instance, annotations

def predictTabThesis(track_instance : TrackInstance, annotations : Annotations, constants : Constants, StrBetaObj, filename=None):
    def close_event(): # https://stackoverflow.com/questions/30364770/how-to-set-timeout-to-pyplot-show-in-matplotlib
        plt.close() #timer calls this function after 3 seconds and closes the window 
    def listen_to_the_intance(audio):
        sf.write('tmp.wav', audio, 16000, 'PCM_16')
        playsound('tmp.wav')

    """Inharmonic prediction of tablature as implemented for thesis """
    for tab_instance, annos_instance in zip(track_instance.tablature.tablature, annotations.tablature.tablature):
        # ToolBoxObj = ToolBox(compute_partials_with_order, compute_inharmonicity, [tab_instance.fundamental/2, constants, StrBetaObj], [])
        # TODO: make inharmonicity_compute_func have a meaning, also 1st arg of partial_func_args
        ToolBoxObj = ToolBox(partial_tracking_func=compute_partials, inharmonicity_compute_func=compute_inharmonicity, 
                            partial_func_args=[constants.no_of_partials, tab_instance.fundamental/2, constants, StrBetaObj], inharmonic_func_args=[])
        note_instance = NoteInstance(tab_instance.fundamental, tab_instance.onset, tab_instance.note_audio, ToolBoxObj, track_instance.sampling_rate, constants)
        Inharmonic_Detector.DetectString(note_instance, StrBetaObj, constants.betafunc, constants)
        tab_instance.string = note_instance.string
        if tab_instance.string != 6: # 6 marks inconclusive
            tab_instance.fret = Inharmonic_Detector.hz_to_midi(note_instance.fundamental) - constants.tuning[note_instance.string]
        else:
            tab_instance.fret = None

        # NOTE
        if constants.plot:
            x = threading.Thread(target=listen_to_the_intance, args=(tab_instance.note_audio,))
            x.start()
            fig = plt.figure(figsize=(15, 10))
            timer = fig.canvas.new_timer(interval = 3000) #creating a timer object and setting an interval of 3000 milliseconds
            timer.add_callback(close_event)
            ax1 = fig.add_subplot(2, 1, 1)
            ax2 = fig.add_subplot(2, 1, 2)
            #  TODO: fix lim
            note_instance.plot_partial_deviations(lim=30, res=note_instance.abc, ax=ax1, note_string=str(note_instance.string), annos_string=str(annos_instance.string))#, peaks_idx=Peaks_Idx)
            note_instance.plot_DFT(lim=30, ax=ax2)   
            fig.savefig('imgs/auto_img_test_examples/'+str(note_instance.string)+'_'+str(filename)+'.png')
            timer.start()
            plt.show()

def testGuitarSet(constants : Constants, StrBetaObj):
    """ function that runs tests on the jams files mentioned in the given file 
    and plots the confusion matrixes for both the genetic and inharmonic results."""

    # TODO: check multipitch instances, that is

    InhConfusionMatrixObj = ConfusionMatrix((6,7), inconclusive = True)
    GenConfusionMatrixObj = ConfusionMatrix((6,6), inconclusive = False)
    with open(constants.dataset_names_path + constants.listoftracksfile) as n:
        lines = n.readlines()
    # lines = os.listdir(constants.dataset_names_path)
    for count, name in enumerate(lines):
        # if 'solo' not in name:
        #     pass
        # printProgressBar(count,len(lines),decimals=0, length=50)
        name = name.replace('\n', '')
        print()
        print('Audio-based detection running...')
        print(name, count,'/',len(lines))
        track_name = name[:-5] + '_' + constants.dataset +'.wav'
        track_instance, annotations = load_data(track_name, name, constants)
        predictTabThesis(track_instance, annotations, constants, StrBetaObj, name)
        InhConfusionMatrixObj.add_to_matrix(track_instance.tablature.tablature, annotations)
        _, current_acc = InhConfusionMatrixObj.get_accuracy()
        print('audio_based accuracy: ', current_acc)
        InhConfusionMatrixObj.current_matrix = np.zeros((6,7))
        if constants.run_genetic_alg:
            tab, g = genetic.genetic(track_instance.tablature, constants)
            GenConfusionMatrixObj.add_to_matrix(tab, annotations)
            _, current_acc = GenConfusionMatrixObj.get_accuracy()
            print('GA accuracy: ', current_acc)
            GenConfusionMatrixObj.current_matrix = np.zeros((6,6))
        # if count==0:
        #     break

    InhConfusionMatrixObj.plot_confusion_matrix(constants, normalize= True, 
                                title = str(constants.no_of_partials) + 'Inharmonic Confusion Matrix' +str(round(InhConfusionMatrixObj.get_accuracy()[0],3)))
    print('Audio-based mean accuracy:', round(InhConfusionMatrixObj.get_accuracy()[0]))
    GenConfusionMatrixObj.plot_confusion_matrix(constants, normalize= True, 
                             title = 'Genetic Confusion Matrix'+str(round(GenConfusionMatrixObj.get_accuracy()[0],3)))
    print('GA context-based mean accuracy:', round(GenConfusionMatrixObj.get_accuracy()[0]))



if __name__ == '__main__':
    print('Check if you are OK with certain important configuration constants:')
    print('****************************')
    print('dataset:', constants.dataset)
    print('train_mode:', constants.train_mode)
    print('train_frets:', constants.train_frets)
    print('polyfit:', constants.polyfit)
    print('****************************')
    print()

    StrBetaObj = GuitarSetTrainWrapper(constants)
    # compute_partial_orders(StrBetaObj, constants)
    testGuitarSet(constants, StrBetaObj)