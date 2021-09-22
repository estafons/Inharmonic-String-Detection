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

import statistics

import warnings
warnings.filterwarnings("ignore")

from GuitarSetTest import read_tablature_from_GuitarSet

parser = argparse.ArgumentParser()
parser.add_argument('config_path', type=str)
parser.add_argument('workspace_folder', type=str)
parser.add_argument('-plot', action='store_true') 
parser.add_argument('-run_genetic_alg', action='store_true') 

args = parser.parse_args()

try:
    constants = Constants(args.config_path, args.workspace_folder)
except Exception as e:
    print(e)
 

# HARDWIRE CONSTANTS
constants.plot = args.plot
constants.run_genetic_alg = args.run_genetic_alg


def load_debleeded_channel_data(track_name, annotation_name, constants : Constants):
    """function that loads annotation and audio file and returns instances"""
    track_name = Path(constants.track_path + track_name)
    annotation_name = Path(constants.annos_path + annotation_name)

    multi_channel_data, _ = librosa.core.load(track_name, constants.sampling_rate, mono=False) # _ cause dont need to reassign sampling rate
    data = multi_channel_data#TODO:[channel,:]

    annotations = read_tablature_from_GuitarSet(annotation_name, constants)
    tups = [(x.onset, x.fundamental, 6) for x in annotations.tablature.tablature]
    # TODO: onsets!!
    tablature = Tablature(tups, data, constants)
    track_instance = TrackInstance(tablature, data, constants)
    return track_instance, annotations


def compute_track_betas(track_instance : TrackInstance, annotations : Annotations, constants : Constants, StrBetaObj, filename=None):
    global betas
    global median_betas
    """Inharmonic prediction of tablature as implemented for thesis """
    for tab_instance, annos_instance in zip(track_instance.tablature.tablature, annotations.tablature.tablature):
        ToolBoxObj = ToolBox(partial_tracking_func=compute_partials, inharmonicity_compute_func=compute_inharmonicity, partial_func_args=[constants.no_of_partials, tab_instance.fundamental/2, constants, StrBetaObj], inharmonic_func_args=[])
        note_instance = NoteInstance(tab_instance.fundamental, tab_instance.onset, tab_instance.note_audio, ToolBoxObj, track_instance.sampling_rate, constants)
        Inharmonic_Detector.DetectString(note_instance, StrBetaObj, constants.betafunc, constants)
        tab_instance.string = note_instance.string
        if note_instance.string!= 6:
            tab_instance.fret = Inharmonic_Detector.hz_to_midi(note_instance.fundamental) - constants.tuning[note_instance.string]
            betas[note_instance.string,tab_instance.fret].append(note_instance.beta)   

def compute_all_betas(constants : Constants, StrBetaObj):
    def listen_to_the_intance(audio):
        sf.write('tmp.wav', audio, 16000, 'PCM_16')
        playsound('tmp.wav')    
    """ function that runs tests on the jams files mentioned in the given file 
    and plots the confusion matrixes for both the genetic and inharmonic results."""

    lines = os.listdir(constants.dataset_names_path+'/data/audio')
    for count, name in enumerate(lines):
        if '_hex' not in name:
            continue        
        print(name)
        track_name = name
        name = name.split('.')[0]
        name = name[:-8] + '.jams'
        print(name, count,'/',len(lines))
        track_instance, annotations = load_debleeded_channel_data(track_name, name, constants)
        compute_track_betas(track_instance, annotations, constants, StrBetaObj, name)
        # if count==0:
        #     break
        if constants.plot:
            listen_to_the_intance(tab_instance.note_audio)
            fig = plt.figure(figsize=(15, 10))
            ax1 = fig.add_subplot(2, 1, 1)
            ax2 = fig.add_subplot(2, 1, 2)
            #  TODO: fix lim
            note_instance.plot_partial_deviations(lim=30, res=note_instance.abc, ax=ax1, note_string=str(note_instance.string), annos_string=str(annos_instance.string))#, peaks_idx=Peaks_Idx)
            note_instance.plot_DFT(lim=30, ax=ax2)   
            plt.show()
            plt.close()

if __name__ == '__main__':
    print('Check if you are OK with certain important configuration constants:')
    print('****************************')
    print('dataset:', constants.dataset)
    print('train_mode:', constants.train_mode)
    print('train_frets:', constants.train_frets)
    print('polyfit:', constants.polyfit)
    print('dataset_names_path:', constants.dataset_names_path)
    print('****************************')
    print()

    betas = np.array([[None]*20]*6)
    for s in range(6):
        for n in range(20):
            betas[s,n] = []

    median_betas = np.array([[None]*20]*6)

    StrBetaObj = GuitarSetTrainWrapper(constants)
    # compute_partial_orders(StrBetaObj, constants)
    compute_all_betas(constants, StrBetaObj)

    for s in range(6):
        for n in range(20):
            if betas[s,n]: # not None
                median_betas[s,n] = statistics.median(betas[s,n])

    np.savez("./results/n_tonos.npz", median_betas=median_betas)
    # npzfile = np.load('./results/n_tonos.npz', allow_pickle=True) # allow_pickle is needed because dtype=object, since None elements exist.

    for s in range(6):
        plt.plot(range(20), median_betas[s,:], label=str(s))

    plt.xticks(range(1,20))
    plt.grid()
    plt.legend()
    plt.savefig('./results/n_tonos.png')
    plt.show()