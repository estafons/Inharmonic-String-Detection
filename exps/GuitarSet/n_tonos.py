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

from GuitarSetTest import load_data, predictTabThesis, 

parser = argparse.ArgumentParser()
parser.add_argument('config_path', type=str)
parser.add_argument('workspace_folder', type=str)
args = parser.parse_args()

try:
    constants = Constants(args.config_path, args.workspace_folder)
except Exception as e:
    print(e)
 


def compute_all_betas(constants : Constants, StrBetaObj):
    """ function that runs tests on the jams files mentioned in the given file 
    and plots the confusion matrixes for both the genetic and inharmonic results."""

    lines = os.listdir(constants.dataset_names_path+'/data/audio')
    for count, name in enumerate(lines):
        track_name = name
        name = name.split('.')[0]
        name = name[:-4] + '.jams'
        print(name, count,'/',len(lines))
        track_instance, annotations = load_data(track_name, name, constants)
        predictTabThesis(track_instance, annotations, constants, StrBetaObj, name)


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

    for s in range(6):
        plt.plot(range(20), median_betas[s,:], label=str(s))

    plt.xticks(range(1,20))
    plt.grid()
    plt.legend()
    plt.show()