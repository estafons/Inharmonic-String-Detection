
from pathlib import Path
import argparse
import os, sys

BASE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
cur_path = Path(BASE_PATH + '/src/InharmonicStringDetection')
sys.path.append(str(cur_path))
cur_path = Path(BASE_PATH + '/exps')
sys.path.append(str(cur_path))

from GuitarTrain import *

parser = argparse.ArgumentParser()
parser.add_argument('config_path', type=str)
args = parser.parse_args()

#input from user
#config_path = Path("C:\\Users/stefa/Documents//Inharmonic String Detection/exps/constants.ini")
try:
    constants = Constants(args.config_path)
except:
    raise RuntimeError(('could not open ' + str(args.config_path) + ', does not exist or given' +
                 'in wrong format try again as C:\\Users/user/Documents/path_to_config.ini'))

def HjerrildChristensenTrain(strBetaObj, constants : Constants, train_frets = [0]):
    if constants.guitar == 'firebrand':
        dataset_nums = [1,2,3,5,6,7,8,9,10]
    elif constants.guitar == 'martin':
        dataset_nums = [1,2,3,4,5,6,7,8,9,10]
    for dataset_no in dataset_nums:
        for string in range(0,6):
            for fret in constants.train_frets:
                midi = constants.tuning[string] + fret
                path_to_track = Path(constants.path_to_hjerrild_christensen +
                                     constants.guitar + str(dataset_no) + 
                                            '/string' +str(string + 1) +'/' + str(fret) +'.wav')
                audio, _ = librosa.load(path_to_track, constants.sampling_rate)
                y = librosa.onset.onset_detect(audio, constants.sampling_rate)
                audio = audio[y[0]:] # adding this line because not all tracks start at the beggining
                note_instance = strBetaObj.input_instance(audio, midi, string, constants)
                strBetaObj.add_to_list(note_instance)

def TrainWrapper(constants : Constants):
    strBetaObj = GuitarSetStringBetas(np.zeros((len(constants.tuning), constants.no_of_frets)), constants)
    HjerrildChristensenTrain(strBetaObj, constants, train_frets = constants.train_frets)
    strBetaObj.list_to_medians()
    print(strBetaObj.betas_array)
    return strBetaObj

#TrainWrapper(constants)