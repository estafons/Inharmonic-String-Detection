from pathlib import Path
import os, sys
import librosa
import argparse



BASE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
cur_path = Path(BASE_PATH + '/src/InharmonicStringDetection')
sys.path.append(str(cur_path))
cur_path = Path(BASE_PATH + '/exps')
sys.path.append(str(cur_path))

from track_class import *
from helper import ConfusionMatrix, compute_partial_orders
from HjerrildTrain import TrainWrapper
from inharmonic_Analysis import (compute_partials, compute_inharmonicity, NoteInstance, 
                                    ToolBox, compute_partials_with_order, 
                                        compute_partials_with_order_strict)
import Inharmonic_Detector

parser = argparse.ArgumentParser()
parser.add_argument('config_path', type=str)
parser.add_argument('workspace_folder', type=str)
args = parser.parse_args()

#input from user
#config_path = Path("C:\\Users/stefa/Documents//Inharmonic String Detection/InharmonicStringDetection/constants.ini")
try:
    constants = Constants(args.config_path, args.workspace_folder)
except Exception as e:
    print(e)

def testHjerrildChristensen(constants : Constants, StrBetaObj):
    InhConfusionMatrixObj = ConfusionMatrix((6,7), inconclusive = True)
    if constants.guitar == 'firebrand':
        dataset_nums = [1,2,3,5,6,7,8,9,10]
    elif constants.guitar == 'martin':
        dataset_nums = [1,2,3,4,5,6,7,8,9,10]
    print("Testing to the training set complement...")
    for dataset_no in dataset_nums:
        print(dataset_no)
        for string in range(0,6):
            for fret in range(0,12):
                path_to_track = Path(constants.path_to_hjerrild_christensen +
                                     constants.guitar + str(dataset_no) + 
                                            '/string' +str(string + 1) +'/' + str(fret) +'.wav')
                audio, _ = librosa.load(path_to_track, constants.sampling_rate)

                # Onset detection (instances not always occur at the beginning of the recording)
                # NOTE: better think about this method again
                y = librosa.onset.onset_detect(audio, constants.sampling_rate)
                audio = audio[y[0]:] # adding this line because ther5e might be more than one onsets occurring in the recording

                # Better fundamental estimation (TODO: use librosa.pyin instead, delete next line and se midi_flag=False to avoid f0 re-compute)
                fundamental_init = librosa.midi_to_hz(constants.tuning[string] + fret)
                # ToolBoxObj = ToolBox(partial_tracking_func=compute_partials_with_order, inharmonicity_compute_func=compute_inharmonicity, 
                #                 partial_func_args=[fundamental_init/2, constants, StrBetaObj], inharmonic_func_args=[])
                ToolBoxObj = ToolBox(partial_tracking_func=compute_partials, inharmonicity_compute_func=compute_inharmonicity, 
                                partial_func_args=[constants.no_of_partials, fundamental_init/2, constants, StrBetaObj], inharmonic_func_args=[])
                note_instance = NoteInstance(fundamental_init, 0, audio, ToolBoxObj, constants.sampling_rate, constants, midi_flag=True)

                # Detect plucked string (i.e. assigns value to note_instance.string)
                Inharmonic_Detector.DetectString(note_instance, StrBetaObj, constants.betafunc, constants)

                # Compute Confusion Matrix
                InhConfusionMatrixObj.matrix[string][note_instance.string] += 1
   
    InhConfusionMatrixObj.plot_confusion_matrix(constants, normalize= True, 
                                                    title = str(constants.guitar) + str(constants.no_of_partials) +
                                                        'Inharmonic Confusion Matrix' +
                                                        str(round(InhConfusionMatrixObj.get_accuracy(),3)))

if __name__ == '__main__':

    StrBetaObj = TrainWrapper(constants)
    # compute_partial_orders(StrBetaObj, constants)
    testHjerrildChristensen(constants, StrBetaObj)