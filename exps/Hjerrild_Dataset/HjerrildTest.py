from pathlib import Path
import os, sys
import librosa
import argparse

'''from sympy import symbols, solve
from sympy import Poly
from sympy.solvers.inequalities import solve_poly_inequality
k = symbols('k')
print(solve_poly_inequality(Poly(k, k, domain='ZZ'), '<'))

polyn = Poly(10**(-4)*k**4 -k -1/2, k)
sol = solve(polyn)
#sol = solve_poly_inequality(polyn, '==')
print(sol)'''

BASE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
cur_path = Path(BASE_PATH + '/src/InharmonicStringDetection')
sys.path.append(str(cur_path))
cur_path = Path(BASE_PATH + '/exps')
sys.path.append(str(cur_path))

from track_class import *
from helper import ConfusionMatrix
from HjerrildTrain import TrainWrapper
from inharmonic_Analysis import *
import Inharmonic_Detector

parser = argparse.ArgumentParser()
parser.add_argument('config_path', type=str)
args = parser.parse_args()

#input from user
#config_path = Path("C:\\Users/stefa/Documents//Inharmonic String Detection/InharmonicStringDetection/constants.ini")
try:
    constants = Constants(args.config_path)
except:
    raise RuntimeError(('could not open ' + str(args.config_path) + ', does not exist or given' +
                 'in wrong format try again as C:\\Users/user/Documents/path_to_config.ini'))


StrBetaObj = TrainWrapper(constants)
def testHjerrildChristensen(constants : Constants):
    InhConfusionMatrixObj = ConfusionMatrix((6,7), inconclusive = True)
    if constants.guitar == 'firebrand':
        dataset_nums = [1,2,3,5,6,7,8,9,10]
    elif constants.guitar == 'martin':
        dataset_nums = [1,2,3,4,5,6,7,8,9,10]
    for dataset_no in dataset_nums:
        for string in range(0,6):
            for fret in range(0,12):
                path_to_track = Path(constants.path_to_hjerrild_christensen +
                                     constants.guitar + str(dataset_no) + 
                                            '/string' +str(string + 1) +'/' + str(fret) +'.wav')
                audio, _ = librosa.load(path_to_track, constants.sampling_rate)

                y = librosa.onset.onset_detect(audio, constants.sampling_rate)
                audio = audio[y[0]:] # adding this line because not all tracks start at the beggining
                fundamental = librosa.midi_to_hz(constants.tuning[string] + fret)
                ToolBoxObj = ToolBox(compute_partials, compute_inharmonicity, 
                                [constants.no_of_partials, fundamental/2, constants], [])
                note_instance = NoteInstance(fundamental, 0, audio, ToolBoxObj, constants.sampling_rate, constants)
                Inharmonic_Detector.DetectString(note_instance, StrBetaObj, Inharmonic_Detector.expfunc, constants)
                InhConfusionMatrixObj.matrix[string][note_instance.string] += 1
    InhConfusionMatrixObj.plot_confusion_matrix(constants, normalize= True, 
                                                    title = str(constants.guitar) + str(constants.no_of_partials) +
                                                        'Inharmonic Confusion Matrix' +
                                                        str(round(InhConfusionMatrixObj.get_accuracy(),3)))


testHjerrildChristensen(constants)
