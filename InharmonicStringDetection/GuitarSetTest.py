import jams
from pathlib import Path
import matplotlib.pyplot as plt
import itertools
import configparser

from track_class import *
from Inharmonic_Detector import *
from inharmonic_Analysis import *
from constants_parser import Constants
import genetic

#input from user
config_path = Path("C:\\Users/stefa/Documents//Inharmonic String Detection/InharmonicStringDetection/constants.ini")

constants = Constants()
beta_dict = {0: 1.84196264*10**(-4), 1: 1.13998209*10**(-4), 2: 5.61036666*10**(-5),
                                 3: 3.53238139*10**(-5), 4: 6.07431574*10**(-5), 5: 3.12346527*10**(-5)} # beta dictionary for the open fret based on the GuitarTrain script
beta_dict = {0: 1.79747163*10**(-4), 1: 1.23713369*10**(-4), 2: 5.92775513*10**(-5),
                                 3: 3.82823463*10**(-5), 4: 7.63428658*10**(-5), 5: 2.94933906*10**(-5)}

barray = [[0 if x != 0 else beta_dict[i] for x in range(0,17)] for i in range(0,6)] # array of betas as trained
StrBetaObj = Inharmonic_Detector.StringBetas(barray = barray, constants = constants)

class ConfusionMatrix():
    def __init__(self, size, inconclusive):
        self.matrix = np.zeros(size)
        self.x_classes = ['E', 'A', 'D', 'G', 'B', 'e']
        if inconclusive:
            self.y_classes = ['E', 'A', 'D', 'G', 'B', 'e', 'inconclusive']
        else:
             self.y_classes = ['E', 'A', 'D', 'G', 'B', 'e']
    
    def add_to_matrix(self, tab_as_list, annotations : Annotations):
        for tab_instance, annos_instance in zip(tab_as_list, annotations.tablature.tablature):
            self.matrix[annos_instance.string][tab_instance.string] += 1
    
    def get_accuracy(self):
        return np.trace(self.matrix)/np.sum(self.matrix)
    def plot_confusion_matrix(self, constants,
                            normalize=False,
                            title='Confusion matrix',
                            cmap=plt.cm.Blues):
            """
            This function prints and plots the confusion matrix.
            Normalization can be applied by setting `normalize=True`.
            """
            plt.clf()
            if normalize:
                self.matrix = self.matrix.astype('float') / self.matrix.sum(axis=1)[:, np.newaxis]
                np.nan_to_num(self.matrix,False)
                print("Normalized confusion matrix")
            else:
                print('Confusion matrix, without normalization')
            plt.imshow(self.matrix, interpolation='nearest', cmap=cmap)
            plt.title(title)
            plt.colorbar()
            
            tick_marks_y = np.arange(len(self.y_classes))
            tick_marks_x = np.arange(len(self.x_classes))
            plt.xticks(tick_marks_x,self.x_classes , rotation=45)
            plt.yticks(tick_marks_y, self.y_classes)

            fmt = '.2f' if normalize else '.2f'
            thresh = self.matrix.max() / 2.
            for i, j in itertools.product(range(self.matrix.shape[0]), range(self.matrix.shape[1])):
                plt.text(j, i, format(self.matrix[i, j], fmt),
                            horizontalalignment="center",
                            color="white" if self.matrix[i, j] > thresh else "black")

            plt.ylabel('True label')
            plt.xlabel('Predicted label')
            plt.tight_layout()
            #plt.show()
            plt.savefig(Path(constants.RESULT_PATH + title.replace(" ", "") +'.png'))
            return plt


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
    track_name = Path(constants.TRACK_PATH + track_name)
    annotation_name = Path(constants.ANNOS_PATH + annotation_name)

    data, _ = librosa.load(track_name, constants.sampling_rate) # _ cause dont need to reassign sampling rate
    annotations = read_tablature_from_GuitarSet(annotation_name, constants)
    tups = [(x.onset,x.fundamental, 6) for x in annotations.tablature.tablature]
    tablature = Tablature(tups, data, constants)
    track_instance = TrackInstance(tablature, data, constants)
    return track_instance, annotations

def predictTabThesis(track_instance : TrackInstance, constants = Constants):
    """Inharmonic prediction of tablature as implemented for thesis """
    for tab_instance in track_instance.tablature.tablature:
        ToolBoxObj = ToolBox(compute_partials, compute_inharmonicity, [constants.NO_OF_PARTIALS, tab_instance.fundamental/2, constants], [])
        note_instance = NoteInstance(tab_instance.fundamental, tab_instance.onset, tab_instance.note_audio, ToolBoxObj, track_instance.sampling_rate, constants)
        Inharmonic_Detector.DetectString(note_instance, StrBetaObj, Inharmonic_Detector.betafunc, constants)
        tab_instance.string = note_instance.string
        if tab_instance.string != 6: # 6 marks inconclusive
            tab_instance.fret = Inharmonic_Detector.hz_to_midi(note_instance.fundamental) - constants.tuning[note_instance.string]
        else:
            tab_instance.fret = None


def testGuitarSet(constants : Constants):
    """ function that runs tests on the jams files mentioned in the given file 
    and plots the confusion matrixes for both the genetic and inharmonic results."""

    InhConfusionMatrixObj = ConfusionMatrix((6,7), inconclusive = True)
    GenConfusionMatrixObj = ConfusionMatrix((6,6), inconclusive = False)
    with open(constants.DATASET_NAMES_PATH + constants.LISTOFTRACKSFILE) as n:
        lines = n.readlines()
    for name in lines[:3]:
        print(name)
        name = name.replace('\n', '')
        track_name = name[:-5] + '_' + constants.DATASET +'.wav'
        track_instance, annotations = load_data(track_name, name, constants)
        predictTabThesis(track_instance, constants)
        InhConfusionMatrixObj.add_to_matrix(track_instance.tablature.tablature, annotations)
        #tab, g = genetic.genetic(track_instance.tablature, constants)
        #GenConfusionMatrixObj.add_to_matrix(tab, annotations)
    InhConfusionMatrixObj.plot_confusion_matrix(constants, normalize= True, 
                                title = str(constants.NO_OF_PARTIALS) + 'Inharmonic Confusion Matrix' +str(round(InhConfusionMatrixObj.get_accuracy(),3)))
    #GenConfusionMatrixObj.plot_confusion_matrix(constants, normalize= True, 
    #                          title = 'Genetic Confusion Matrix'+str(round(GenConfusionMatrixObj.get_accuracy(),3)))


testGuitarSet(constants)