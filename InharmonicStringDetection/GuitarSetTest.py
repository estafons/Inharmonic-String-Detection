import jams
from pathlib import Path
import matplotlib.pyplot as plt
import itertools

from track_class import *
from Inharmonic_Detector import *
from inharmonic_Analysis import *
import constants
import genetic

StrBetaObj = Inharmonic_Detector.StringBetas(barray = constants.barray)

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
    def plot_confusion_matrix(self,
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


def read_tablature_from_GuitarSet(jam_name):
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
    return Annotations(tups)

def load_data(track_name, annotation_name):
    """function that loads annotation and audio file and returns instances"""
    track_name = Path(constants.TRACK_PATH + track_name)
    annotation_name = Path(constants.ANNOS_PATH + annotation_name)

    data, _ = librosa.load(track_name, constants.sampling_rate) # _ cause dont need to reassign sampling rate
    annotations = read_tablature_from_GuitarSet(annotation_name)
    tups = [(x.onset,x.fundamental, 6) for x in annotations.tablature.tablature]
    tablature = Tablature(tups, data)
    track_instance = TrackInstance(tablature, data)
    return track_instance, annotations

def predictTabThesis(track_instance : TrackInstance, annotation : Annotations):
    """Inharmonic prediction of tablature as implemented for thesis """
    for tab_instance in track_instance.tablature.tablature:
        ToolBoxObj = ToolBox(compute_partials, compute_inharmonicity, [constants.NO_OF_PARTIALS, tab_instance.fundamental/2], [])
        note_instance = NoteInstance(tab_instance.fundamental, tab_instance.onset, tab_instance.note_audio, ToolBoxObj, track_instance.sampling_rate)
        Inharmonic_Detector.DetectString(note_instance, StrBetaObj, Inharmonic_Detector.betafunc)
        tab_instance.string = note_instance.string
        if tab_instance.string != 6: # 6 marks inconclusive
            tab_instance.fret = Inharmonic_Detector.hz_to_midi(note_instance.fundamental) - constants.tuning[note_instance.string]
        else:
            tab_instance.fret = None


def testGuitarSet():
    """ function that runs tests on the jams files mentioned in the given file 
    and plots the confusion matrixes for both the genetic and inharmonic results."""
    print(constants.NO_OF_PARTIALS)
    InhConfusionMatrixObj = ConfusionMatrix((6,7), inconclusive = True)
    GenConfusionMatrixObj = ConfusionMatrix((6,6), inconclusive = False)
    with open(Path(constants.DATASET_NAMES + constants.LISTOFTRACKSFILE)) as n:
        lines = n.readlines()
    for name in lines:
        name = name.replace('\n', '')
        track_name = name[:-5] + '_' + constants.DATASET +'.wav'
        track_instance, annotations = load_data(track_name, name)
        predictTabThesis(track_instance, annotations)
        InhConfusionMatrixObj.add_to_matrix(track_instance.tablature.tablature, annotations)
        tab, g = genetic.genetic(track_instance.tablature)
        GenConfusionMatrixObj.add_to_matrix(tab, annotations)
    InhConfusionMatrixObj.plot_confusion_matrix(normalize= True, 
                                title = str(constants.NO_OF_PARTIALS) + 'Inharmonic Confusion Matrix' +str(round(InhConfusionMatrixObj.get_accuracy(),3)))
    GenConfusionMatrixObj.plot_confusion_matrix(normalize= True, 
                              title = 'Genetic Confusion Matrix'+str(round(GenConfusionMatrixObj.get_accuracy(),3)))

testGuitarSet()