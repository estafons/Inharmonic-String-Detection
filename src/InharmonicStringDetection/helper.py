import numpy as np
from pathlib import Path
import itertools
import matplotlib.pyplot as plt

from track_class import Annotations
import math
from Inharmonic_Detector import StringBetas

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
            plt.savefig(Path(constants.result_path + title.replace(" ", "") +'.png'))
            return plt

