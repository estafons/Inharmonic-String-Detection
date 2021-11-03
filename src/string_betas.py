from constants_parser import Constants
import librosa
from inharmonic_Analysis import NoteInstance, ToolBox, iterative_compute_of_partials_and_betas, compute_beta_with_regression
import numpy as np

class StringBetas():
    def __init__(self, barray, constants : Constants):
        self.betas_array = barray #0->string 1->fret
        self.betas_list_array = [[[] for x in range(0,constants.no_of_frets)] for i in range(0,len(constants.tuning))]

    def add_to_list(self, note_instance):
        self.betas_list_array[note_instance.string][note_instance.fret].append(note_instance.beta)
        #self.betas_list_array.remove(0)
    
    # TODO: maybe, need to move function aeay from this place
    def input_instance(self, instance_audio, midi_note, string, constants : Constants):
        fundamental = librosa.midi_to_hz(midi_note)
        ToolBoxObj = ToolBox(iterative_compute_of_partials_and_betas, compute_beta_with_regression, [constants.no_of_partials, fundamental/2, constants], [])
        note_instance = NoteInstance(fundamental, 0, instance_audio, ToolBoxObj, constants.sampling_rate, constants, midi_flag = True, Training = True)
        # NOTE: changed after paper (maybe I souldn't)
        fundamental = note_instance.recompute_fundamental(constants)
        note_instance = NoteInstance(fundamental, 0, instance_audio, ToolBoxObj, constants.sampling_rate, constants, Training = True) # compute again with recomputed fundamental
        ToolBoxObj = ToolBox(iterative_compute_of_partials_and_betas, compute_beta_with_regression, [constants.no_of_partials, fundamental/2, constants], [])
        
        note_instance.string = string
        note_instance.fret = midi_note - constants.tuning[note_instance.string]
        return note_instance  

    def list_to_medians(self):
        for i, r in enumerate(self.betas_list_array):
            for j, l in enumerate(r):
                self.betas_array[i][j] = np.median(l)

    def set_limits(self, constants):
        max_beta = 0
        for string, s in enumerate(constants.tuning):
            if self.betas_array[string][0] > max_beta:
                max_beta = self.betas_array[string][0]

        constants.upper_limit =  max_beta*2**((constants.no_of_frets)/6+2) # +12 fret margin. didnt ceil up to the nearest, because sometimes 10**-3 appear ceiling up to 10**-2 making a huge difference
        # constants.lower_limit = 10**(math.floor(math.log(np.nanmin(self.betas_array), 10)))
        constants.lower_limit = 10**(-7)