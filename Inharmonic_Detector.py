
import math

import constants
from inharmonic_Analysis import NoteInstance

class StringBetas():
    def __init__(self, barray):
        self.betas_array = barray #0->string 1->fret

class InharmonicDetector():
    def __init__(self, NoteObj : NoteInstance, StringBetasObj : StringBetas):
        self.StringBetasObj = StringBetasObj


def DetectString(NoteObj : NoteInstance, StringBetasObj : StringBetas, betafunc):
    """ betafunc is the function to simulate beta. As input takes the combination and the beta array."""
    combs = determine_combinations(constants.tuning, constants.no_of_frets, NoteObj.fundamental)
    betas = [(abs(betafunc(comb, StringBetasObj) - NoteObj.beta), comb) for comb in combs]
    NoteObj.string = min(betas, key = lambda a: a[0])[1][0] # returns comb where 0 arguement is string
        

def hz_to_midi(fundamental):
    return round(12*math.log(fundamental/440,2)+69)

def determine_combinations(tuning, no_of_frets, fundamental):
    res = []
    midi_note = hz_to_midi(fundamental)
    fretboard = [range(x, x + no_of_frets) for x in tuning]
    for index, x in enumerate fretboard:
        if midi_note in list(x):
            res.append((index, midi_note-tuning[index])) # index is string, second is fret
    return res

def betafunc(comb, StringBetasObj : StringBetas):
    beta = StringBetasObj.betas_array[comb[0]][0] * 2**(comb[1]/6)
    return beta

#----------------lambda beta func example for exp model
def expfunc(comb, StringBetasObj : StringBetas):
    fret1, fret2 = 0, 12
    b2, b1 = StringBetasObj.betas_array[fret2], StringBetasObj.betas_array[fret1]
    a = 6 * (math.log2(b2) - math.log2(b1)) / (fret2-fret1)
    beta = StringBetasObj.betas_array[comb[0]][0] * 2**(a * comb[1]/6)
    return beta