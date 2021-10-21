
import math
import librosa
from constants_parser import Constants
from inharmonic_Analysis import NoteInstance, ToolBox, compute_partials, compute_inharmonicity
import numpy as np
import threading
import matplotlib.pyplot as plt


class StringBetas():
    def __init__(self, barray, constants : Constants):
        self.betas_array = barray #0->string 1->fret
        self.betas_list_array = [[[] for x in range(0,constants.no_of_frets)] for i in range(0,len(constants.tuning))]

    def add_to_list(self, note_instance):
        self.betas_list_array[note_instance.string][note_instance.fret].append(note_instance.beta)
        #self.betas_list_array.remove(0)
    
    def input_instance(self, instance_audio, midi_note, string, constants : Constants):
        fundamental = librosa.midi_to_hz(midi_note)
        ToolBoxObj = ToolBox(compute_partials, compute_inharmonicity, [constants.no_of_partials, fundamental/2, constants], [])
        note_instance = NoteInstance(fundamental, 0, instance_audio, ToolBoxObj, constants.sampling_rate, constants, midi_flag = True)
        fundamental = note_instance.recompute_fundamental(constants)
        note_instance = NoteInstance(fundamental, 0, instance_audio, ToolBoxObj, constants.sampling_rate, constants) # compute again with recomputed fundamental
        ToolBoxObj = ToolBox(compute_partials, compute_inharmonicity, [constants.no_of_partials, fundamental/2, constants], [])
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

class InharmonicDetector():
    def __init__(self, NoteObj : NoteInstance, StringBetasObj : StringBetas):
        self.StringBetasObj = StringBetasObj


def DetectString(NoteObj : NoteInstance, StringBetasObj : StringBetas, betafunc, constants : Constants):
    """ betafunc is the function to simulate beta. As input takes the combination and the beta array."""
    combs = determine_combinations(NoteObj.fundamental, constants)
    if (constants.lower_limit < NoteObj.beta < constants.upper_limit):
        betas_diff = [(abs( betafunc(comb, StringBetasObj, constants) - NoteObj.beta ), comb) for comb in combs]
        NoteObj.string = min(betas_diff, key = lambda a: a[0])[1][0] # returns comb where 0 argument is string
    else:
        NoteObj.string = 6

def DetectStringBarbancho(NoteObj : NoteInstance, StringBetasObj : StringBetas, betafunc, constants : Constants):
    combs = determine_combinations(NoteObj.fundamental, constants)
    betas_star = [betafunc(comb, StringBetasObj, constants) for comb in combs]
    # for beta_star in betas_star:
    
    _, e0_star = voting_scheme(constants, NoteObj, combs, betas_est=betas_star, R=10, Wcont=2)
    D=-e0_star
    # aaa
    for beta_star in betas_star:
        voting_scheme(constants, NoteObj, combs, betas_est=[beta_star], R=2.3, Wcont=0.5, D=D)

def voting_scheme(constants : Constants, NoteObj : NoteInstance, combs, betas_est, R, Wcont, D=0):
    # Initialize
    k0=2
    lim=30 # TODO: fix this
    Eks = []
    step=(R)/30 # sliding step (not mentioned in Barbancho et al.)
    Hist= np.array( [0]*len(np.arange(-R, R-2, step)) )
    Hbins = np.array( [l+Wcont/2 for l in np.arange(-R, R-2, step)] )
    window_length = round(2*R*NoteObj.fft.size/NoteObj.sampling_rate) # bins

    # Build container-hits histogram for all string-fret combinations  
    for beta_est in betas_est:
        Hist, Eks = create_histogram(NoteObj, Hist, Hbins, Eks, k0, lim, window_length, beta_est=beta_est, D=D) # Hist accumulates counts for each (s,n) comb

    pos = np.arange(len(Hist))

    if constants.plot:
        plot_voting_scheme(k0, lim, Eks, combs, pos, Hist, Hbins, R)

    hist_max_idx = np.argmax(Hist)
    e0_star = round(Hbins[hist_max_idx],2)
    return hist_max_idx, e0_star


def create_histogram(NoteObj : NoteInstance, Hist, Hbins, Eks, k0, lim, window_length, beta_est=None, D=0):
    NoteObj.partials=[]
    NoteObj.find_partials(lim, window_length, k0, window_centering_func='beta_based', beta_est=beta_est, D=D) # result in note_instance.partials

    F_hat = [partial.frequency for partial in NoteObj.partials]
    F_star= [k*NoteObj.fundamental * np.sqrt(1+beta_est*k**2) for k in range(k0,lim)] # NOTE: "for each partial found". 2 to lim or 10 to lim ??
    Ek = [(F_star[k-k0] - F_hat[k-k0]) / k*np.sqrt(1+beta_est*k**2) for k in range(k0,lim)]
    Eks += [Ek]

    # Create histogram
    for ek in Ek:
        argmin = np.argmin(np.abs(Hbins-ek)) 
        Hist[argmin] +=1

    return Hist, Eks


def plot_voting_scheme(k0, lim, Eks, combs, pos, Hist, Hbins, R):
    fig = plt.figure(figsize=(15, 10))
    # timer = fig.canvas.new_timer(interval = 3000) #creating a timer object and setting an interval of 3000 milliseconds
    # timer.add_callback(close_event)
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    
    for Ek, comb in zip(Eks, combs):
        ax1.plot(range(k0,lim), Ek, label = str(comb))
    ax1.set_ylim(-R-1,R+1)
    ax1.grid()
    ax1.legend()

    ax2.barh(pos, Hist)
    ax2.grid()
    ax2.set_yticks(pos[::3])
    ax2.set_yticklabels(np.round_(Hbins[::3],1), rotation='0')
    plt.show()

def close_event(): # just for plotting
    plt.close() #timer calls this function after 3 seconds and closes the window 


def hz_to_midi(fundamental):
    return round(12*math.log(fundamental/440,2)+69)

def midi_to_hz(midi):
    return 440*2**((midi-69)/12)

def determine_combinations(fundamental, constants):
    res = []
    midi_note = hz_to_midi(fundamental)
    fretboard = [range(x, x + constants.no_of_frets) for x in constants.tuning]
    for index, x in enumerate(fretboard):
        if midi_note in list(x):
            res.append((index, midi_note-constants.tuning[index])) # index is string, second is fret
    try:
        assert(res == []), "No combinations found"
    except AssertionError:
        pass
    return res

