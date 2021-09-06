import random
import numpy as np
import scipy
from random import choice
from scipy.optimize import least_squares
import librosa
import os

import constants

class Partial():
    def __init__(self, frequency, order):
        self.frequency = frequency
        self.order = order

class ToolBox():
    """here all tools developed are stored. designed this way so 
    it can be expanded and incorporate other methods for partial detection 
    or computing beta coefficient etc. For example usage/alterations see bellow"""

    def __init__(self, partial_tracking_func, inharmonicity_compute_func, partial_func_args, inharmonic_func_args):
        self.partial_func = partial_tracking_func
        self.inharmonic_func = inharmonicity_compute_func
        self.partial_func_args = partial_func_args
        self.inharmonic_func_args = inharmonic_func_args
    
class NoteInstance():
    """move to other level of package"""
    def __init__(self, fundamental, onset, audio ,ToolBoxObj:ToolBox ,sampling_rate):
        self.fundamental = fundamental
        self.onset = onset
        self.audio = audio
        self.sampling_rate = constants.sampling_rate
        self.fft=np.fft.fft(self.audio,n = constants.size_of_fft)
        self.frequencies=np.fft.fftfreq(constants.size_of_fft,1/self.sampling_rate)
        #self.recompute_fundamental() # delete if not needed
        self.partials = []
        ToolBoxObj.partial_func(self, ToolBoxObj.partial_func_args) # if a different partial tracking is incorporated keep second function arguement, else return beta from second function and change entirely
        ToolBoxObj.inharmonic_func(self, ToolBoxObj.inharmonic_func_args)

    def recompute_fundamental(self): # delete if not needed
        filtered = zero_out(self.fft, self.fundamental, 10)
        peaks, _  =scipy.signal.find_peaks(np.abs(filtered),distance=100000) # better way to write this?
        max_peak = self.frequencies[peaks[0]]
        self.fundamental = max_peak

def compute_partials(note_instance, partial_func_args):
    """compute up to no_of_partials partials for note instance. 
    Freq_deviate is the length of window arround k*f0 that the partials are tracked with highest peak."""
    no_of_partials = partial_func_args[0]
    freq_diviate = partial_func_args[1]
    diviate = round(freq_diviate/(note_instance.sampling_rate/note_instance.fft.size))
    for i in range(2,no_of_partials):
        filtered = zero_out(note_instance.fft, i*note_instance.fundamental, diviate)
        peaks, _  =scipy.signal.find_peaks(np.abs(filtered),distance=100000) # better way to write this?
        max_peak = note_instance.frequencies[peaks[0]]
        note_instance.partials.append(Partial(max_peak, i))


def compute_differences(note_instance):
    differences = []
    for i, partial in enumerate(note_instance.partials):
        differences.append((abs(partial.frequency-(i+2)*note_instance.fundamental), i)) # i+2 since we start at first partial of order 2
    return differences

def compute_inharmonicity(note_instance, inharmonic_func_args):
    differences, orders = zip(*compute_differences(note_instance))
    u=np.array(orders)
    res=compute_least(u,differences)
    [a,b,c]=res
    beta=2*a/(note_instance.fundamental+b)
    note_instance.beta = beta
    

def compute_least(u,y):
    def model(x, u):
        return x[0] * u**3 + x[1]*u + x[2]
    def fun(x, u, y):
        return model(x, u)-y
    def jac(x, u, y):
        J = np.empty((u.size, x.size))
        J[:, 0] = u**3
        J[:, 1] = u
        J[:, 2] = 1
        return J
    x0=[0.00001,0.00001,0.000001]
    res = least_squares(fun, x0, jac=jac,bounds=(0,np.inf), args=(u, y),loss = 'soft_l1', verbose=0)
    return res.x    

def zero_out(fft, center_freq, window_length):
    """return amplitude values of fft arround a given frequency when outside window amplitude is zeroed out"""
    sz = fft.size
    x = np.zeros(sz,dtype=np.complex64)
    temp = fft
    dom_freq_bin = int(round(center_freq*sz/constants.sampling_rate))
    window_length = int(window_length)
    for i in range(dom_freq_bin-window_length,dom_freq_bin+window_length):
        x[i] = temp[i]**2
    return x


"""here on tests"""

#-----------ToolBox Example--------------------

#when changing only partial computing function
def example_compute_partial(note_instance, partial_func_args):
    arg0 = partial_func_args[0]
    arg1 = partial_func_args[1]
    arg2 = partial_func_args[2]
    for i in range(0, arg0):
        x = random.choice([arg1, arg2])
        note_instance.partials.append(Partial(x, i))
# example changing beta computation and probably bypassing previous partial computation
def example_inharmonicity_computation(note_instance, inharmonic_func_args):
    arg0 = partial_func_args[0]
    arg1 = partial_func_args[1]
    # do more stuff...
    note_instance.beta = arg0

#-----------end of ToolBox Example---------------

def wrapper_func(fundamental, audio, sampling_rate):
    
    ToolBoxObj = ToolBox(compute_partials, compute_inharmonicity, [14, fundamental/2], [])
    note_instance = NoteInstance(fundamental, audio, ToolBoxObj, sampling_rate)
    print(note_instance.beta, [x.frequency for x in note_instance.partials])


def main():
    name = os.path.join('C:\\','Users','stefa','Documents','guit_workspace','crop50',
                        '00_BN2-131-B_comp_outputstring1n8.wav')
    audio, sampling_rate = librosa.load(name, constants.sampling_rate)
    fundamental = 146.83
    wrapper_func(fundamental, audio, sampling_rate)

if __name__ == '__main__':
    main()
    
