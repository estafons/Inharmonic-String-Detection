import random
import numpy as np
from numpy.core.fromnumeric import std
from numpy.lib.function_base import median
import scipy
from random import choice
from scipy.optimize import least_squares
import librosa
import os
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import (LinearRegression, TheilSenRegressor, RANSACRegressor, HuberRegressor)

from constants_parser import Constants

import sys
# sys.path.append('../../../')
print(sys.path)
from ransac import RansacModel
from linearleastsquare import LinearLeastSqaureModel

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
    def __init__(self, fundamental, onset, audio ,ToolBoxObj:ToolBox ,sampling_rate, constants : Constants, midi_flag = False):
        self.fundamental = fundamental
        self.onset = onset
        self.audio = audio
        self.sampling_rate = constants.sampling_rate
        self.fft=np.fft.fft(self.audio,n = constants.size_of_fft)
        # np.arrange(constants.size_of_fft)
        
        self.frequencies=np.fft.fftfreq(constants.size_of_fft,1/self.sampling_rate)
        self.partials = []
        if midi_flag:
            self.recompute_fundamental(constants, fundamental/2)
        ToolBoxObj.partial_func(self, ToolBoxObj.partial_func_args) # if a different partial tracking is incorporated keep second function arguement, else return beta from second function and change entirely
        if ToolBoxObj.partial_func == compute_partials:
            ToolBoxObj.inharmonic_func(self, ToolBoxObj.inharmonic_func_args)
        
    def recompute_fundamental(self, constants : Constants, window = 10): # delete if not needed
        filtered = zero_out(self.fft, self.fundamental, window, constants)
        peaks, _  =scipy.signal.find_peaks(np.abs(filtered),distance=100000) # better way to write this?
        max_peak = self.frequencies[peaks[0]]
        self.fundamental = max_peak
        return max_peak

def compute_partials(note_instance, partial_func_args):
    """compute up to no_of_partials partials for note instance. 
    Freq_deviate is the length of window arround k*f0 that the partials are tracked with highest peak."""
    no_of_partials = partial_func_args[0]
    freq_diviate = partial_func_args[1]
    constants = partial_func_args[2]
    diviate = round(freq_diviate/(note_instance.sampling_rate/note_instance.fft.size))
    for i in range(2,no_of_partials):
        filtered = zero_out(note_instance.fft, i*note_instance.fundamental, diviate, constants)
        peaks, _  =scipy.signal.find_peaks(np.abs(filtered),distance=100000) # better way to write this?
        max_peak = note_instance.frequencies[peaks[0]]
        note_instance.partials.append(Partial(max_peak, i))





# 
def compute_partials_with_order(note_instance, partial_func_args):
    freq_diviate = partial_func_args[0]
    constants = partial_func_args[1]
    StrBetaObj = partial_func_args[2]
    b = []
    midi_note = round(librosa.hz_to_midi(note_instance.fundamental))
    t =  len(StrBetaObj.beta_lim[midi_note - 40])
    for n in range(0, t):
        note_instance.partials = []
        StrBetaObj.beta_lim[midi_note - 40].sort(reverse = True)
        k_max = StrBetaObj.beta_lim[midi_note - 40][n]
        #if k_max > constants.k_max:
        #    k_max = constants.k_max
        compute_partials(note_instance, [k_max, freq_diviate, constants])
        compute_inharmonicity(note_instance, [])
        #print(k_max, note_instance.beta) #uncoment if you want to check variation of betas in relation with k_max. Indicates why bellow code was added

        # NOTE: check for new threshold
        if note_instance.beta > 10**(-7):
            b.append(note_instance.beta)
    # print()
    # print(b, StrBetaObj.beta_lim[midi_note - 40])        
    if std(b)/np.mean(b) < 0.5: #coefficient of variation. If small then low variance of betas so get median, else mark as inconclusive
        note_instance.beta = median(b)
    else:
        note_instance.beta = 10**(-10)
        #    continue
        #else:
        #    return
    return

def compute_partials_with_order_strict(note_instance, partial_func_args):
    freq_diviate = partial_func_args[0]
    constants = partial_func_args[1]
    StrBetaObj = partial_func_args[2]
    midi_note = round(librosa.hz_to_midi(note_instance.fundamental))
    t =  len(StrBetaObj.beta_lim[midi_note - 40])
    for n in range(0, t):
        note_instance.partials = []
        k = min(StrBetaObj.beta_lim[midi_note - 40])
        
        compute_partials(note_instance, [k, freq_diviate, constants])
        compute_inharmonicity(note_instance, [])
        if note_instance.beta < 10**(-7):
            continue
        else:
            return
    return

def compute_differences(note_instance):
    differences = []
    for i, partial in enumerate(note_instance.partials):
        differences.append((abs(partial.frequency-(i+2)*note_instance.fundamental), i)) # i+2 since we start at first partial of order 2
    return differences

def compute_inharmonicity(note_instance, inharmonic_func_args):
    differences, orders = zip(*compute_differences(note_instance))
    u=np.array(orders)
    res=compute_least(u,differences) # least_squares
    # res=compute_least_TheilSen(u,differences) # least_squares
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


# https://scikit-learn.org/stable/auto_examples/linear_model/plot_robust_fit.html#sphx-glr-auto-examples-linear-model-plot-robust-fit-py
# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.TheilSenRegressor.html#sklearn.linear_model.TheilSenRegressor
def compute_least_TheilSen(u,y): 
    u = u[:, np.newaxis]
    poly = PolynomialFeatures(3)
    # print
    u_poly = poly.fit_transform(u)
    u_poly = np.delete(u_poly, 2, axis=1) # delete second coefficient (i.e. b=0, for  b * x**2)

    # estimator =  LinearRegression(fit_intercept=False)
    # estimator.fit(u_poly, y)

    estimator = TheilSenRegressor(random_state=42)
    # estimator = HuberRegressor()
    estimator.fit(u_poly, y)

    # print("coefficients:", estimator.coef_)
    return estimator.coef_[::-1]


def zero_out(fft, center_freq, window_length, constants : Constants):
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
    arg0 = inharmonic_func_args[0]
    arg1 = inharmonic_func_args[1]
    # do more stuff...
    note_instance.beta = arg0

#-----------end of ToolBox Example---------------

def wrapper_func(fundamental, audio, sampling_rate, constants : Constants):
    
    ToolBoxObj = ToolBox(compute_partials, compute_inharmonicity, [14, fundamental/2, constants], [])
    note_instance = NoteInstance(fundamental, 0, audio, ToolBoxObj, sampling_rate, constants)
    print(note_instance.beta, [x.frequency for x in note_instance.partials])