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
import matplotlib.patches as mpatches
from constants_parser import Constants

import matplotlib.pyplot as plt

# from ransac import RansacModel
# from linearleastsquare import LinearLeastSqaureModel


class Partial():
    def __init__(self, frequency, order, peak_idx):
        self.frequency = frequency
        self.order = order
        self.peak_idx = peak_idx

class ToolBox():
    """here all tools developed are stored. designed this way so 
    it can be expanded and incorporate other methods for partial detection 
    or computing beta coefficient etc. For example usage/alterations see bellow"""

    def __init__(self, partial_tracking_func, inharmonicity_compute_func, partial_func_args, inharmonic_func_args):
        self.partial_tracking_func = partial_tracking_func  # e.g. iterative_compute_of_partials_and_betas
        self.inharmonic_func = inharmonicity_compute_func # e.g. compute_beta_with_regression
        self.partial_func_args = partial_func_args  # e.g. [constants.no_of_partials, tab_instance.fundamental/2, constants, StrBetaObj]
        self.inharmonic_func_args = inharmonic_func_args # e.g. []
    
class NoteInstance():
    """move to other level of package"""
    def __init__(self, fundamental, onset, audio, ToolBoxObj:ToolBox ,sampling_rate, constants : Constants, longaudio=np.array([]), midi_flag = False, Training=False):
        self.fundamental = fundamental
        self.onset = onset
        self.audio = audio
        self.longaudio = longaudio # may be used for ("internal") f0 computation
        self.sampling_rate = constants.sampling_rate
        self.polyfit = constants.polyfit
        self.fft=np.fft.fft(self.audio,n = constants.size_of_fft)
        self.frequencies=np.fft.fftfreq(constants.size_of_fft,1/self.sampling_rate)
        self.partials = []
        self.differences = []
        self.abc = []
        self.large_window = None
        # self.train = False
        self.string = None
        self.constants = constants
        if midi_flag:
            self.recompute_fundamental(constants, fundamental/2)

        if constants.detector == 'custom' or Training:
            ToolBoxObj.partial_tracking_func(self, ToolBoxObj.partial_func_args) # if a different partial tracking is incorporated keep second function arguement, else return beta from second function and change entirely

    def find_partials(self, lim, window_length, k0, window_centering_func='polyfit', a=None,b=None,c=None, beta_est=None, D=0):
        f0 = self.fundamental
        for k in range(k0,lim): # NOTE: k0(=2) stands for the 2nd partial!         
            if window_centering_func == 'beta_based':
                center_freq = k*(f0+D) * np.sqrt(1+beta_est*k**2)
            elif window_centering_func == 'polyfit':
                center_freq = ployfit_centering_func(k,f0, a=a,b=b,c=c) # centering window in which to look for peak/partial
            
            try:
                filtered = zero_out(self.fft, center_freq=center_freq , window_length=window_length, constants=self.constants)
                # # peaks, _  = scipy.signal.find_peaks(np.abs(filtered),distance=100000, prominence=0.0001) # better way to write this?
                # peaks_idx, _  = scipy.signal.find_peaks(np.abs(filtered),distance=100000) # better way to write this?
                peaks_idx = [np.argmax(np.abs(filtered))]
                # peaks = scipy.signal.find_peaks_cwt(np.abs(filtered),widths=np.arange(1,10))
                # peaks, _  = scipy.signal.find_peaks(np.abs(filtered),distance=None) 
                max_peak = self.frequencies[peaks_idx[0]]
                np.argmax(np.abs(filtered))
                # max_peak = self.weighted_argmean(peak_idx=peaks[0], w=6)
                ###### RESULT ###### 
                self.partials.append(Partial(frequency=max_peak, order=k, peak_idx=peaks_idx[0]))
                ####################
            
            except Exception as e:
                print(e)
                print('MyExplanation: Certain windows where peaks are to be located surpassed the length of the DFT.')
                break

    def plot_DFT(self, peaks=None, peaks_idx=None, lim=None, ax=None, save_path=None, w=None, D=0, beta_est=None, window_centering_func='polyfit'):
        if window_centering_func=='polyfit':
            [a,b,c] = self.abc
        if not w:
            w = self.large_window     
        # main function
        ax.plot(self.frequencies, self.fft.real)

        for k in range(50): # draw vertical red dotted lines indicating harmonics
            ax.axvline(x=self.fundamental*k, color='r', ls='--', alpha=0.85, lw=0.5)     

        f0 = self.fundamental
        if w: # draw windows as little boxes
            for k in range(1,lim+1):
                if window_centering_func=='polyfit':
                    f = ployfit_centering_func(k,f0, a=a,b=b,c=c)
                elif window_centering_func=='beta_based':
                    # f = k*f0 * np.sqrt(1+beta_est*k**2)
                    if beta_est: # for 'barbancho' detector we want b_est (i.e. β*)
                        f = k*(f0+D) * np.sqrt(1+beta_est*k**2)
                        # f = beta_centering_func(k, beta_est, f0, D) # 
                    else:
                        f = k*(f0+D) * np.sqrt(1+self.beta*k**2)
                        # f = beta_centering_func(k, self.beta, f0, D)

                rect=mpatches.Rectangle((f-w//2,-80),w,160, fill=False, color="purple", linewidth=2)
                ax.add_patch(rect)

        if peaks and peaks_idx: # draw peaks
            # ax.plot(peaks, self.fft.real[peaks_idx], "x", alpha=0.7)
            ax.plot(peaks, self.frequencies[peaks_idx], "x", alpha=0.7)

        if window_centering_func=='polyfit':
            ax.set_xlim(0, ployfit_centering_func(lim+1,f0, a=a,b=b,c=c))
        elif window_centering_func=='beta_based':
            ax.set_xlim(0, beta_centering_func(lim+1, beta_est, f0, D))

        ax.set_ylim(-100, 100)

        return ax

    def plot_partial_deviations(self, lim=None, res=None, peaks_idx=None, ax=None, note_instance=None, annos_string=None, tab_instance=None, w=None):

        differences = self.differences
        if not w:
            w = self.large_window
        [a,b,c] = res
        kapa = np.linspace(0, lim, num=lim*10)
        y = a*kapa**3 + b*kapa + c

        if peaks_idx:
            PeakAmps = [ self.frequencies[peak_idx] for peak_idx in peaks_idx ]
            PeakAmps = PeakAmps / max(PeakAmps) # Normalize
        else:
            PeakAmps = 1
        ax.scatter(np.arange(2,len(differences)+2), differences, alpha=PeakAmps)
        f0 = self.fundamental
        # plot litte boxes
        for k in range(2, len(differences)+2):
            pos = ployfit_centering_func(k, f0, a, b, c) - k*f0

            rect=mpatches.Rectangle((k-0.25, pos-w//2), 0.5, w, fill=False, color="purple", linewidth=2, alpha=0.8)
            # plt.gca().add_patch(rect)
            ax.add_patch(rect)

        ax.plot(kapa, y, label = 'new_estimate')
        ax.grid()
        ax.legend()

        if annos_string:
            if note_instance.string == annos_string:
                c = 'green'
            else:
                c = 'red'
        else:
            c = 'black'
        
        if tab_instance:
            plt.title("pred: "+ str(note_instance.string) + ", annotation: " + str(annos_string) + ', fret: ' + str(tab_instance.fret) + ' || f0: ' + str(round(self.fundamental,2)) + ', beta_estimate: '+ str(round(self.beta,6)), color=c) # + '\n a = ' + str(round(a,5)), color=c)
        else:
            plt.title("pred: "+ str(note_instance.string) + ", annotation: " + str(annos_string) + ' || f0: ' + str(round(self.fundamental,2)) + ', beta_estimate: '+ str(round(self.beta,6)), color=c)# + '\n a = ' + str(round(a,5)), color=c)

        return ax

    def weighted_argmean(self, peak_idx, w=6):
        min_idx = max(peak_idx - w//2, 0)
        max_idx = min(peak_idx + w//2, len(self.frequencies)-1)
        window = range(min_idx, max_idx+1)
        amp_sum = sum([self.fft.real[i] for i in window])
        res = sum( [self.fft.real[i]/amp_sum * self.frequencies[i] for i in window] )
        return res

    def recompute_fundamental(self, constants : Constants, window = 10): 
        
        if not self.longaudio.any(): # if we want to work on given self.audio (and not on self.longaudio)
            filtered = zero_out(self.fft, self.fundamental, window, constants)
        else:
            longaudio_fft = np.fft.fft(self.longaudio,n = constants.size_of_fft)
            filtered = zero_out(longaudio_fft, self.fundamental, window, constants)
        peaks, _  =scipy.signal.find_peaks(np.abs(filtered),distance=100000) # better way to write this?
        
        max_peak_freq = self.weighted_argmean(peak_idx=peaks[0], w=0) # w=0 means that no weighted argmean is employed, Note: equivalent to # max_peak_freq = self.frequencies[peaks[0]]

        self.fundamental = max_peak_freq
        return max_peak_freq


def beta_centering_func(k, beta, f0, D=0):
    center_freq = k*(f0+D) * np.sqrt(1+beta*k**2)
    return center_freq
        
def ployfit_centering_func(k,f0=None,a=None,b=None,c=None):
    center_freq = a*k**3+b*k+c + (k*f0)
    return center_freq        




def iterative_compute_of_partials_and_betas(note_instance, partial_func_args):
    """compute partials for note instance. 
    large_window is the length of window arround k*f0 that the partials are tracked with highest peak."""
    # no_of_partials = partial_func_args[0] NOTE: deal with it somehow
    note_instance.large_window = partial_func_args[1]
    constants = partial_func_args[2]
    window_length = round(note_instance.large_window*note_instance.fft.size/note_instance.sampling_rate)
    f0 = note_instance.fundamental

    # Beta Computation/Measuement
    a, b, c = 0, 0, 0
    step=2
    k0=2 # first partial to consider
    bound = constants.no_of_partials + constants.no_of_partials % step
    for lim in range(6,31,step):
        ##### CORE COMPUTATION ###
        note_instance.find_partials(lim, window_length, k0, window_centering_func='polyfit', a=a,b=b,c=c) # result in note_instance.partials
        # iterative beta estimates
        _, [a,b,c] = compute_beta_with_regression(note_instance, [])
        note_instance.abc = [a,b,c]
        # compute differences/deviations
        note_instance.differences, orders = zip(*compute_differences(note_instance))
        if lim<30:
            note_instance.partials=[]   

    if constants.plot_train: 
        peak_freqs = [partial.frequency for partial in note_instance.partials]
        peaks_idx = [partial.peak_idx for partial in note_instance.partials]
        fig = plt.figure(figsize=(15, 10))
        ax1 = fig.add_subplot(2, 1, 1)
        ax2 = fig.add_subplot(2, 1, 2)
        note_instance.plot_partial_deviations(lim=30, res=note_instance.abc, ax=ax1, note_instance=note_instance) #, peaks_idx=Peaks_Idx)
        note_instance.plot_DFT(peak_freqs, peaks_idx, lim=30, ax=ax2)   
        plt.show()
    

def compute_differences(note_instance):
    differences = []
    for i, partial in enumerate(note_instance.partials):
        differences.append((abs(partial.frequency-(i+2)*note_instance.fundamental), i)) # i+2 since we start at first partial of order k=2
    return differences

def compute_beta_with_regression(note_instance, inharmonic_func_args):
    differences, orders = zip(*compute_differences(note_instance))
  
    u=np.array(orders)+2
    if note_instance.polyfit == 'lsq':
        res=compute_least(u,differences) # least_squares
    if note_instance.polyfit == 'Thei':
        res=compute_least_TheilSen(u,differences) 
    
    [a,b,c]=res
    beta=2*a/(note_instance.fundamental+b) # Barbancho et al. (17)
    note_instance.beta = beta
    return beta, res
    
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
    """return amplitude values of fft arround a given frequency; when outside window amplitude is zeroed out"""
    sz = fft.size
    x = np.zeros(sz,dtype=np.complex64)
    temp = fft
    # temp = fft.real # NOTE: could this be a good idea
    dom_freq_bin = int(round(center_freq*sz/constants.sampling_rate))
    window_length = int(window_length)

    # for i in range(dom_freq_bin-window_length,dom_freq_bin+window_length): #NOTE: possible error
    for i in range(dom_freq_bin-window_length//2, dom_freq_bin+window_length//2): # __gb_
        x[i] = temp[i]**2

    return x

