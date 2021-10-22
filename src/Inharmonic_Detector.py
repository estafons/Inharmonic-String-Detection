
import math
import librosa
from constants_parser import Constants
from inharmonic_Analysis import NoteInstance, ToolBox, compute_partials, compute_inharmonicity
import numpy as np
import threading
import matplotlib.pyplot as plt
from string_betas import StringBetas
import utils


class InharmonicDetector():
    def __init__(self, NoteObj : NoteInstance, StringBetasObj : StringBetas):
        self.StringBetasObj = StringBetasObj
        self.NoteObj = NoteObj
        self.Eks=[]



    def DetectString(self, betafunc, constants : Constants):
        """ betafunc is the function to simulate beta. As input takes the combination and the beta array."""
        combs = utils.determine_combinations(self.NoteObj.fundamental, constants)
        if (constants.lower_limit < self.NoteObj.beta < constants.upper_limit):
            betas_diff = [(abs( betafunc(comb, self.StringBetasObj, constants) - self.NoteObj.beta ), comb) for comb in combs]
            self.NoteObj.string = min(betas_diff, key = lambda a: a[0])[1][0] # returns comb where 0 argument is string
        else:
            self.NoteObj.string = 6

    def DetectStringBarbancho(self, betafunc, constants : Constants):
        k0=2
        lim=30 # TODO: fix this
        combs = utils.determine_combinations(self.NoteObj.fundamental, constants)
        betas_star = [betafunc(comb, self.StringBetasObj, constants) for comb in combs]

        _, _, e0_star = self.__voting_scheme(constants, k0, lim, combs, betas_est=betas_star, R=10, Wcont=2)
        D=-e0_star

        R=2.3
        Wcont=0.5
        max=0
        Hists=[]
        for beta_star, comb in zip(betas_star, combs):
            hist_max, Hist, _ = self.__voting_scheme(constants, k0, lim, combs, betas_est=[beta_star], R=R, Wcont=Wcont, D=D, multi=True)
            if max < hist_max:
                max = hist_max 
                self.NoteObj.string = comb[0]
            Hists.append(Hist)

        if constants.plot:
            self.plot_voting_scheme(k0, lim, combs, Hists, Hbins= np.array( [l+Wcont/2 for l in np.arange(-R, R-Wcont+(R)/30, (R)/30)] ), R=R, D=D, betas_est=betas_star, multi=True)

    def __voting_scheme(self, constants : Constants, k0, lim, combs, betas_est, R, Wcont, D=0, multi=False):
        # Initialize
        # Eks = []
        step=(R)/30 # sliding step (not mentioned in Barbancho et al.)
        Hist= np.array( [0]*len(np.arange(-R, R-Wcont+step, step)) )
        Hbins = np.array( [l+Wcont/2 for l in np.arange(-R, R-Wcont+step, step)] )
        window_length = round(2*R*self.NoteObj.fft.size/self.NoteObj.sampling_rate) # bins

        # Build container-hits histogram for all string-fret combinations  
        for beta_est in betas_est:
            Hist, self.Eks = self.__create_histogram(Hist, Hbins, k0, lim, window_length, beta_est=beta_est, D=D) # Hist accumulates counts for each (s,n) comb
       
        if not multi and constants.plot:
            self.plot_voting_scheme(k0, lim, combs, [Hist], Hbins, R)

        hist_max = np.max(Hist)
        hist_max_idx = np.argmax(Hist)
        e0_star = round(Hbins[hist_max_idx],2)

        return hist_max, Hist, e0_star


    def __create_histogram(self, Hist, Hbins, k0, lim, window_length, beta_est=None, D=0):
        self.NoteObj.partials=[]
        self.NoteObj.find_partials(lim, window_length, k0, window_centering_func='beta_based', beta_est=beta_est, D=D) # result in note_instance.partials

        F_hat = [partial.frequency for partial in self.NoteObj.partials]
        F_star= [k*self.NoteObj.fundamental * np.sqrt(1+beta_est*k**2) for k in range(k0,lim)] # NOTE: "for each partial found". 2 to lim or 10 to lim ??
        Ek = [(F_star[k-k0] - F_hat[k-k0]) / k*np.sqrt(1+beta_est*k**2) for k in range(k0,lim)]
        self.Eks += [Ek]

        # Create histogram
        for ek in Ek:
            argmin = np.argmin(np.abs(Hbins-ek)) 
            Hist[argmin] +=1

        return Hist, self.Eks


    def plot_voting_scheme(self, k0, lim, combs, Hists, Hbins, R, D=0, betas_est=None, multi=False):
        # TODO:
        pos = np.arange(len(Hists[0]))

        N = len(Hists)
        fig5 = plt.figure(constrained_layout=True, figsize=(15, 10))
        widths = [7] + [2] * N
        heights = [10]
        spec5 = fig5.add_gridspec(ncols=N+1, nrows=1, width_ratios=widths, height_ratios=heights)
        axs=[]
        for col in range(N+1):
            ax = fig5.add_subplot(spec5[0, col])
            axs.append(ax)

        for Ek, comb in zip(self.Eks, combs):
            axs[0].plot(range(k0,lim), Ek, label = str(comb))
        axs[0].set_ylim(-R-0.1,R+0.1)
        axs[0].grid()
        axs[0].legend()

        for i, Hist in enumerate(Hists):
            axs[i+1].barh(pos, Hist)
            axs[i+1].grid()
            axs[i+1].set_yticks(pos[::3])
            axs[i+1].set_yticklabels(np.round_(Hbins[::3],1), rotation='0')

        plt.show()


        #Another fig
        if multi:
            fig = plt.figure(figsize=(15, 10))
            timer = fig.canvas.new_timer(interval = 3000) # creating a timer object and setting an interval of 3000 milliseconds
            timer.add_callback(utils.close_event)

            axs=[]
            for row in range(N):
                ax = fig.add_subplot(N, 1, row+1)
                axs.append(ax)

            # ax1 = fig.add_subplot(2, 1, 1)
            # ax2 = fig.add_subplot(2, 1, 2)
            #  TODO: fix lim
            peak_freqs = [partial.frequency for partial in self.NoteObj.partials]
            peaks_idx = [partial.peak_idx for partial in self.NoteObj.partials]
            
            # note_instance.plot_partial_deviations(lim=30, res=self.N.abc, ax=ax1, note_instance=note_instance, annos_string=-1, tab_instance=None) #, peaks_idx=Peaks_Idx)
            for i, beta_est in enumerate(betas_est):
                self.NoteObj.plot_DFT(peak_freqs, peaks_idx, lim=30, ax=axs[i], w=2*R, D=D, beta_est=beta_est, window_centering_func='beta_based') 
            plt.show()



