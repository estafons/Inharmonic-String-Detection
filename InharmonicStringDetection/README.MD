# Inharmonic String Detection

## Introduction
In this project, software developed for my thesis in Electrical and Computer Engineering in NTUA is determined. 

The main goal of the thesis is to introduce a modular approach towards guitar tablature transcription in the case of monophonic (no chords) audio recordings. The main contribution is the combination of a novel method for guitar string classification based on inharmonicity analysis and a module incorporating playability constraints. Inharmonicity computation of the distinct note instances facilitates a first step for string classification. An agile few-sample adaptation phase is introduced as a significant phase before the classification. Relying on just a small set of samples of different note recordings, we make it possible for guitar players to adapt the system to their guitar on the spot, regardless of the instrument’s particular physical characteristics. As a second stage, physical limitations and common standards of human performers are incorporated by employing a genetic algorithm which significantly improves accuracy. The genetic algorithm is the last module and has proven to be very beneficial for the system’s accuracy while not needing any training or adaptation.

## Goal
The goal of this project is to simultaneously present the results and conclusions drawn from my thesis, as well as provide some basic adjustable methods for later experiments in the field, especially concerning guitar string detection and inharmonicity computation. 

## Structure

### Classes

#### TrackInstance
TrackInstance represents a whole guitar track recorded. 

#### TabInstance
TabInstance represents one note in a Tablature (onset, fundamental, string).

#### Tablature
Tablature as stated represents a guitar tablature and it's primary use is to hold a list of TabInstances (notes).

### InharmonicAnalysis
The InharmonicAnalysis script holds methods necesary for the computation of the inharmonicity coefficient. The two main functionalities consist of tracking partials and computing the inharmonicity coefficient. Both of them can be replaced by user given functions. This property is encoded with the ToolBox class as it will be explained later.

#### ToolBox class
In order to compute the inharmonicity coefficient usually both the partials must be tracked and then a method for the computation of beta must be employed. In this implementation the methods explained in my thesis are supplied though any user can easily swap each of them with one of her choice. The ToolBox class is used to generate an object that holds the functions that will be used to perform the above tasks.

#### NoteInstance class
The NoteInstance class represents a single note and holds the necessary info to detect the note's partials and the inharmonicity coefficient.

#### Example Usage
```python
ToolBoxObj = ToolBox(compute_partials, compute_inharmonicity, [14, fundamental/2], [])
note_instance = NoteInstance(fundamental, audio, ToolBoxObj, sampling_rate)
print(note_instance.beta, [x.frequency for x in note_instance.partials])
```
where *compute_partials* and *compute_inharmonicity* are the methods described in the thesis and are given in the same script, and the following arguements are the neccesary input for each function

### Inharmonic_Detector
In the Inharmonic_Detector script basic functionality for string classification depending on the inharmonicity computed is presented. 
#### InharmonicDetector.DetectString
This method takes as input a NoteInstance (inharmonicity coefficient is already computed at this stage), an object containing the model devoleped for beta (in our case an array where the position (i,j) holds the beta theoretical estimation for string i and fret j), and a beta_func which is used to estimate the theoretical beta of a string fret combination that does not exist in the model. beta_func can be replaced by the users choice. (For more detail reffer to the thesis)
#### 2 Examples for beta_func
```python
def betafunc(comb, StringBetasObj : StringBetas): #Basic Model
    beta = StringBetasObj.betas_array[comb[0]][0] * 2**(comb[1]/6)
    return beta

def expfunc(comb, StringBetasObj : StringBetas): #Exponential Model
    fret1, fret2 = 0, 12
    b2, b1 = StringBetasObj.betas_array[fret2], StringBetasObj.betas_array[fret1]
    a = 6 * (math.log2(b2) - math.log2(b1)) / (fret2-fret1)
    beta = StringBetasObj.betas_array[comb[0]][0] * 2**(a * comb[1]/6)
    return beta
```
#### Example Usage
```python
ToolBoxObj = ToolBox(compute_partials, compute_inharmonicity, [14, tab_instance.fundamental/2], [])
note_instance = NoteInstance(tab_instance.fundamental, tab_instance.onset, tab_instance.note_audio, ToolBoxObj, track_instance.sampling_rate)
Inharmonic_Detector.DetectString(note_instance, StrBetaObj, Inharmonic_Detector.betafunc)
```