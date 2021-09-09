# Inharmonicity arguements
tuning = [40, 45, 50, 55, 59, 64] # tuning of guitar   
no_of_frets = 20 # number of frets of guitar accounted
beta_dict = {0: 1.84196264*10**(-4), 1: 1.13998209*10**(-4), 2: 5.61036666*10**(-5),
                                 3: 3.53238139*10**(-5), 4: 6.07431574*10**(-5), 5: 3.12346527*10**(-5)} # beta dictionary for the open fret based on the GuitarTrain script
barray = [[0 if x != 0 else beta_dict[i] for x in range(0,17)] for i in range(0,6)] # array of betas as trained
sampling_rate = 44100
size_of_fft = 2**18
crop_win = 0.06 #size of in miliseconds of croped note instances
NO_OF_PARTIALS = 14

#genetic algorithm arguements
INITIAL_POP = 40000
INIT_MUTATION_RATE = 0.3 # mutation rate per note on initial population
TOURNSIZE = 5 # size of tournament on selection
NO_OF_PARENTS = 3000 # Number of parents that are chosen with tournament
NGEN = 100 # number of generations
CXPB = 0.5 # crossover rate
MUTPB = 0.2 # mutation rate per tab
MUTPN =0.1 # mutation rate per note instance
PARENTS_TO_NEXT_GEN = 100 # Number of parents that will go on to the next generation
OFFSPRING_TO_NEXT_GEN = 3000 # Number of offspring that will go on to the next generation
END_NO = 500 # Termination condition when first END_NO individuals are identical

#constraints arguements
CONSTRAINTS_COF = 1 # Coefficient on constraints criterion for the genetic algorithm's evaluation function
SIMILARITY_COF = 2 # Coefficient on similarity criterion for the genetic algorithm's evaluation function
open_cof, avg_cof, string_cof, fret_cof, depress_cof= 1, 1, 1, 1, 1 # Coefficients of each constraint
avg_length = 7 # number of notes that will be taken each time to measure average (avg_function)
time_limit = 1 # onset window arround which average accounts for (in seconds). (avg_function)

#Training arguements
TRAIN_FRETS = [0]


# Paths and miscelaneous
DATASET = 'mix'
TRACK_PATH = "data/single_notes/" # Path where tracks are stored
TRAINING_PATH = "data/crops/" # Path were training data are stored
RESULT_PATH = "results/" # Path were results should be stored
ANNOS_PATH = "data/annos/" # Path were annotations are stored
LISTOFTRACKSFILE = "names.txt" # txt file that contains the list of names of the tracks to be tested (GuitarSet)
DATASET_NAMES = "C:\\Users/stefa/Documents/guit_workspace/single_notes/" # Path were LISTOFTRACKSFILE is stored
