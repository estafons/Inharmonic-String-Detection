tuning = [40, 45, 50, 55, 59, 64] # tuning of guitar   
no_of_frets = 20 # number of frets of guitar accounted
beta_dict = {0: 0.00012423557794641988, 1: 9.720145237288956*10**(-5), 2: 5.3187112359074767*10**(-5),
                                 3: 3.26859277852029*10**(-5), 4: 5.799282669545177*10**(-5), 5: 2.2414154979144663*10**(-5)}
barray = [[0 if x != 0 else beta_dict[i] for x in range(0,17)] for i in range(0,6)] # array of betas as trained
sampling_rate = 44100
size_of_fft = 2**18
crop_win = 0.06 #size of in miliseconds of croped note instances

#genetic arguements
INITIAL_POP = 40000
INIT_MUTATION_RATE = 0.3 # mutation rate per note on initial population
TOURNSIZE = 5 # size of tournament on selection
NO_OF_PARENTS = 3000
NGEN = 100 # number of generations
CXPB = 0.5 # crossover rate
MUTPB = 0.2 # mutation rate per tab
MUTPN =0.1 # mutation rate per note instance
PARENTS_TO_NEXT_GEN = 100
OFFSPRING_TO_NEXT_GEN = 3000
END_NO = 500

#constraints arguements
CONSTRAINTS_COF = 1
SIMILARITY_COF = 2
open_cof, avg_cof, string_cof, fret_cof, depress_cof= 1, 1, 1, 1, 1
avg_length = 7 # number of notes that will be taken each time to measure average
time_limit = 1 # onset window arround which average accounts for.