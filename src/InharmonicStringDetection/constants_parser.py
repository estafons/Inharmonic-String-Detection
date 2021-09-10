import configparser
from pathlib import Path
class Constants():
    def __init__(self, config_name):
        config = configparser.ConfigParser()
        config.read(config_name)
        #PATHS
        self.TRACK_PATH = config.get('GUITARSET_PATHS', 'TRACK_PATH')
        self.TRAINING_PATH = config.get('GUITARSET_PATHS', 'TRAINING_PATH')
        self.RESULT_PATH = config.get('GUITARSET_PATHS', 'RESULT_PATH')
        self.ANNOS_PATH = config.get('GUITARSET_PATHS', 'ANNOS_PATH')
        self.LISTOFTRACKSFILE = config.get('GUITARSET_PATHS', 'LISTOFTRACKSFILE')
        self.DATASET_NAMES_PATH = config.get('GUITARSET_PATHS', 'DATASET_NAMES_PATH')
        self.DATASET = config.get('GUITARSET_PATHS', 'DATASET')
        #INHARMONICITY
        self.tuning = [int(x) for x in config.get('INHARMONICITY', 'tuning').split(", ")]
        self.no_of_frets = config.getint('INHARMONICITY', 'no_of_frets')
        self.sampling_rate = config.getint('INHARMONICITY', 'sampling_rate')
        self.size_of_fft = 2**config.getint('INHARMONICITY', 'size_of_fft')
        self.crop_win = config.getfloat('INHARMONICITY', 'crop_win')
        self.NO_OF_PARTIALS = config.getint('INHARMONICITY', 'NO_OF_PARTIALS')
        #GENETIC
        self.INITIAL_POP = config.getint('GENETIC', 'INITIAL_POP')
        self.INIT_MUTATION_RATE = config.getfloat('GENETIC', 'INIT_MUTATION_RATE')
        self.TOURNSIZE = config.getint('GENETIC', 'TOURNSIZE')
        self.NO_OF_PARENTS = config.getint('GENETIC', 'NO_OF_PARENTS')
        self.NGEN = config.getint('GENETIC', 'NGEN')
        self.CXPB = config.getfloat('GENETIC', 'CXPB')
        self.MUTPB = config.getfloat('GENETIC', 'MUTPB')
        self.MUTPN = config.getfloat('GENETIC', 'MUTPN')
        self.PARENTS_TO_NEXT_GEN = config.getint('GENETIC', 'PARENTS_TO_NEXT_GEN')
        self.OFFSPRING_TO_NEXT_GEN = config.getint('GENETIC', 'OFFSPRING_TO_NEXT_GEN')
        self.END_NO = config.getint('GENETIC', 'END_NO')

        #CONSTRAINTS
        self.CONSTRAINTS_COF = config.getfloat('CONSTRAINTS', 'CONSTRAINTS_COF')
        self.SIMILARITY_COF = config.getfloat('CONSTRAINTS', 'SIMILARITY_COF')
        self.open_cof = config.getfloat('CONSTRAINTS', 'open_cof')
        self.avg_cof = config.getfloat('CONSTRAINTS', 'avg_cof')
        self.string_cof = config.getfloat('CONSTRAINTS', 'string_cof')  
        self.fret_cof = config.getfloat('CONSTRAINTS', 'fret_cof')
        self.depress_cof = config.getfloat('CONSTRAINTS', 'depress_cof')
        self.avg_length = config.getint('CONSTRAINTS', 'avg_length')
        self.time_limit = config.getfloat('CONSTRAINTS', 'time_limit')

        #TRAINING
        self.TRAIN_FRETS = [int(x) for x in config.get('TRAINING', 'TRAIN_FRETS').split(", ")]




