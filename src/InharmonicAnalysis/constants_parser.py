import configparser
from InharmonicAnalysis.betafuncs import aphfunc, betafunc, expfunc, linfunc

def is_string(value):
    try:
        str(value)
        return True
    except(ValueError):
        return False

def is_float(value):
    try:
        float(value)
        return True
    except(ValueError):
        return False

def is_int(value):
    try:
        int(value)
        return True
    except(ValueError):
        return False

def is_list_of_int(value):
    try:
        [int(x) for x in value.split(", ")]
        return True
    except(ValueError):
        return False

class Constants():
    def __init__(self, config_name, workspace_folder):
        config = configparser.ConfigParser()
        config.read(config_name)

        for section_name in config.sections():
            for key, value in config.items(section_name):
                if str(key) == 'size_of_fft':
                    setattr(self, key, 2**(int(value)))
                elif is_int(value):
                    value = int(value)
                    setattr(self, key, value)
                elif is_float(value):
                    value = float(value)
                    setattr(self, key, value)
                elif is_list_of_int(value):
                    value = [int(x) for x in value.split(", ")]
                    setattr(self, key, value)
                elif is_string(value):
                    setattr(self, key, value)
                else:
                    raise ValueError("constants.ini arguement with name " + str(key) + "is of innapropriate value."+
                         "chansge value or suplement Constants class in constants_parser.py")    
        self.track_path = workspace_folder + '/data/audio/'
    # Path were training data are stored
        self.training_path = workspace_folder + '/data/train/'
        # Path were results should be stored
        # self.result_path = workspace_folder + '/data/results/'
        self.result_path = workspace_folder + '/results/'
        # Path were annotations are stored
        self.annos_path = workspace_folder + '/data/annos/'
        # txt file that contains the list of names of the tracks to be tested (GuitarSet)
        self.listoftracksfile = '/names.txt'
        # Path were listoftracksfile is stored NOTE: commented out!!
        # self.dataset_names_path = workspace_folder

        if self.train_mode == '1Fret':
            self.betafunc = betafunc
            assert len(self.train_frets) > 0
        elif self.train_mode == '2FretA':
            self.betafunc = expfunc
            assert len(self.train_frets) == 2
        if self.train_mode == '2FretB':
            self.betafunc = linfunc
            assert len(self.train_frets) == 2
        elif self.train_mode == '3Fret':
            self.betafunc = aphfunc
            assert len(self.train_frets) == 3



