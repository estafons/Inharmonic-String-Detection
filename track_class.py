class TrackInstance():
    def __init__(self, tups, audio, crop_win): # tup gets the onset -fundamental tuple
        """main class. crop_win is the length 
        of the excerpts we will consider. 
        if we consider 60ms windows then all note 
        instances will be cropped at [onset, onset+60ms]"""
        self.tablature = Tablature(tups, audio)
        self.audio = audio
        #self.crop_win = crop_win

class Tablature():
    def __init__(self, tups):
        self.tablature = []
        for x in tups:
            self.tablature.append(TabInstance(x))

class TabInstance():
    def __init__(self, tup):
        self.onset, self.fundamental, self.string = tup
        
