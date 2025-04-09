import numpy as np
import pandas as pd
from astropy.io import fits

import cuts as Cuts

class Star(object):
    """class object to contain the star objects for a single tract
    
    """
    def __init__(self, name):
        self.name = name
        self.ra = None
        self.dec = None
        
    def load_stars(self, path, tract):
        filename = path + f"/tracts_{self.name}/{tract}.fits"
        with fits.open(path) as hdu:
            data = hdu[1].data
            self.ra = data['ra']%360
            self.dec = data['dec']
            
class Random(object):
    """class object to contain the random objects for a single tract
    """
    
    def __init__(self):
        self.ra = None
        self.dec = None
        self.mask = None
        
    def load_random(self, path, tract):
        filename = path + f"/tracts_random/{tract}.fits"
        with fits.open(path) as hdu:
            data = hdu[1].data
            self.ra = data['ra']%360
            self.dec = data['dec']
            self.mask = Cuts.random_masking(data)

class Patches(object):
    """Class object to include the properties of the entire patch
    
    """
    
    def __init__(self):
        self.tract = None
        self.patch = None
        self.skymap_id = None
        self.property = None
    
    def load_patches(self, path, property_list):
        filename = path + f"/patches.fits"
        with fits.open(path) as hdu:
            data = hdu[1].data
            self.tract = data['tract']
            self.patch = data['patch']
            self.skymap_id = data['skymap_id']
            properties = {}
            for key in property_list:
                properties[key] = data[key]
            self.property = properties
            
    def get_properties(self, tract):
        properties = {}
        selection = (self.tract == tract)
        for key, data in self.property.item():
            properties[key] = data[selection]
            
        properties['patch'] = self.patch[selection]
        return properties