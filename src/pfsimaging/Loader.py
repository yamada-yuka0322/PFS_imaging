import numpy as np
import pandas as pd
from astropy.io import fits

from pfstarget import cuts as Cuts
import os

class Star(object):
    """class object to contain the star objects for a single tract
    
    """
    def __init__(self, name):
        self.name = name
        self.ra = None
        self.dec = None
        
    def load_stars(self, path, tract):
        filename = path + f"/tracts_{self.name}/{tract}.fits"
        if os.path.exists(filename):
            with fits.open(filename) as hdu:
                if len(hdu)==2:
                    data = hdu[1].data
                    self.ra = data['ra']%360
                    self.dec = data['dec']
            
class Random(object):
    """class object to contain the random objects for a single tract
    """
    
    def __init__(self):
        self.ra = None
        self.dec = None
        self.patch = None
        self.mask = None
        
    def load_random(self, path, tract):
        filename = path + f"/tracts_ran/{tract}.fits"
        if os.path.exists(filename):
            with fits.open(filename) as hdu:
                if len(hdu)==2:
                    data = hdu[1].data
                    self.ra = data['ra']%360
                    self.dec = data['dec']
                    self.patch = data['patch']
                    self.mask = Cuts.random_masking(data)
        else:
            print(f'cannot find {filename}')

class Patches(object):
    """Class object to include the properties of the entire patch
    
    """
    
    def __init__(self):
        self.tract = None
        self.patch = None
        self.skymap_id = None
        self.property = None
    
    def load_patches(self, path, property_list, tract):
        filename = path + f"/tracts_patch/{tract}.fits"
        if os.path.exists(filename):
            with fits.open(filename) as hdu:
                if len(hdu)==2:
                    data = hdu[1].data
                    self.tract = data['tract']
                    self.patch = data['patch']
                    self.skymap_id = data['skymap_id']
                    properties = {}
                    for key in property_list:
                        val = data[key]
                        if val.dtype.byteorder == '>':
                            val = val.astype(val.dtype.newbyteorder('='))
                        properties[key] = val
                    self.property = pd.DataFrame(properties)
            
    def get_properties(self, tract):
        properties = {}
        selection = (self.tract == tract)
        for key, data in self.property.items():
            properties[key] = data[selection]
            
        properties['patch'] = self.patch[selection]
        return pd.DataFrame(properties)
