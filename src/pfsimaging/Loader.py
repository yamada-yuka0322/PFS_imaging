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
        self.mask = None
        
    def load_stars(self, tract, config):
        path = os.path.join(config["hsc"]["output_dir"], "star", "tract")
        filename = os.path.join(path, f"{tract}.fits")
        
        mask_path = os.path.join(config["Gaia"]["bsmask_dir"] , f"star/{tract}.fits")
        
        filename = path + f"/tracts_{self.name}/{tract}.fits"
        if os.path.exists(filename) and os.path.exists(mask_path):
            with fits.open(filename) as hdu:
                if len(hdu)==2:
                    data = hdu[1].data
                    g = data['g_input_count']
                    r = data['r_input_count']
                    i = data['i_input_count']
                    z = data['z_input_count']
                    inpatch = data['detect_ispatchinner']
                    intract = data['detect_istractinner']
                    
                    mask = (g >= 4) & (r >= 4) & (i >= 5) & (z >= 5) & inpatch & intract
                    
                    self.ra = data['ra'][mask]%360
                    self.dec = data['dec'][mask]
                    
            with fits.open(mask_path) as hdu:
                if len(hdu)==2:
                    data = hdu[1].data
                    self.ra = data['ra'][mask]
                    self.dec = data['dec'][mask]
                    self.mask = data['halo'][mask] | data['ghost'][mask] | data['blooming'][mask] #bright star mask. true if 'inside' mask
                else:
                    print(f'cannot open {mask_path}')
                    
            
class Random(object):
    """class object to contain the random objects for a single tract
    """
    
    def __init__(self):
        self.ra = None
        self.dec = None
        self.patch = None
        self.mask = None
        
    def load_random(self, tract, config):
        random_path = os.path.join(config["hsc"]["output_dir"], "random", "tract")
        filename = os.path.join(random_path, f"{tract}.fits")
        
        mask_path = outdir = os.path.join(config["Gaia"]["bsmask_dir"] , f"random/{tract}.fits")
        
        if os.path.exists(filename) and os.path.exists(mask_path):
            with fits.open(filename) as hdu:
                if len(hdu)==2:
                    data = hdu[1].data
                    no_overlap = data['detect_ispatchinner'] & data['detect_istractinner']
                    self.patch = data['patch'][no_overlap]
                    mask = Cuts.random_masking(data[[no_overlap]])
                    
            with fits.open(mask_path) as hdu:
                if len(hdu)==2:
                    data = hdu[1].data
                    self.ra = data['ra'][no_overlap]
                    self.dec = data['dec'][no_overlap]
                    bsmask = data['halo'][no_overlap] | data['ghost'][no_overlap] | data['blooming'][no_overlap] #True if inside mask
                    self.mask = bsmask | mask #input count, bright star and pixel related masks. True if 'inside' the masked region
                else:
                    print(f'cannot open {mask_path}')
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
    
    def load_patches(self, property_list, config):
        path = os.path.join(config["hsc"]["output_dir"], "patchqa", "tract_group")
        filename = path + f"0.fits"
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

class TractPatch(object):
    """ Class object for patch tract information. 
        
        """
    def __init__(self, field):
        r"""
        Parameter
        ---------------------
        field: string
        The name of the field

        Example
        ----------------------
        >>> autumn = TractPatch("autumn")
        """
        self.field = field
        self.data = load_patch(field)

    def get_tract(self):
        data = self.data
        return list(data.keys())

def load_patch(field):
    """Function to load the coordinates of tract and patch coordinates

    Parameter
    ---------------------------
    field: string
    The name of the field

    Output
    ---------------------------------
    data: dictionary
    {int(tract):{'center':tuple(ra, dec), 
                'corner':[[ra, dec],[ra, dec],[ra, dec],[ra, dec]],
                'patch':{(0, 0):tuple(ra, dec), (0, 1):tuple(ra, dec) ,..., (8, 8):tuple(ra, dec)}
                }}
    """
    
    data = {}

    file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'dat', 'Field', f'tracts_patches_W-{field}.txt')
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    tract = None
    patch = None
    corners = []

    for line in lines:
        # center of tract
        tract_match = re.search(r"Tract: (\d+)  Center \(RA, Dec\): \((-?[\d.]+) , (-?[\d.]+)\)", line)
        if tract_match:
            tract = int(tract_match.group(1))
            ra = float(tract_match.group(2))
            dec = float(tract_match.group(3))
            if tract not in data:
                data[tract] = {'center':(ra, dec), 'corner':[], 'patch':{}}

        # The four corners of tract
        tract_match = re.search(r"Tract: (\d+)  Corner(\d+) \(RA, Dec\): \((-?[\d.]+) , (-?[\d.]+)\)", line)
        if tract_match:
            tract = int(tract_match.group(1))
            num = int(tract_match.group(2))
            if(num != 4):
                ra = float(tract_match.group(3))
                dec = float(tract_match.group(4))
                corner = data[tract]['corner']
                corner.append([ra, dec])
                data[tract]['corner'] =  corner

        # center of patch
        patch_match = re.match(r"Tract: \d+  Patch: (\d+),(\d+)  Center \(RA, Dec\): \((-?[\d.]+) , (-?[\d.]+)\)", line)
        if patch_match:
            patch_x = int(patch_match.group(1))
            patch_y = int(patch_match.group(2))
            ra = float(patch_match.group(3))
            dec = float(patch_match.group(4))
            patch = (patch_x, patch_y)
            data[tract]['patch'][patch] = (ra, dec)

    return data
