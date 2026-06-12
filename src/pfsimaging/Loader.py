import numpy as np
import pandas as pd
from astropy.io import fits

from pfstarget import cuts as Cuts
import os

import re

class Target(object):
    """Container for stellar objects　in a single tract.

    Attributes
    ------------------------------------------------------
    ra: array or None
        ra of stars in degrees. 0~360 degrees

    dec: array or None
        Dec of stars in degrees.

    mask: array of bool or None
        updated bright stellar mask. True if "inside" the mask

    Methods
    ------------------------------------------------------
    load_targets(tract, config):
        Load cosmology targets and corresponding bright-star mask
        for the specified tract. config is a dictionary of path read from the config.yaml file in the config directory
        
    load_targets(tractlist, config):
        Load cosmology targets and corresponding bright-star mask
        for all tract included in tractlist. 
    """
    def __init__(self):
        self.ra = None
        self.dec = None
        self.mask = None
        self.patch = None
        
    def load_targets(self, tract, config):
        path = os.path.join(config["hsc"]["output_dir"], "galaxy", "tract")
        filename = os.path.join(path, f"{tract}.fits")
        
        mask_path = os.path.join(config["Gaia"]["bsmask_dir"] , f"galaxy/{tract}.fits")

        if os.path.exists(filename) and os.path.exists(mask_path):
            with fits.open(filename) as hdu:
                if len(hdu)==2:
                    data = hdu[1].data
                hsc = Cuts._prepare_hsc(data, dust_extinction='desi')
                targets = Cuts.isCosmology(hsc)
                    
                self.ra = hsc[targets]['RA']%360
                self.dec = hsc[targets]['DEC']
                self.patch = hsc[targets]['PATCH']
                    
            with fits.open(mask_path) as hdu:
                if len(hdu)==2:
                    data = hdu[1].data
                    self.mask = data['halo'][targets] | data['ghost'][targets] | data['blooming'][targets] #bright star mask. true if 'inside' mask
                else:
                    print(f'cannot open {mask_path}')
    def load_all_targets(self, tractlist, config):
        all_ra = []
        all_dec = []
        all_mask = []
        all_patch = []
        
        for tract in tractlist:
            ra, dec, patch, mask = load_target_tract(tract, config)
            if ((ra is None)|(dec is None)|(patch is None)):
                continue
            elif (mask is None):
                mask = np.zeros(len(ra), dtype=bool)
            all_ra.append(list(ra))
            all_dec.append(list(dec))
            all_patch.append(list(patch))
            all_mask.append(list(mask))
        self.ra = np.array(all_ra)
        self.dec = np.array(all_dec)
        self.patch = np.array(all_patch)
        self.mask = np.array(all_mask)
        
def load_target_tract(tract, config):
    path = os.path.join(config["hsc"]["output_dir"], "galaxy", "tract")
    filename = os.path.join(path, f"{tract}.fits")
        
    mask_path = os.path.join(config["Gaia"]["bsmask_dir"] , f"galaxy/{tract}.fits")

    if os.path.exists(filename) and os.path.exists(mask_path):
        with fits.open(filename) as hdu:
            if len(hdu)==2:
                data = hdu[1].data
                hsc = Cuts._prepare_hsc(data, dust_extinction='desi')
                targets = Cuts.isCosmology(hsc)
                    
                ra = hsc[targets]['RA']%360
                dec = hsc[targets]['DEC']
                patch = hsc[targets]['PATCH']
            else:
                print(f'cannot open {filename}')
                return None, None, None, None
                    
        with fits.open(mask_path) as hdu:
            if len(hdu)==2:
                data = hdu[1].data
                mask = data['halo'][targets] | data['ghost'][targets] | data['blooming'][targets] #bright star mask. true if 'inside' mask
            else:
                print(f'cannot open {mask_path}')
                return ra, dec, patch, None
        return ra, dec, patch, mask
    
    else:
        return None, None, None, None
class Star(object):
    """Container for stellar objects　in a single tract.

    Attributes
    ------------------------------------------------------
    ra: array or None
        ra of stars in degrees. 0~360 degrees

    dec: array or None
        Dec of stars in degrees.

    mask: array of bool or None
        updated bright stellar mask. True if "inside" the mask
        
    patch: array. or None
        patch ID of stars

    Methods
    ------------------------------------------------------
    load_stars(tract, config):
        Load stellar objects and corresponding bright-star mask
        for the specified tract. config is a dictionary of path read from the config.yaml file in the config directory
    """
    def __init__(self):
        self.ra = None
        self.dec = None
        self.mask = None
        self.patch = None
        
    def load_stars(self, tract, config):
        path = os.path.join(config["hsc"]["output_dir"], "star", "tract")
        filename = os.path.join(path, f"{tract}.fits")
        
        mask_path = os.path.join(config["Gaia"]["bsmask_dir"] , f"star/{tract}.fits")

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
                    self.patch = data['patch'][mask]
                    
            with fits.open(mask_path) as hdu:
                if len(hdu)==2:
                    data = hdu[1].data
                    self.ra = data['ra'][mask]
                    self.dec = data['dec'][mask]
                    self.mask = data['halo'][mask] | data['ghost'][mask] | data['blooming'][mask] #bright star mask. true if 'inside' mask
                else:
                    print(f'cannot open {mask_path}')
                    
            
class Random(object):
    """Container for HSC randoms　in a single tract.

    Attributes
    ------------------------------------------------------
    ra: array or None
        ra of randoms in degrees. 0~360 degrees

    dec: array or None
        Dec of randoms in degrees.
        
    patch: array. or None
        patch ID of randoms

    mask: array of bool or None
        updated bright stellar mask. True if "inside" the mask

    Methods
    ------------------------------------------------------
    load_random(tract, config):
        Load randoms and corresponding bright-star mask
        for the specified tract. config is a dictionary of path read from the config.yaml file in the config directory
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
                    mask = Cuts.random_masking(data[no_overlap]) # standard mask. True if 'inside' the masked region
                    
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
    """Container for imaging properties defined per patch　in a single tract.

    Attributes
    ------------------------------------------------------
    patch: array. or None
        patch ID. must be unique within each tract

    skymap_id: array or None
        tract ID + patch ID. uniquestion throughout observation
        
    property: pandas DataFrame or None
        imaging property for each patch. The columns should be specified in the property_list
    Methods
    ------------------------------------------------------
    load_patches(tract, config, property_list):
        Load patches and corresponding imaging properties
        The imaging properties should be specified in the property_list parameter
    """
    
    def __init__(self):
        self.patch = None
        self.skymap_id = None
        self.property = None
    
    def load_patches(self, tract, config, property_list = ['gseeing', 'rseeing', 'iseeing', 'zseeing', 'yseeing', 'g_depth', 'r_depth', 'i_depth', 'z_depth', 'y_depth']):
        path = os.path.join(config["hsc"]["output_dir"],"patchqa", "tract")
        filename = os.path.join(path, f"{tract}.fits")
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
                    properties['patch'] = self.patch
                    self.property = pd.DataFrame(properties)

class TractPatch(object):
    """Container for tract patch information of a specified field

    Attributes
    ------------------------------------------------------
    field: string or None
        name of the considered field
        
    data: dictionary
        output of load_patch(field) function. Includes the information about the tract, patch polygon
        
    ------------------------------------------------------
    get_tract():
        returns the whole tract included in the condsidered field
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
        autumn_tract = autumn.get_tract()
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
