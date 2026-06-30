import numpy as np
import pandas as pd
from astropy.io import fits

from pfstarget import cuts as Cuts
import os
from glob import glob

import re

"""
data containerの決め事
Target, Star, Randomはinitの段階ではmain observationでかかったmaskをかける
maskに入るのは追加のmask、つまりbgmaskとbsmask

effective area calculation用のtotal randomは別でloadする
"""
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
    load_targets(self, config):
        Load all cosmology targets. config is a dictionary of path read from the config.yaml file in the config directory
        
    load_targets(tractlist, config):
        Load cosmology targets and corresponding bright-star mask
        for all tract included in tractlist. 
    """
    def __init__(self, config):
        self.ra, self.dec, self.tract, self.patch, self.objectID = load_targets(config)
        self.mask = np.zeros(len(self.ra), dtype=bool)
        
    def load_bsmask(self, config):
        bsmask_file = os.path.join(config["bsmask"]["target_bsmask"] , "s23b_ssp_co_targets_bsmask.fits")
        if os.path.exists(bsmask_file):
            with fits.open(bsmask_file) as hdu:
                data = hdu[1].data   
                ID = data['id']
                halo = data['halo']
                ghost = data['ghost']
                blooming = data['blooming']
                
            # bright star mask
            bsmask = halo | ghost | blooming

            mask = match_bsmask(self.objectID, ID, bsmask) #true if inside mask
            self.mask |= mask
            
        else:
            print(f"does not have {bsmask_file}. Run GenerateStarMask.py")
            
    def load_bgmask(self, config):
        bgmask_file = os.path.join(config["bgmask"]["target_bgmask"] , "s23b_ssp_co_targets_bgmask.fits")
        if os.path.exists(bgmask_file):
            with fits.open(bgmask_file) as hdu:
                data = hdu[1].data
            targetId = data['object_id']
            mask = ~np.isin(self.objectID, targetId) #true if inside bgmask
            self.mask |= mask
        else:
            print(f"does not have {bgmask_file}.")
        
        
    def get_tract(self, tract):
        inTract =  (self.tract == tract)
        return self.ra[inTract], self.dec[inTract], self.patch[inTract], self.mask[inTract], self.objectID[inTract]
        
def load_targets(config):
    filename = config["data"]["target_file"]
    
    if os.path.exists(filename):
        with fits.open(filename) as hdu:
            data = hdu[1].data   
            ra = data['ra']%360
            dec = data['dec']
            patch = data['patch']
            tract = data['tract']
            ID = data['object_id']
        return ra, dec, tract, patch, ID
    else:
        print(f'cannot open {filename}')
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
    def __init__(self, field, config):
        self.ra, self.dec, self.objectID = load_stars(field, config)
        
        
def load_stars(field, config):
    directory = os.path.join(config["Gaia"]["star"], field)
    star_files = glob(os.path.join(directory, "*.fits"))
    
    ra = []
    dec = []
    objectId = []
    
    if len(star_files)==0:
        print("Gaia stars not downloaded yet. Run bin/get_gaia_stars.py")
        return None, None, None
    
    else:
        for file in star_files:
            with fits.open(file) as hdu:
                data = hdu[1].data
            _ra = data['ra']
            _dec = data['dec']
            _objectId = data['source_id']
            
            ra.append(_ra)
            dec.append(_dec)
            objectId.append(_objectId)
            
        ra = np.concatenate(ra)
        dec = np.concatenate(dec)
        objectId = np.concatenate(objectId)

        # get only unique sources
        _, idx = np.unique(objectId, return_index=True)
        dx = np.sort(idx)

        return ra[idx], dec[idx], objectId[idx]
            
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
    
    def __init__(self, tract, config):
        self.tract = tract
        self.ra, self.dec, self.patch, self.objectID = load_random(tract, config)
        if(self.ra is None):
            self.mask = None
        else:
            self.mask = np.zeros_like(self.ra, dtype=bool)
            
    def load_bsmask(self, config):
        maskfile = os.path.join(config["bsmask"]["random_bsmask"], f"{self.tract}.fits")
        if os.path.exists(maskfile):
            with fits.open(maskfile) as hdu:
                if len(hdu)==2:
                    data = hdu[1].data
                    bsmaskID = data['id']
                    bsmask = data['halo'] | data['ghost'] | data['blooming'] #bright star mask. true if 'inside' mask
                    self.mask |= match_bsmask(self.objectID, bsmaskID, bsmask)
                else:
                    print(f'{maskfile} does not have objects')
        else:
            print(f'cannot open {maskfile}')
            
    def load_bgmask(self, config):
        bgmask_file = os.path.join(config["bgmask"]["random_bgmask"] , f"{self.tract}_bgmask.fits")
        if os.path.exists(bgmask_file):
            with fits.open(bgmask_file) as hdu:
                data = hdu[1].data
            targetId = data['object_id']
            mask = np.isin(self.objectID, targetId) #true if inside bgmask
            self.mask |= mask
        else:
            print(f"does not have {bgmask_file}.")
        
def load_random(tract, config):
    filename = os.path.join(config["data"]["random_dir"], f"{tract}.fits")
    if os.path.exists(filename):
        with fits.open(filename) as hdu:
            if len(hdu)==2:
                data = hdu[1].data
                #no_overlap = data['detect_ispatchinner'] & data['detect_istractinner']
                patch = data['patch']
                ra = data['ra']
                dec = data['dec']
                objectID = data['object_id']
                mask = mask_random(data) # standard mask. True if 'inside' the masked region
                return ra[~mask], dec[~mask], patch[~mask], objectID[~mask]
            else:
                print(f'{filename} does not have data')
    else:
        print(f'cannot open {filename}')
    return None, None, None, None

def mask_random(data):
    mask = (data['g_mask_brightstar_halo']  |
            data['r_mask_brightstar_halo']  |
            data['i_mask_brightstar_halo']  |
            data['z_mask_brightstar_halo']  |
            data['g_mask_brightstar_ghost'] |
            data['r_mask_brightstar_ghost'] |
            data['i_mask_brightstar_ghost'] |
            data['z_mask_brightstar_ghost']
           )
    return mask
        
        
def load_random_all(tract, config):
    filename = os.path.join(config["data"]["random_dir_masked"], f"{tract}.fits")
    if os.path.exists(filename):
        with fits.open(filename) as hdu:
            if len(hdu)==2:
                data = hdu[1].data
                ra = data['ra']
                dec = data['dec']
                patch = data['patch']
                ID = data['object_id']
            else:
                print(f'{filename} does not have data')
                ra = np.array([])
                dec = np.array([])
                patch = np.array([])
                ID = np.array([])
    else:
        print(f'cannot open {filename}')
        ra = np.array([])
        dec = np.array([])
        patch = np.array([])
        ID = np.array([])
        
    filename = os.path.join(config["data"]["random_dir"], f"{tract}.fits")
    if os.path.exists(filename):
        with fits.open(filename) as hdu:
            if len(hdu)==2:
                data = hdu[1].data
                ra = np.append(ra, np.array(data['ra']))
                dec = np.append(dec, np.array(data['dec']))
                patch = np.append(patch, np.array(data['patch']))
                ID = np.append(ID, np.array(data['object_ID']))
            else:
                print(f'{filename} does not have data')
    else:
        print(f'cannot open {filename}')
    return ra, dec, patch, ID


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
        path = os.path.join(config["data"]["patchqa_dir"], "tract")
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
                    
def match_bsmask(inputID, bsmaskID, bsmask):
    # id -> mask dictionary
    mask_dict = dict(zip(bsmaskID, bsmask))
    # self.object_id order
    mask = np.array(
        [mask_dict.get(objid, False) for objid in inputID],
        dtype=bool
    )
    return mask

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
