import re
import numpy as np
from astropy.io import fits
from astropy.table import Table
import healpy as hp
import os
from multiprocessing import Pool
import astropy.io.ascii as ascii
from functools import partial
import pandas as pd

from . import Loader as loader

field = ['AEGIS', 'autumn', 'hectomap', 'spring']
stardict = {'star_default':'star'}
#stardict = {}
property_name = ['gseeing', 'rseeing', 'iseeing', 'zseeing', 'yseeing', 'g_depth', 'r_depth', 'i_depth', 'z_depth', 'y_depth']

args    =   None
diffver =   '-colorterm'

path = "/home/YukaYamada/data/PFS" #これどうしよう
datapath = f"{path}/s23{diffver}"
nside = 256
area = hp.nside2pixarea(nside,degrees=True)

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

##############################################################################################################
def get_property_all(tractpatch, tractlist, dustmap):
    """function to get the properties of healpixels in a single field (AEGIS, autumn, hectomap or spring)

    Parameter
    ------------------------------------------
    tractpatch: instance
    instance of tractpatch class. Includes the information about the tract and patch included in the field considered.

    target_healpix: np array
    array of target healpixels

    output
    ------------------------------------------
    t:table
    table  of imaging properties for each healpixels in the field
    includes seeing, depth, stellar density, target density and extinction
    """


    tractall = np.array(tractpatch.get_tract())
    tractlist = np.array(tractlist)
    
    tractlist = tractlist[np.isin(tractlist, tractall)]
    if len(tractlist)==0:
        print(f"No tracts in {tractpatch.field}")
        return pd.DataFrame()
    else:
        with Pool(processes=20) as pool:  # Adjust number of processes based on your CPU
            func = partial(get_property_tract, dustmap=dustmap)
            results = pool.map(func, tractlist)
        
        # Exclude None results
        valid_results = [res for res in results if res is not None]

        if valid_results:
            all_property = pd.concat(valid_results, ignore_index=True)

            #When healpixels are overlapping between different tracts, take the averaage weighted by the effective overlapping area for all of the tracts 
            stars = list(stardict.values())
            all_columns = property_name + [dust+'_extinction' for dust in dustmap]
            property = all_property.groupby('healpix').apply(lambda x: pd.Series(
                {col: np.sum(x[col] * x['Mask']) / np.sum(x['Mask']) for col in all_columns} |  # Seeing, depth, extinction
                {'area': np.sum(x['Mask']) / np.sum(x['counts']) * area}|
                {col: np.sum(x[col])/area for col in stars})).reset_index()
            return property
        else:
            # return table if empty
            return pd.DataFrame()
    
    

def get_property_tract(tract, dustmap):
    """function to get the property of the healpix within a tract

    Parameters
    -------------------------------------------------
    tract:int
    number of tract considered
    
    tractpatch_dict: dictionary
    dictionary including the tract patch information]

    Output
    ------------------------------------------------
    properties:dataframe
    dataframe of imaging properties for each healpixels in the tract
    includes seeing, depth, stellar density, target density and extinction
    """
    random = loader.Random()
    random.load_random(datapath, tract)
    ra = random.ra
    dec = random.dec
    patch = random.patch
    mask = random.mask #within mask
    if ra is None:
        print(f"Can not find random data in {datapath} ")
        return None
    
    edges = (ra<np.max(ra) - 0.1)&(ra>np.min(ra) + 0.1)&(dec<np.max(dec) - 0.1)&(dec>np.min(dec) + 0.1)
    ra = ra[edges]
    dec = dec[edges]
    patch = patch[edges]
    mask = mask[edges]
    
    out_mask = ~mask
    
    healpy = hp.ang2pix(nside=nside, theta=ra, phi=dec, lonlat=True) #entire healpix in the tract
    
    _ra = ra[out_mask]
    _dec = dec[out_mask]
    _patch = patch[out_mask]
    _healpy = healpy[out_mask] #healpix outside the stellar mask
    #print(f"random distribution ra max:{np.max(_ra)} min:{np.min(_ra)}, dec  max:{np.max(_dec)} min:{np.min(_dec)}")
    if (len(_patch)>0):
        if isinstance(_patch, np.ndarray) and _patch.dtype.byteorder == '>':
            _patch = _patch.astype(_patch.dtype.newbyteorder('='))
        if isinstance(_healpy, np.ndarray) and _healpy.dtype.byteorder == '>':
            _healpy = _healpy.astype(_healpy.dtype.newbyteorder('='))
        random_data = pd.DataFrame({'patch':_patch, 'healpix':_healpy})
    else:
        print(f"All observation area of tract {tract} is inside the bright stellar mask")
        return None
    
    patches = loader.Patches()
    patches.load_patches(datapath, property_name, tract)
    property = patches.property
    # Check for empty dataframe
    if property is None:
        print(f"patch column missing in tract {tract} property")
        return None
    else:
        property['patch'] = patches.patch

    properties = pd.merge(random_data, property, on='patch', how='left')
    # take the average of the properties for the random points between all the overlapping patches
    properties = properties.groupby('healpix')[property_name].mean().reset_index()
    
    u, counts = np.unique(healpy, return_counts=True)

    data1 = {
        'healpix':u,
        'counts':counts
    }
    table1 = pd.DataFrame(data1)
    properties = pd.merge(properties, table1, on='healpix', how='left')

    u, counts = np.unique(_healpy, return_counts=True)

    data1 = {
        'healpix':u,
        'Mask':counts
    }
    table1 = pd.DataFrame(data1)
    properties = pd.merge(properties, table1, on='healpix', how='left')
    
    #properties = add_eff_area(mask, properties)
    for dust in dustmap:
        properties = add_ext(properties, dust)
    #############star file name
    properties = add_star_count(properties, stardict, tract)
    return properties
    
def add_ext(properties, dust):
    if dust=='desi':
        dustfile = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'dat', 'desi_dust_gr_512.fits')
        with fits.open(dustfile) as hdu:
            data = hdu[1].data
            healpix = data["HPXPIXEL"]
            EBV = data["EBV_GR"]
        
        npix = hp.nside2npix(512)
        ebv_map = np.zeros(npix, dtype=np.float32)
    
        ebv_map[healpix] = EBV
        ebv = hp.ud_grade(ebv_map, nside)
        properties['desi_extinction'] = ebv[properties['healpix']]
    
    elif dust=='desi-csfd':
        filename = "/lustre/work/jingjing.shi/pfs_co_fa/data_raw/dustmaps/CSFD_DESI_merged_dust_map_NS2048_ring.fits"
        hdul = fits.open(filename)
        df_csfd_desi = hdul[1].data
    
        _nside = hp.get_nside(df_csfd_desi['EBV_CSFD_DESI_merged_at_1deg'])
        ind_galactic = np.arange(hp.get_map_size(df_csfd_desi['EBV_CSFD_DESI_merged_at_1deg']))
        theta_galactic, phi_galactic = hp.pix2ang(_nside, ind_galactic)
        r = hp.Rotator(coord=["C", "G"])
        theta_equatorial, phi_equatorial = r(theta_galactic, phi_galactic)
        ind_equatorial = hp.ang2pix(_nside, theta_equatorial, phi_equatorial)
        
        ebv_map = df_csfd_desi['EBV_CSFD_DESI_merged_at_1deg'][ind_equatorial]
        ebv = hp.ud_grade(ebv_map, nside)
        properties['desi-csfd_extinction'] = ebv[properties['healpix']]
        hdul.close()
        
    elif dust=='csfd':
        filename = "/lustre/work/jingjing.shi/pfs_co_fa/data_raw/dustmaps/CSFD_DESI_merged_dust_map_NS2048_ring.fits"
        hdul = fits.open(filename)
        df_csfd_desi = hdul[1].data
    
        _nside = hp.get_nside(df_csfd_desi['EBV_CSFD'])
        ind_galactic = np.arange(hp.get_map_size(df_csfd_desi['EBV_CSFD']))
        theta_galactic, phi_galactic = hp.pix2ang(_nside, ind_galactic)
        r = hp.Rotator(coord=["C", "G"])
        theta_equatorial, phi_equatorial = r(theta_galactic, phi_galactic)
        ind_equatorial = hp.ang2pix(_nside, theta_equatorial, phi_equatorial)
        
        ebv_map = df_csfd_desi['EBV_CSFD'][ind_equatorial]
        ebv = hp.ud_grade(ebv_map, nside)
        properties['csfd_extinction'] = ebv[properties['healpix']]
        hdul.close()
        
    else:
        print(f'No dust file corresponding to {dust}')
    
    return properties
    

def add_star_count(properties, stardict, tract):
    """function to add stellar counts to pd properties

    Parameter
    ------------------------------------------------------
    properties: pd dataframe with imaging properties of healpixels
    Must include healpix column

    Output
    ------------------------------------------------------
    table2: pd dataframe with imaging properties of healpixels
    """
    for sql, name in stardict.items():
        star = loader.Star(sql)
        star.load_stars(datapath, tract)
        
        ra = star.ra
        dec = star.dec
        
        if ra is None:
            print(f'No stars in tract {tract}')
            properties[name] = np.zeros(len(properties['healpix']))
        else:
            if (np.max(ra) - np.min(ra) < 180):
                edges = (ra<np.max(ra) - 0.1)&(ra>np.min(ra) + 0.1)&(dec<np.max(dec) - 0.1)&(dec>np.min(dec) + 0.1)
            else:
                dec_edge = (dec<np.max(dec) - 0.1)&(dec>np.min(dec) + 0.1)
                _ra = ra-360*(ra>180)
                ra_edge = (_ra<np.max(_ra)-0.1)&(_ra>np.min(_ra) + 0.1)
                edges = dec_edge & ra_edge
                
            ra = ra[edges]
            dec = dec[edges]
            
            healpy = hp.ang2pix(nside=nside, theta=ra, phi=dec, lonlat=True)
        
            _healpy, counts = np.unique(healpy, return_counts=True)
            data1 = {
                'healpix':_healpy,
                name : counts
            }
            table1 = pd.DataFrame(data1)
            properties = pd.merge(properties, table1, on='healpix', how='left')
    
    return properties
    
###########################################################################################################
def imaging_bias(tractlist = '', dustmaps = ['desi'], directory='../property/', savename=''):
    """function to calculate the imaging systematics and target density for each healpixel

    Parameters
    --------------------------------------------------------------------------------------
    object: structured numpy array of HSC objects with relevant columns
        for target selection

    selection: bool array of target galaxies

    keys: dictionary to download from HSC database

    Output
    --------------------------------------------------------------------------------------
    autumn_property, AEGIS_property, hectomap_property, spring_property: table with column: healpix, {g,r,i,z,y}seeing,
    {g,r,i,z,y}_depth, extinction, target, star, area
    """
###################################################################    
    #if no tracts, all tract in HSC database will be downloaded

    if (type(tractlist) is str):
        tractname = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'dat', 'TractInfoS23.csv')
        print(tractname)
        tracts      =   ascii.read(tractname)['tract']
        tractlist = np.array(tracts)

####################################################################
    #tract_patch 
    autumn = TractPatch("autumn")
    spring = TractPatch("spring")
    #patches.load_patches(datapath, property_name)
####################################################################
    #Get Property for each healpixel
    os.makedirs(directory, exist_ok=True)

    autumn_file     =   os.path.join(directory,f'autumn_property{savename}.fits')
    autumn_property = get_property_all(autumn, tractlist, dustmaps)
    if autumn_property.empty:
        print(f"no tract in autumn field")
    else:
        table = Table.from_pandas(autumn_property)
        table.write(autumn_file, format='fits', overwrite=True)
        
    spring_file     =   os.path.join(directory,f'spring_property{savename}.fits')
    spring_property = get_property_all(spring, tractlist, dustmaps)
    if spring_property.empty:
        print(f"no tract in spring field")
    else:
        table = Table.from_pandas(spring_property)
        table.write(spring_file, format='fits', overwrite=True)
    
    #return autumn_property, AEGIS_property, hectomap_property, spring_property
    return autumn_property, spring_property

######################################################################

def get_target_density(targets, Property, name = 'target'):
    if ('area' in Property.columns) and ('healpix' in Property.columns):
        target_ra = targets['RA']
        target_dec = targets['DEC']
        healpix = hp.ang2pix(nside, target_ra, target_dec, nest=False, lonlat=True)
        
        # count the number of galaxies in each healpix
        _healpy, counts = np.unique(healpix, return_counts=True)
        data1 = pd.DataFrame({'healpix':_healpy, name : counts})
        
        merged = pd.merge(Property, data1, on='healpix', how='left')
        merged = merged.fillna({name: 0})
        merged[name] /= merged['area']
        return merged
    else:
        print('No area or healpix column in given data')
        return Property
        
