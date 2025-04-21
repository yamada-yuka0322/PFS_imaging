import re
import numpy as np
from astropy.io import fits
from astropy.table import Table
import healpy as hp
import os
from multiprocessing import Pool
import astropy.io.ascii as ascii
from astropy.coordinates import Galactic, FK5
import astropy.units as u
from functools import partial
import pandas as pd

from Loader import *

field = ['AEGIS', 'autumn', 'hectomap', 'spring']
stardict = {'star_default':'star'}

property_name = ['gseeing', 'rseeing', 'iseeing', 'zseeing', 'yseeing', 'g_depth', 'r_depth', 'i_depth', 'z_depth', 'y_depth']

args    =   None
diffver =   '-colorterm'
datapath = f"/data/PFS/s23{diffver}"
nside = 1024
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

    file_path = f"/PFS_imaging/Field/tracts_patches_W-{field}.txt"
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
def get_property_all(tractpatch, target_healpix, tractlist, patches):
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
            func = partial(get_property_tract, patches=patches)
            results = pool.map(func, tractlist)
        
        # Exclude None results
        valid_results = [res for res in results if res is not None]

        if valid_results:
            all_property = pd.concat(valid_results, ignore_index=True)

            #When healpixels are overlapping between different tracts, take the averaage weighted by the effective overlapping area for all of the tracts 
            stars = list(stardict.values())
            all_columns = property_name + ['extinction']
            property = all_property.groupby('healpix').apply(lambda x: pd.Series(
                {col: np.sum(x[col] * x['Mask']) / np.sum(x['Mask']) for col in all_columns} |  # Seeing, depth, extinction
                {'area': np.sum(x['Mask']) / np.sum(x['counts']) * area}|
                {col: np.sum(x[col])/area for col in stars})).reset_index()

            u, counts = np.unique(target_healpix, return_counts=True)
            data1 = pd.DataFrame({
                'healpix':u,
                'target':counts
            })

            property = pd.merge(property, data1, on='healpix', how='left')
            property['target'] = property['target']/property['area']

            return property
        else:
            # return table if empty
            return pd.DataFrame()
    
    

def get_property_tract(tract, patches):
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
    random = Random()
    random.load_random(datapath, tract)
    ra = random.ra
    dec = random.dec
    patch = random.patch
    mask = random.mask #within mask
    
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
    print(f"random distribution ra max:{np.max(_ra)} min:{np.min(_ra)}, dec  max:{np.max(_dec)} min:{np.min(_dec)}")

    
    random_data = pd.DataFrame({'patch':_patch, 'healpix':_healpy})
    
    property = patches.get_properties(tract)
    # Check for empty dataframe
    if property.empty:
        # return None if empty
        return None

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
    properties = add_ext(properties)
    #############star file name
    properties = add_star_count(properties, stardict, tract)
    return properties

def get_patch_property(patch_id, tract):
    """function to connect the path_id with the patch properties

    Parameters
    ------------------------------------------------------
    patch_id:array
    array of the patch id array([508,706,102,108,8,700,601,...)

    tract:int
    tract number of the tract considered

    Output
    -------------
    result:dataframes
    {patch:patch_id, 
    gseeing: gseeing from patch_sql , 
    rseeing: rseeing from patch_sql ,
    iseeing: iseeing from patch_sql ,
    zseeing: zseeing from patch_sql ,
    yseeing: yseeing from patch_sql ,
    g_depth: g_depth from patch_sql ,
    r_depth: r_depth from patch_sql ,
    i_depth: i_depth from patch_sql ,
    z_depth: z_depth from patch_sql ,
    y_depth: y_depth from patch_sql }
    """
    patch_sql = args.patch_sql
    name = patch_sql[0].split('.')[0]
    prefix2 = f'database/{name}/tracts_{name}'
    filename = os.path.join(prefix2,'%s_bias.fits' %(tract))
    # Read imaging properties of each patch
    if os.path.exists(filename):
        with fits.open(filename) as hdu:
            data = hdu[1].data
            df = pd.DataFrame(data)
            df = df.apply(lambda col: col.values.byteswap().newbyteorder() if col.dtype.byteorder == '>' else col)
            #patch_id of randomly distributed points
            patch_df = pd.DataFrame({'patch': patch_id})

            result = pd.merge(patch_df, df, on='patch', how='left')

    else:
        print(f'No observation in tract {tract}')
        result = pd.DataFrame(columns=['patch'])

        
    
    return result

#def add_eff_area(properties):
    
    #result = pd.merge(table1, properties, on='healpix', how='inner')
    #return result
    
def add_ext(properties):
    dustfile = '/pfstarget/src/pfstarget/dat/desi_dust_gr_512.fits'
    with fits.open(dustfile) as hdu:
        data = hdu[1].data
        healpix = data["HPXPIXEL"]
        EBV = data["EBV_GR"]
        
    npix = hp.nside2npix(512)
    ebv_map = np.zeros(npix, dtype=np.float32)
    
    ebv_map[healpix] = EBV
    ebv = hp.ud_grade(ebv_map, nside)
    properties['extinction'] = ebv[properties['healpix']]
    
    return properties
    
def old_add_ext(properties):
    """function to add extinction column to pd properties

    Parameter
    ------------------------------------------------------
    properties: pd dataframe with imaging properties of healpixels
    Must include healpix column

    Output
    ------------------------------------------------------
    properties: pd dataframe with imaging properties of healpixels
    """
    with fits.open('csfd_ebv.fits') as hdu:
        data = hdu[1].data
        im=data["T"]
    #healpix in galactic coordinate
    m = np.ndarray.flatten(im)
    _nside = hp.get_nside(m)
    
    # center coordinate of HEALPix (Galactic theta:co-latitude(rad), phi:longitude(rad))
    Npix = hp.nside2npix(_nside)
    theta, phi = hp.pix2ang(_nside, np.arange(Npix))
    l = np.degrees(phi)  # Galactic longitude in degrees
    b = 90 - np.degrees(theta)  # Galactic latitude in degrees

    # Galactic -> Equatorial (RA, Dec)
    galactic_coords = Galactic(l * u.deg, b * u.deg)
    equatorial_coords = galactic_coords.transform_to(FK5(equinox='J2000'))

    ra = equatorial_coords.ra.deg
    dec = equatorial_coords.dec.deg

    # Equatorial HEALPix インデックスを取得
    new_pix = hp.ang2pix(_nside, np.radians(90 - dec), np.radians(ra), nest=False)

    # Equatorial map of the dust extinction
    equatorial_map = np.full(Npix, hp.UNSEEN)
    equatorial_map[new_pix] = m

    
    MAG = hp.ud_grade(equatorial_map, nside)

    healpix = properties['healpix']
    a = MAG[healpix]
    properties['extinction'] = a
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
        star = Star(sql)
        star.load_stars(datapath, tract)
        
        ra = star.ra
        dec = star.dec
        
        edges = (ra<np.max(ra) - 0.1)&(ra>np.min(ra) + 0.1)&(dec<np.max(dec) - 0.1)&(dec>np.min(dec) + 0.1)
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
def imaging_bias(object, tractlist = ''):
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

    if (tractlist==''):
        tractname= 'TractInfoS23.csv'
        tracts      =   ascii.read(tractname)['tract']
        tractlist = list(tracts)

####################################################################
    #tract_patch 
    autumn = TractPatch("autumn")
    AEGIS = TractPatch("AEGIS")
    hectomap = TractPatch("hectomap")
    spring = TractPatch("spring")
    
####################################################################
    #load entire patch property
    patches = Patches()
    patches.load_patches(datapath, property_name)
####################################################################
    target = object
    target_ra = target["RA"]
    target_dec = target["DEC"]
    
    print(f"target distribution ra max:{np.max(target_ra)} min:{np.min(target_ra)}, dec  max:{np.max(target_dec)} min:{np.min(target_dec)}")
    
    target_healpix = hp.pixelfunc.ang2pix(nside=nside, theta=target_ra, phi=target_dec, lonlat=True)
    
    #Get Property for each healpixel
    directory = '/output/PFS/property'
    os.makedirs(directory, exist_ok=True)

    autumn_file     =   os.path.join(directory,'autumn_property.fits')
    autumn_property = get_property_all(autumn, target_healpix, tractlist, patches)
    if autumn_property.empty:
        print(f"no tract in autumn field")
    else:
        table = Table.from_pandas(autumn_property)
        table.write(autumn_file, format='fits', overwrite=True)

    AEGIS_file     =   os.path.join(directory,'AEGIS_property.fits')
    AEGIS_property = get_property_all(AEGIS, target_healpix, tractlist, patches)
    if AEGIS_property.empty:
        print(f"no tract in AEGIS field")
    else:
        table = Table.from_pandas(AEGIS_property)
        table.write(AEGIS_file, format='fits', overwrite=True)
        
    hectomap_file     =   os.path.join(directory,'hectomap_property.fits')
    hectomap_property = get_property_all(hectomap, target_healpix, tractlist, patches)
    if hectomap_property.empty:
        print(f"no tract in hectomap field")
    else:
        table = Table.from_pandas(hectomap_property)
        table.write(hectomap_file, format='fits', overwrite=True)
        
    spring_file     =   os.path.join(directory,'spring_property.fits')
    spring_property = get_property_all(spring, target_healpix, tractlist, patches)
    if spring_property.empty:
        print(f"no tract in spring field")
    else:
        table = Table.from_pandas(spring_property)
        table.write(spring_file, format='fits', overwrite=True)
    
    return autumn_property, AEGIS_property, hectomap_property, spring_property

######################################################################
