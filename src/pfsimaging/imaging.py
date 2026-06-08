import re
import numpy as np
from astropy.io import fits
from astropy.table import Table, vstack
import healpy as hp
import os
from multiprocessing import Pool
import astropy.io.ascii as ascii
from functools import partial
import pandas as pd

from . import Loader as loader

field = ['AEGIS', 'autumn', 'hectomap', 'spring']
property_name = ['gseeing', 'rseeing', 'iseeing', 'zseeing', 'yseeing', 'g_depth', 'r_depth', 'i_depth', 'z_depth', 'y_depth']

nside = 256
area = hp.nside2pixarea(nside,degrees=True)

##############################################################################################################
def get_property_all(tractpatch, tractlist, dustmap, config):
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

    patches = loader.Patches()
    patches.load_patches(property_name, config)
    
    func = partial(get_property_tract, dustmap=dustmap, config = config, patches = patches)
    if len(tractlist)==0:
        print(f"No tracts in {tractpatch.field}")
        return pd.DataFrame()
    else:
        with Pool(processes=20) as pool:  # Adjust number of processes based on your CPU
            results = pool.map(func, tractlist)
        
        # Exclude None results
        valid_results = [res for res in results if res is not None]

        if valid_results:
            all_property = pd.concat(valid_results, ignore_index=True)

            #When healpixels are overlapping between different tracts, take the averaage weighted by the effective overlapping area for all of the tracts 
            all_columns = property_name + [dust+'_extinction' for dust in dustmap]
            property = all_property.groupby('healpix').apply(lambda x: pd.Series(
                {col: np.sum(x[col] * x['eff_area']) / np.sum(x['eff_area']) for col in all_columns} |  # Seeing, depth, extinction
                {'area': np.sum(x['eff_area']) / np.sum(x['total']) * area}|
                {'star': np.log10(np.sum(x['star'])/(np.sum(x['eff_area'] / np.sum(x['total']) * area)))})).reset_index()
            return property
        else:
            # return table if empty
            return pd.DataFrame()
    
    

def get_property_tract(tract, dustmap, config, patches):
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
    print(f'start tract {tract}')
    
    randoms = loader.Random()
    randoms.load_random(tract, config)
    random_ra = randoms.ra
    random_dec = randoms.dec
    random_mask = randoms.mask # true if "inside" masked region
    random_patch = randoms.patch

    #healpix全体に入ってるrandomの数
    healpix = hp.ang2pix(nside=nside, theta=random_ra, phi=random_dec, lonlat=True) #entire healpix in the tract
    healpix = to_little_endian(healpix)
    
    df = pd.DataFrame({
        'healpix': healpix,
        'mask': ~random_mask
        'patch': random_patch
    })

    properties = (
        df.groupby('healpix')
        .agg(
            total=('mask', 'size'),   # maskをかける前の総数
            eff_area=('mask', 'sum')    # mask=True の数
        )
        .reset_index()
    )

    # healpixごとに一発集計
    
    
    patch_prop = band_property(tract, patches, df)
    if patch_prop is None:
        return None
    properties = pd.merge(properties, patch_prop, on='healpix', how='left')
    
    #properties = add_eff_area(mask, properties)
    for dust in dustmap:
        print(f'added {dust} in {tract}')
        properties = add_ext(properties, dust)
    #############star file name
    print(f'adding stellar density on {tract}')
    properties = add_star_count(properties, tract,config)
    return properties

def to_little_endian(arr):
    dt = arr.dtype
    if dt.byteorder == '>':  # ONLY big endian
        return arr.byteswap().view(dt.newbyteorder('='))
    return arr

def band_property(tract, patches, df):
    """
    function to add the g,r,i,z,y-depth and seeing of each healpixels
    """
    Property = patches.get_properties(tract)
    
    if Property is None:
        print(f"patch column missing in tract {tract} property")
        return None

    df_valid = df[df['mask']] #Get randoms outside the masked area

    patch_prop = pd.merge(
        df_valid[['healpix','patch']],
        Property,
        on='patch',
        how='left'
    )

    patch_prop = patch_prop.groupby('healpix')[property_name].mean().reset_index()
    return patch_prop
    
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
        filename = "/lustre/work/jingjing.shi/pfs_co_fa/data_raw/dustmaps/CSFD_DESI_merged_dust_map_NS2048_ring--Equatorial.fits"
        hdul = fits.open(filename)
        data = hdul[1].data
        df_csfd_desi = data['EBV_CSFD_DESI_merged_at_1deg']
        hdul.close()
    
        _nside = hp.get_nside(df_csfd_desi)
        #ind_galactic = np.arange(hp.get_map_size(df_csfd_desi['EBV_CSFD_DESI_merged_at_1deg']))
        #theta_galactic, phi_galactic = hp.pix2ang(_nside, ind_galactic)
        #r = hp.Rotator(coord=["C", "G"])
        #theta_equatorial, phi_equatorial = r(theta_galactic, phi_galactic)
        #ind_equatorial = hp.ang2pix(_nside, theta_equatorial, phi_equatorial)

        ebv = hp.ud_grade(df_csfd_desi, nside)
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
    

def add_star_count(properties, tract, config):
    """function to add stellar counts to pd properties

    Parameter
    ------------------------------------------------------
    properties: pd dataframe with imaging properties of healpixels
    Must include healpix column

    Output
    ------------------------------------------------------
    table2: pd dataframe with imaging properties of healpixels
    """
    
    star = loader.Star()
    star.load_stars(tract, config)
    ra = star.ra
    dec = star.dec
    mask = star.mask # true if "inside" bright star mask
        
    if ra is None:
        print(f'No stars in tract {tract}')
        properties['star'] = np.zeros(len(properties['healpix']))
    else:
        healpy = hp.ang2pix(nside=nside, theta=ra[~mask], phi=dec[~mask], lonlat=True)
        
        _healpy, counts = np.unique(healpy, return_counts=True)
        data1 = {
            'healpix':_healpy,
            'star' : counts
        }
        table1 = pd.DataFrame(data1)
        properties = pd.merge(properties, table1, on='healpix', how='left')
        properties['star'] = properties['star'].fillna(0)
    
    return properties
    
###########################################################################################################
def get_imaging_property(tractlist = '', dustmaps = ['desi'], config):
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
        tractname = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'dat', 'Tracttest.csv')
        print(tractname)
        tracts      =   ascii.read(tractname)['tract']
        tractlist = np.array(tracts)

####################################################################
    #tract_patch 
    autumn = loader.TractPatch("autumn")
    spring = loader.TractPatch("spring")
    #patches.load_patches(datapath, property_name)
####################################################################
    #Get Property for each healpixel
    directory = config['Imaging']['imaging_dir']
    os.makedirs(directory, exist_ok=True)
    
    autumn_file     =   os.path.join(directory,f'autumn_property.fits')
    autumn_property = get_property_all(autumn, tractlist, dustmaps, config)
    if autumn_property.empty:
        print(f"no tract in autumn field")
    else:
        table1 = Table.from_pandas(autumn_property)
        table1.write(autumn_file, format='fits', overwrite=True)
        
    spring_file     =   os.path.join(directory,f'spring_property.fits')
    spring_property = get_property_all(spring, tractlist, dustmaps, config)
    if spring_property.empty:
        print(f"no tract in spring field")
    else:
        table2 = Table.from_pandas(spring_property)
        table2.write(spring_file, format='fits', overwrite=True)
        
    all_property = vstack([table1, table2])
    all_file     =   os.path.join(directory,f'all_property.fits')
    all_property.write(all_file, format='fits', overwrite=True)
    
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
    
def get_target_density1(targets, Property, name = 'target', area = 'area'):
    if (area in Property.columns) and ('healpix' in Property.columns):
        target_ra = targets['RA']
        target_dec = targets['DEC']
        healpix = hp.ang2pix(nside, target_ra, target_dec, nest=False, lonlat=True)
        
        # count the number of galaxies in each healpix
        _healpy, counts = np.unique(healpix, return_counts=True)
        data1 = pd.DataFrame({'healpix':_healpy, name : counts})
        
        merged = pd.merge(Property, data1, on='healpix', how='left')
        merged = merged.fillna({name: 0})
        merged[name] /= merged[area]
        return merged
    else:
        print('No area or healpix column in given data')
        return Property
        
