import numpy as np
from astropy.io import fits
from astropy.table import Table, vstack, join
import healpy as hp
import os
from multiprocessing import Pool
import astropy.io.ascii as ascii
from functools import partial
import pandas as pd

from pfsimaging import Loader as loader

field = ['AEGIS', 'autumn', 'hectomap', 'spring']
property_name = ['gseeing', 'rseeing', 'iseeing', 'zseeing', 'yseeing', 'g_depth', 'r_depth', 'i_depth', 'z_depth', 'y_depth']

nside = 256
area = hp.nside2pixarea(nside,degrees=True)

##############################################################################################################
def get_property_all(tractpatch, dustmap, config):
    """function to get the properties of healpixels in a single field (AEGIS, autumn, hectomap or spring)

    Parameter
    ------------------------------------------
    tractpatch: instance
    instance of tractpatch class. Includes the information about the tract and patch included in the field considered.

    tractlist: array
    array of tract that would included in the output table
    
    dustmap: list
    dust map names to be applied ["desi", "desi-csfd", "csfd"]
    
    config: dictionary
    dictionary of path read from the config.yaml file in the config directory

    output
    ------------------------------------------
    t:table
    table  of imaging properties for each healpixels in the field
    column: healpix, {g,r,i,z,y}seeing, {g,r,i,z,y}_depth, extinction, target, star, area, total
    """
    tractlist = tractpatch.get_tract()
    
    func = partial(get_property_tract, config = config)
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

            #When healpixels are overlapping between different tracts, take the area weighted average between the effective overlapping area for all of the tracts 
            all_columns = property_name
            properties = all_property.groupby('healpix').apply(lambda x: pd.Series(
                {col: np.sum(x[col] * x['eff_area']) / np.sum(x['eff_area']) for col in all_columns} |  # Seeing, depth
                {'area': np.sum(x['eff_area']) / np.sum(x['total']) * area}|
                #{'star': np.log10(np.sum(x['star'])/(np.sum(x['eff_area'] / np.sum(x['total']) * area)))} |
                {'total': np.sum(x['total'])})).reset_index()
            
            print(f'adding dust extinction for {dustmap}')
            properties = add_ext(properties, dustmap)
            
            print('adding target density')
            properties = add_target_density(config, properties)
            return property
        else:
            # return table if empty
            return pd.DataFrame()
    
    

def get_property_tract(tract,  config):
    """function to get the property of the healpix within a single tract

    Parameters
    -------------------------------------------------
    tract:int
    ID of tract considered
    
    config: dictionary
    dictionary of path read from the config.yaml file in the config directory

    Output
    ------------------------------------------------
    properties:dataframe
    dataframe of imaging properties for each healpixels in the tract
    includes seeing, depth, stellar density, target density and extinction
    """
    print(f'start tract {tract}')
    
    all_ra, all_dec, all_patch, all_ID = loader.load_random_all(tract, config)
    #healpix of the all randoms
    all_healpix = hp.ang2pix(nside=nside, theta=all_ra, phi=all_dec, lonlat=True) #entire healpix in the tract
    all_healpix = to_little_endian(all_healpix)
    all_patch = to_little_endian(all_patch)
    
    randoms = loader.Random(tract, config)
    randoms.load_bsmask(config)
    randoms.load_bgmask(config)
    
    random_mask = randoms.mask # true if "inside" masked region
    if(random_mask is None) or (np.sum(random_mask) == len(random_mask)):
        print(f"No effective observation area in tract {tract}")
        return None
    random_ID = randoms.objectID[~random_mask] #objectID of the randoms that were outside masks
    
    df = pd.DataFrame({
        'healpix': all_healpix,
        'mask': np.isin(all_ID, random_ID), #true if 'outside' masks
        'patch': all_patch,
    })

    properties = (
        df.groupby('healpix')
        .agg(
            total=('mask', 'size'),   # total random count before masking
            eff_area=('mask', 'sum')    # random count after masking
        )
        .reset_index()
    )

    patch_prop = band_property(tract, config, df)
    if patch_prop is None:
        return None
    properties = pd.merge(properties, patch_prop, on='healpix', how='left')
    
    #############star file name
    #print(f'adding stellar density on {tract}')
    #properties = add_star_count(properties, tract,config)
    return properties

def to_little_endian(arr):
    dt = arr.dtype
    if dt.byteorder == '>':  # ONLY big endian
        return arr.byteswap().view(dt.newbyteorder('='))
    return arr

def band_property(tract, config, df):
    """
    function to add the g,r,i,z,y-depth and seeing of each healpixels
    
    Parameter
    ------------------------------------------
    tract: int
    ID of tract considered
    
    df: PdDataframe
    dataframe including the randoms in the concidered tract
    column: 'healpix', 'mask', 'patch'
    1. healpix: which healpix the random is included in
    2. mask: true if outside mask
    3. patch: which patch the random is included in

    output
    ------------------------------------------
    t:PdDataframe
    dataframe  of imaging properties for each randoms points
    defined by cross matching patch ID
    """
    patches = loader.Patches()
    patches.load_patches(tract, config, property_name)
    Property = patches.property
    
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
    """
    function to read in the dust map file and get E(B-V) for each healpixel
    Parameter
    ------------------------------------------
    tract: int
    ID of tract considered

    patches: instance
    instance of load.patch class. Includes the {g,r,i,z,y}-depth and {g,r,i,z,y}-seeing defined for each patch
    
    df: PdDataframe
    dataframe including the randoms in the concidered tract
    column: 'healpix', 'mask', 'patch'
    healpix: which healpix the random is included in
    mask: true if outside mask
    patch: which patch the random is included in

    output
    ------------------------------------------
    properties: PdDataframe
    dataframe  of imaging properties for each healpix ()
    
    """
    #desi dust map
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
    
    #desi csfd matched dust map
    elif dust=='csfd_desi':
        filename = "/lustre/work/jingjing.shi/pfs_co_fa/data_raw/dustmaps/CSFD_DESI_merged_dust_map_NS2048_ring--Equatorial.fits"
        hdul = fits.open(filename)
        data = hdul[1].data
        df_csfd_desi = data['EBV_CSFD_DESI_merged_at_1deg']
        hdul.close()
    
        _nside = hp.get_nside(df_csfd_desi)

        ebv = hp.ud_grade(df_csfd_desi, nside)
        properties['csfd_desi_extinction'] = ebv[properties['healpix']]
        hdul.close()
        
    #csfd dust map
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
    properties: pd dataframe with imaging properties of healpixels
    with 'star' the total stellar count
    """
    
    star = loader.Star(tract, config)
    star.load_bsmask(config)
    star.load_bgmask(config)
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
def get_imaging_property(config, tractlist = None, dustmap = 'desi'):
    """function to calculate the imaging systematics and target density for each healpixel

    Parameters
    --------------------------------------------------------------------------------------
    config: dictionary
    dictionary of path read from the config.yaml file in the config directory
    
    tractlist: list 
    list of tracts that would be included in the output table
    
    dustmaps: list
    dust map names to be applied ["desi", "desi-csfd", "csfd"]

    Output
    --------------------------------------------------------------------------------------
    autumn_property, spring_property: table
    column: healpix, {g,r,i,z,y}seeing, {g,r,i,z,y}_depth, extinction, target, star, area, total
    """
###################################################################    
    #if no tracts, all tract in HSC database will be downloaded

    if (tractlist is None):
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
    autumn_property = get_property_all(autumn, tractlist, dustmap, config)
    if autumn_property.empty:
        print(f"no tract in autumn field")
    else:
        table1 = Table.from_pandas(autumn_property)
        table1.write(autumn_file, format='fits', overwrite=True)
        
    spring_file     =   os.path.join(directory,f'spring_property.fits')
    spring_property = get_property_all(spring, tractlist, dustmap, config)
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

def add_target_density(config, properties):
    """function to calculate target density for each healpixel

    Parameter
    ------------------------------------------------------
    targets:structured array
    output of pfstarget.isCosmology()
    
    Property: astropy Table with imaging properties of healpixels
    Must include 'healpix' and 'area' column

    Output
    ------------------------------------------------------
    properties: astropy Table with imaging properties of healpixels
    with 'star' the total stellar count
    """
    if ('area' in properties.columns) and ('healpix' in properties.columns):
        Target = loader.Target(config)
        Target.load_bsmask(config)
        Target.load_bgmask(config)
        target_ra = Target.ra
        target_dec = Target.dec
        target_mask = ~ Target.mask #True if outside the bright stellar or bright galaxy mask
        healpix = hp.ang2pix(nside, target_ra[target_mask], target_dec[target_mask], nest=False, lonlat=True)
        
        # count the number of galaxies in each healpix
        _healpy, counts = np.unique(healpix, return_counts=True)
        data1 = pd.DataFrame({'healpix':_healpy, 'target' : counts})
        
        merged = pd.merge(properties, data1, on='healpix', how='left')
        
        merged = merged.fillna({'target': 0})
        merged['target'] /= merged['area']
        return merged
    else:
        print('No area or healpix column in given data')
        return Property
    
#######################################################################
def anomaly():
    ra_deg = 163.8
    dec_deg = 0.2
    radius_deg = 0.85

    # 中心方向ベクトル
    vec = hp.ang2vec(ra_deg, dec_deg, lonlat=True)

    # 円内の healpix index
    pix = hp.query_disc(
        nside,
        vec,
        np.radians(radius_deg),
        inclusive=False,
        nest=False
    )

    return pix

def clean_pixels(table, field):
    #remove anomaly
    if(field == 'spring'):
        anomaly_pix = anomaly()
        mask = ~np.isin(table['healpix'], anomaly_pix)
    else:
        mask = np.ones_like(table['healpix'], dtype=bool) 
        
    #remove healpix with small random count
    mean_count = area * 3600.0 * 100.0
    mask &= table['total'] > mean_count / 2.0
    
    #remove healpix with small effective area
    mask &= table['area'] > 0.0
    
    #remove healpix with nan
    mask &= ~np.isnan(table.as_array().tolist()).any(axis=1)
    
    #remove healpix with no star
    mask &= mask['star'] > 0.0
    
    return table[mask]
    