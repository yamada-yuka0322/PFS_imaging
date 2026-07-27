import os
import numpy as np
import healpy as hp
from astropy.io import fits
import astropy.io.ascii as ascii
from astropy.table import Table, vstack, join

from multiprocessing import Pool

from functools import partial

from pfsimaging import Loader as loader


field = ['AEGIS', 'autumn', 'hectomap', 'spring']
property_name = ['gseeing', 'rseeing', 'iseeing', 'zseeing', 'yseeing', 'g_depth', 'r_depth', 'i_depth', 'z_depth', 'y_depth']

nside = 256
area = hp.nside2pixarea(nside,degrees=True)

##############################################################################################################
def get_property_all(tractpatch, dustmap, config, nprocess=1, verbose=False):
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
    t : table
        table  of imaging properties for each healpixels in the field
    column: healpix, {g,r,i,z,y}seeing, {g,r,i,z,y}_depth, extinction, target, star, area, total
    """
    tractlist = tractpatch.get_tract()
    
    func = partial(get_property_tract, config=config, verbose=verbose)

    if len(tractlist)==0:
        print(f"No tracts in {tractpatch.field}")
        return None
    else:
        if verbose: print(f'multiprocessing using {nprocess} processes') 
        with Pool(processes=nprocess) as pool:  
            results = pool.map(func, tractlist)
        
        # Exclude None results
        valid_results = [res for res in results if res is not None]

        if valid_results:
            all_property = vstack(valid_results)

            # When healpixels are overlapping between different tracts
            # take the area weighted average between the effective overlapping 
            # area for all of the tracts 
            
            groups = all_property.group_by('healpix').groups

            rows = []

            for g in groups:
                row = {}
                row['healpix'] = g['healpix'][0]

                eff_area = np.asarray(g['eff_area']) #normalized effective area
                t_eff_area = np.sum(eff_area)

                # effective area weighted mean of imaging attributes
                for col in property_name:
                    x = np.asarray(g[col]) #imaging attribute
                    row[col] = np.sum(x * eff_area) / t_eff_area

                # area
                row['area'] = t_eff_area / np.sum(g['total']) * area #effective area

                # total
                row['total'] = np.sum(g['total']) #total number of randoms

                rows.append(row)

            properties = Table(rows)
            
            if verbose: print(f'adding dust extinction for {dustmap}')
            properties = add_ext(properties, dustmap)
            
            if verbose: print('adding target density')
            properties = add_target_density(config, properties)
            
            if verbose: print('adding stellar density')
            properties = add_stellar_density(config, tractpatch.field, properties)
            return properties
        else:
            # return table if empty
            return None
    
    

def get_property_tract(tract, config=None, verbose=False):
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
    if verbose: print(f'getting properties from tract {tract}')
    
    all_ra, all_dec, all_patch, all_ID = loader.load_random_all(tract, config)
    #healpix of the all randoms
    all_healpix = hp.ang2pix(nside=nside, theta=all_ra, phi=all_dec, lonlat=True) #entire healpix in the tract
    all_healpix = to_little_endian(all_healpix)
    all_patch = to_little_endian(all_patch)
    
    if (len(all_ra)==0):
        print(f"tract {tract} is outside observed footprint")
        return None
    
    randoms = loader.Random(tract, config)
    randoms.load_bsmask(config)
    randoms.load_bgmask(config)
    random_mask = randoms.mask # true if "inside" masked region
    
    if(randoms.ra is None) or (np.sum(random_mask) == len(random_mask)):
        #Has randoms but all of them were inside masks or outside footprint
        print(f"No effective observation area in tract {tract}")
        mask = np.zeros(len(all_ID), dtype=bool)
    else:
        random_ID = randoms.objectID[~random_mask] #objectID of the randoms that were outside masks
        mask = np.isin(all_ID, random_ID)
                       
    table = Table({
        'healpix': all_healpix,
        'mask': mask, #true if 'outside' masks
        'patch': all_patch,
    })
    
    #group all the randoms into healpixels
    grouped = table.group_by("healpix").groups

    rows = []
    for g in grouped:
        rows.append({
            "healpix": g["healpix"][0],
            "total": len(g),               # mask の size
            "eff_area": np.sum(g["mask"])  # mask の sum
        })

    properties = Table(rows)

    patch_prop = band_property(tract, config, table)
    if patch_prop is None:
        return None
    properties = join(properties, patch_prop, keys='healpix', join_type='left')
    
    return properties


def to_little_endian(arr):
    dt = arr.dtype
    if dt.byteorder == '>':  # ONLY big endian
        return arr.byteswap().view(dt.newbyteorder('='))
    return arr


def band_property(tract, config, table):
    """
    function to add the g,r,i,z,y-depth and seeing of each healpixels
    
    Parameter
    ------------------------------------------
    tract: int
    ID of tract considered
    
    table: astropy table
    Table including the randoms in the concidered tract
    column: 'healpix', 'mask', 'patch'
    1. healpix: which healpix the random is included in
    2. mask: true if outside mask
    3. patch: which patch the random is included in

    output
    ------------------------------------------
    t:astropy table
    Table  of imaging properties for each randoms points
    defined by cross matching patch ID
    """
    patches = loader.Patches()
    patches.load_patches(tract, config, property_name)
    Property = patches.property
    
    if Property is None:
        print(f"patch column missing in tract {tract} property")
        return None

    table_valid = table[table['mask']] #Get randoms outside the masked area
    
    if(np.sum(table['mask']) == 0):#If there are no randoms outside mask (effective area = 0)
        # set all the imaging attributes as zero
        # This will not matter because the effective area is 0
        u, indices = np.unique(table['healpix'], return_index=True)
        patch_prop = Table({
            'healpix':u,
            'patch':table['patch'][indices]
        })
        for col in property_name:
            patch_prop[col] = np.zeros(len(u))
        return patch_prop

    patch_prop = join(
        table_valid[['healpix','patch']],
        Property,
        keys='patch',
        join_type='left'
    )

    #take the average per heal pixel
    groups = patch_prop.group_by('healpix').groups

    rows = []

    for g in groups:
        row = {'healpix': g['healpix'][0]}

        for col in property_name:
            row[col] = np.mean(g[col])

        rows.append(row)

    patch_prop = Table(rows)
    return patch_prop
   

def add_ext(properties, dust='csfd_desi', verbose=False):
    """ function to read in the dust map file and get E(B-V) for each healpixel
    
    Parameter
    ------------------------------------------
    tract: int
        ID of tract considered

    patches: instance
        instance of load.patch class. Includes the {g,r,i,z,y}-depth and {g,r,i,z,y}-seeing defined for each patch
    

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
        filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'dat', 
                "CSFD_DESI_merged_dust_map_NS2048_ring--Equatorial_nside256.npy")
        ebv = np.load(filename) 
        properties['csfd_desi_extinction'] = ebv[properties['healpix']]
        
    #csfd dust map
    elif dust=='csfd':
        filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'dat', 
                "CSFD_DESI_merged_dust_map_NS2048_ring.fits")
        with fits.open(filename) as hdul :
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
    else:
        print(f'No dust file corresponding to {dust}')
    
    return properties
    

def add_stellar_density(config, field, properties):
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
    
    star = loader.Star(field, config)
    ra = star.ra
    dec = star.dec
        
    if ra is None:
        print(f'No stars in {field} field')
        properties['star'] = np.zeros(len(properties['healpix']))
    else:
        healpy = hp.ang2pix(nside=nside, theta=ra, phi=dec, lonlat=True)
        
        _healpy, counts = np.unique(healpy, return_counts=True)
        data1 = {
            'healpix':_healpy,
            'star' : counts
        }
        table1 = Table(data1)
        properties = join(properties, table1, keys='healpix', join_type='left')

        properties['star'] = np.ma.filled(properties['star'], 0).astype(float)
        properties['star'] /= area
    
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
        table1 = Table({'healpix':_healpy, 'target' : counts})
        
        merged = join(properties, table1, keys='healpix', join_type='left')
        
        merged['target'] = merged['target'].filled(0)
        merged['target'] = merged['target'].astype(float)
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

    vec = hp.ang2vec(ra_deg, dec_deg, lonlat=True)

    # healpix index within radius_deg
    pix = hp.query_disc(
        nside,
        vec,
        np.radians(radius_deg),
        inclusive=False,
        nest=False
    )

    return pix

def clean_pixels(table, field, verbose=False):
    if verbose:
        total = len(table['healpix'])
        print(f"Original pixel count is {total} ")
    
    #remove anomaly
    if(field == 'spring'):
        anomaly_pix = anomaly()
        mask = ~np.isin(table['healpix'], anomaly_pix)
        table = table[mask]
        
        if verbose:
            masked = total - np.sum(mask)
            print(f"Removed {masked} pixels in anomalous region in {field} field")
            
    #remove healpix with small effective area
    mask = table['area'] > 0.0
    
    if verbose:
        total = len(table['healpix'])
        masked = total - np.sum(mask)
        print(f"Removed {masked} pixels with 0 effective area in {field} field")
    table = table[mask]
        
    #remove healpix with small random count
    mean_count = area * 3600.0 * 100.0
    mask = table['total'] > mean_count / 2.0
    if verbose:
        total = len(table['healpix'])
        masked = total - np.sum(mask)
        print(f"Removed {masked} pixels with small random count in {field} field")
    table = table[mask]

    #remove healpix with nan
    mask = np.ones(len(table['healpix']), dtype=bool)
    for col in table.colnames:
        if np.issubdtype(table[col].dtype, np.number):
            mask &= ~np.isnan(table[col])

    if verbose:
        total = len(table['healpix'])
        masked = total - np.sum(mask)
        print(f"Removed {masked} pixels with nans in {field} field")
    table = table[mask]
    
    #remove healpix with no star
    mask = table['star'] > 0.0
    if verbose:
        total = len(table['healpix'])
        masked = total - np.sum(mask)
        print(f"Removed {masked} pixels with 0 stars in {field} field")
    table = table[mask]
    
    return table
    
