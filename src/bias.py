import re
import numpy as np
from astropy.io import fits
from astropy.table import Table, vstack, Column
import healpy as hp
import sys,os
import multiprocessing
from multiprocessing import Pool
import hscReleaseQuery as HQR
import argparse
import astropy.io.ascii as ascii
from astropy.coordinates import Galactic, FK5
import astropy.units as u
from functools import partial
from scipy.spatial import cKDTree
import pandas as pd

field = ['AEGIS', 'autumn', 'hectomap', 'spring']
version =   20190924.1
args    =   None
#doDownload  =   True
#doUnzip =   True
Download = True
diffver =   '-colorterm'
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
    """Function to load the coordicates of tract and patch coordinates

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

    file_path = f"tracts_patches_W-{field}.txt"
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

#############################################################################
#Download data from HSC

def Download_HSC(args, sqls, tracts2, Unzip=True):
    """Function to download data from HSC database

    Parameter
    ----------------------------------------------------
    args:arguments
    must include {user, out_format, delete_job, api_url, password-env}

    sqls:list(strings)
    The sql file name.

    tracts2: list(int)
    list of the tract number you want to download (in int)

    Unzip: bool
    Set True if you want to separate the data files into tracts.
    """
    release_year   =   "s21"
    release_version =   'dr4-citus'

    ig = np.arange(len(tracts2))
    Input = list(zip(ig, tracts2))

    ig_tract = [[i, elem] for i, sublist in enumerate(tracts2) for elem in sublist]
    
    for i,sql in enumerate(sqls):
        name = sqls[i].split('.')[0]
        prefix = f'database/{name}/sql_{name}'
        prefix2 = f'database/{name}/tracts_{name}'
        
        os.makedirs(prefix, exist_ok=True)
        os.makedirs(prefix2, exist_ok=True)
        # Read SQL file content here
        with open(sql, 'r') as f:
            sql_content = f.read()

        if Download:
            credential  =   {'account_name': args.user, 'password': HQR.getPassword(args)}
            for _ig,_tractL in enumerate(tracts2):
                #restart
                #if(ig<194): continue
                print('Group: %s' %_ig)
                HQR.downloadTracts(args, prefix, release_version, credential, sql_content, _ig, _tractL)
        if Unzip:
            with Pool(processes=50) as pool:  # Adjust number of processes based on your CPU
                func = partial(separateTract, prefix=prefix, prefix2=prefix2)
                results = pool.map(func, ig_tract)
    return None
    

def separateTract(ig_tract, prefix,prefix2):
    """function to separate the tract group files into tract files
    
    Parameters
    -------------------------------------------------
    prefix:string
    directory path to the tract group files

    prefix2:string
    directory path to the tract files

    ig_tract:list[int,int]
    list of tract group index and the tract number

    
    """
    ig, tract = ig_tract
    print('unzipping group: %s' %ig)
    infname     =   f'{ig}_bias.fits'
    infname     =   os.path.join(prefix,infname)
    if not os.path.exists(infname):
        print('Does not have input file')
        return
    fitsAll     =   fits.getdata(infname)
    print('read %s galaxies' %len(fitsAll))

    outfname    =   os.path.join(prefix2,'%s_bias.fits' %(tract))
    if os.path.exists(outfname):
        print('already have file for tract: %s'%tract)
    else:
        Fits        =   fitsAll[fitsAll['tract']==int(tract)]
        if len(Fits)>10:
            fits.writeto(outfname,Fits)
        del Fits
    return

class arguments(object):
    """Class to convert argument dictionary into argument class

    Parameter
    ---------------------------------------
    dict:dictionary
    Must include 'user', 'out_format', 'delete_job', 'api_url', 'nomail', 'skip_syntax_check' and 'password_env'
    """
    def __init__(self, dict):
        self.user = dict['user']
        self.out_format = dict['out_format']
        self.delete_job = dict['delete_job']
        self.api_url = dict['api_url']
        self.nomail = dict['nomail']
        self.skip_syntax_check = dict['skip_syntax_check']
        self.password_env = dict['password_env']

###############################################################################
def Mask(tract, TractPatch_dict):
    """function to get the stellar mask for tract

    Parameter
    ----------------------------------
    tract:int
    tract number

    TractPatch_dict:dictionary
    dictionary including the location of tracts and patches

    Output
    ___________________________________
    filtered_df:dataFrame('ra', 'dec', "IsOutsideMask", "healpix")
    random points in the target tract, IsOutsideMask = 1 for points outside the stellar mask. healpix with small coverage is excluded
    """
    data = TractPatch_dict
    corner = np.array(data[tract]['corner'])
    ra = corner[:,0]%360
    dec = corner[:,1]

    # 配列をソートして二番目に大きい値と二番目に小さい値を取得
    sorted_ra = np.sort(ra)
    max_ra = sorted_ra[-2]-0.1  # 二番目に大きい値
    min_ra = sorted_ra[1]+0.1  # 二番目に小さい値

    sorted_dec = np.sort(dec)
    max_dec = sorted_dec[-2]-0.1  # 二番目に大きい値
    min_dec = sorted_dec[1]+0.1  # 二番目に小さい値
    
    os.makedirs('mask', exist_ok=True)
    filename = f"mask/tract_{tract}_flagged.fits"
    if not os.path.exists(filename):
        if (max_ra-min_ra>300):
            max_ra = sorted_ra[0]-0.1  # 1番大きい値
            min_ra = sorted_ra[-1]+0.1  # 1番小さい値

            max_dec = sorted_dec[-2]-0.1  # 二番目に大きい値
            min_dec = sorted_dec[1]+0.1  # 二番目に小さい値
            file = f"tract_{tract}_left.fits"
            command = "HSC-SSP_brightStarMask_Arcturus/venice-4.0.3/bin/venice -r -xmin %s -xmax %s -ymin %s -ymax %s -coord spher -o mask/tract_%s_left.fits"%(0, max_ra, min_dec, max_dec,tract)
            os.system(command)
        
            command1 = "HSC-SSP_brightStarMask_Arcturus/venice-4.0.3/bin/venice -m HSC-SSP_brightStarMask_Arcturus/reg/masks_all.reg -cat mask/tract_%s_left.fits -xcol ra -ycol dec -f all -flagName isOutsideMask -o mask/tract_%s_flagged_left.fits"%(tract,tract)
            os.system(command1)
        
            command = "HSC-SSP_brightStarMask_Arcturus/venice-4.0.3/bin/venice -r -xmin %s -xmax %s -ymin %s -ymax %s -coord spher -o mask/tract_%s_right.fits"%(min_ra, 360, min_dec, max_dec,tract)
            os.system(command)
        
            command1 = "HSC-SSP_brightStarMask_Arcturus/venice-4.0.3/bin/venice -m HSC-SSP_brightStarMask_Arcturus/reg/masks_all.reg -cat mask/tract_%s_right.fits -xcol ra -ycol dec -f all -flagName isOutsideMask -o mask/tract_%s_flagged_right.fits"%(tract,tract)
            os.system(command1)
        
            hdu = fits.open("mask/tract_%s_flagged_left.fits" %tract)
            data = hdu[1].data
            ra_left = data["ra"]
            dec_left = data["dec"]
            flag_left = data["IsOutsideMask"]
            hdu.close()
                
            hdu = fits.open("mask/tract_%s_flagged_right.fits" %tract)
            data = hdu[1].data
            ra_right = data["ra"]
            dec_right = data["dec"]
            flag_right = data["IsOutsideMask"]
            hdu.close()
                
                
            ra = np.concatenate([ra_left,ra_right])
            dec = np.concatenate([dec_left,dec_right])
            flag = np.concatenate([flag_left, flag_right])

            t = Table([ra, dec, flag], names=('ra', 'dec', "IsOutsideMask"))
            filename = f"mask/tract_{tract}_flagged.fits"
            t.write(filename, format='fits', overwrite=True)
            remove = f"rm mask/tract_{tract}_flagged_left.fits mask/tract_{tract}_left.fits mask/tract_{tract}_flagged_right.fits mask/tract_{tract}_right.fits"
            os.system(remove)
            
            
        else:
            command = "HSC-SSP_brightStarMask_Arcturus/venice-4.0.3/bin/venice -r -xmin %s -xmax %s -ymin %s -ymax %s -coord spher -o mask/tract_%s.fits"%(min_ra, max_ra, min_dec, max_dec,tract)
            os.system(command)
            command1 = "HSC-SSP_brightStarMask_Arcturus/venice-4.0.3/bin/venice -m HSC-SSP_brightStarMask_Arcturus/reg/masks_all.reg -cat mask/tract_%s.fits -xcol ra -ycol dec -f all -flagName isOutsideMask -o mask/tract_%s_flagged.fits"%(tract,tract)
            os.system(command1)

            hdu = fits.open("mask/tract_%s_flagged.fits" %tract)
            data = hdu[1].data
            ra = data["ra"]
            dec = data["dec"]
            flag = data["IsOutsideMask"]
            hdu.close()

            #remove = f"rm mask/tract_{tract}_flagged.fits mask/tract_{tract}.fits"
            #os.system(remove)
            t = Table([ra, dec, flag], names=('ra', 'dec', "IsOutsideMask"))
    else:
        print(f'already have mask for tract {tract}')
        hdu = fits.open("mask/tract_%s_flagged.fits" %tract)
        data = hdu[1].data
        ra = data["ra"]
        dec = data["dec"]
        flag = data["IsOutsideMask"]
        hdu.close()

        #remove = f"rm mask/tract_{tract}_flagged.fits mask/tract_{tract}.fits"
        #os.system(remove)
        t = Table([ra, dec, flag], names=('ra', 'dec', "IsOutsideMask"))

    healpy = hp.pixelfunc.ang2pix(nside=nside, theta=ra, phi=dec, lonlat=True)
    # add healpix column
    new_col = Column(healpy, name='healpix', dtype='int32')
    t.add_column(new_col)

    #remove healpy that are on the edge of tract
    df = t.to_pandas()
    counts = df['healpix'].value_counts()

    # remove healpix with less than 16000 dots
    #filtered_df = df[~df['healpix'].isin(counts[counts <= 16000].index)]
    #return filtered_df
    return df

##############################################################################################################
def get_property_all(tractpatch):
    """function to get the properties of healpixels in a single field (AEGIS, autumn, hectomap or spring)

    Parameter
    ------------------------------------------
    tractpatch: instance
    instance of tractpatch class. Includes the information about the tract and patch included in the field considered.

    output
    ------------------------------------------
    t:table
    table  of imaging properties for each healpixels in the field
    includes seeing, depth, stellar density, target density and extinction
    """

    
    tractpatch_dict = tractpatch.data
    tractlist = tractpatch.get_tract()
    with Pool(processes=20) as pool:  # Adjust number of processes based on your CPU
        func = partial(get_property_tract, tractpatch_dict=tractpatch_dict)
        results = pool.map(func, tractlist)
        
    # Exclude Nore results
    valid_results = [res for res in results if res is not None]

    if valid_results:
        all_property = pd.concat(valid_results, ignore_index=True)
        # If a single healpixel overlaps between multiple tracts, select the healpixels with the largest effective area
        #result = all_property.loc[all_property.groupby('healpix')['area'].idxmax()]

        #propertyはMaskで重みをつけて平均
        seeing_columns = ['gseeing', 'rseeing', 'iseeing', 'zseeing', 'yseeing']
        depth_columns = ['g_depth', 'r_depth', 'i_depth', 'z_depth', 'y_depth']
        names = [sql.split('.')[0] for sql in args.sql]
        all_columns = seeing_columns + depth_columns + ['extinction']
        property = all_property.groupby('healpix').apply(lambda x: pd.Series(
            {col: np.sum(x[col] * x['Mask']) / np.sum(x['Mask']) for col in all_columns} |  # Seeing, depth, extinction
            {'area': np.sum(x['Mask']) / np.sum(x['counts']) * area}|
            {col: np.sum(x[col])/(np.sum(x['Mask']) / np.sum(x['counts']) * area) for col in names})).reset_index()
        
        t = Table.from_pandas(property)
        return t
    else:
        # return table if empty
        return Table()
    
    

def get_property_tract(tract, tractpatch_dict):
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
    mask = Mask(tract, tractpatch_dict)
    out_mask = (mask["IsOutsideMask"]==1)
    #_table = mask[out_mask]
    #_ra = np.array(_table['ra'])
    #_dec = np.array(_table['dec'])
    #_healpy = np.array(_table['healpix'])

    ra = np.array(mask['ra'])
    dec = np.array(mask['dec'])
    healpy = np.array(mask['healpix'])
    _ra = ra[out_mask]
    _dec = dec[out_mask]
    _healpy = healpy[out_mask]

    patch_dict = tractpatch_dict[tract]['patch']

    patch_id = get_closest_patch(_ra, _dec, patch_dict)
    property = get_patch_property(patch_id, tract)
    # Check for empty dataframe
    if property.empty:
        # return None if empty
        return None
        
    property['healpix'] = _healpy
    property['healpix'] = property['healpix'].astype('int32')
    
    # take the average of the properties for the target points
    properties = property.groupby('healpix')[['gseeing', 'rseeing', 'iseeing', 'zseeing', 'yseeing', 'g_depth', 'r_depth', 'i_depth', 'z_depth', 'y_depth']].mean().reset_index()

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
    properties = add_object_count(properties, args.sql,tract)
    return properties

def get_closest_patch(ra, dec, patch_dict):
    """function to get the id of the closest patch

    Parameters
    -------------------------------------------------
    ra:array
    ra of target points in deg

    dec:array
    dec of target points in deg

    patch_dict:dictionary
    dictionary including the patch ids and center coordinate {(0,0):(ra,dec), (0,1):(ra,dec),...}

    Output
    ------------------------------------------------
    patch_id:array
    array of the closest patch id array([508,706,102,108,8,700,601,...)
    """
    patch = np.array(list(patch_dict.values()))

    # list of ra, dec of target points
    target_points = np.concatenate([ra.reshape(-1,1), dec.reshape(-1,1)],axis=1)

    # wrap the patch coordinates
    wrapped_ra_points = np.mod(patch[:, 0], 360)

    # build KD-Tree
    tree = cKDTree(np.column_stack((wrapped_ra_points, patch[:, 1])))

    # look for the closest patch coordinate for the target points
    wrapped_target_points = np.array([wrap_ra(target[0]) for target in target_points])
    wrapped_target_points = np.column_stack((wrapped_target_points, target_points[:, 1]))

    distances, indices = tree.query(wrapped_target_points)

    arr = np.array(list(patch_dict.keys()))[indices]
    
    patch_id = np.array(np.char.add(np.char.add(arr[:, 0].astype(str), '0'), arr[:, 1].astype(str)), dtype=int)
    
    return patch_id

# wrap ra to consider periodicity
def wrap_ra(ra):
    return ra % 360

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
    

def add_object_count(properties, sqls, tract):
    table2 = properties
    for sql in sqls:
        name =  sql.split('.')[0]
        filename = f'database/{name}/tracts_{name}/{tract}_bias.fits'
        if not os.path.exists(filename):
            print(f"No {name} in tract {tract}")
            table2[name] = 0
            return table2
        with fits.open(filename) as hdu:
            data = hdu[1].data
            ra = data['ra']%360
            dec = data['dec']
        healpy = hp.pixelfunc.ang2pix(nside=nside, theta=ra, phi=dec, lonlat=True)
        _healpy, counts = np.unique(healpy, return_counts=True)
        data1 = {
            'healpix':_healpy,
            name : counts
        }
        table1 = pd.DataFrame(data1)
        table2 = pd.merge(table2, table1, on='healpix', how='left')
    return table2
    
###########################################################################################################
def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--user', '-u', help='specify your STARS account')
    parser.add_argument('--password-env', default='HSC_SSP_CAS_PASSWORD', help='environment variable for STARS')
    parser.add_argument('--api-url',
                        default='https://hscdata.mtk.nao.ac.jp/datasearch/api/catalog_jobs/',
                        help='for developers')
    #If only using parts of the whole HSC data
    parser.add_argument("--tracts", nargs = "*", help = "properties", default = '')
    parser.add_argument("--sql" , nargs= "*", help = "sql file for stars and target galaxies", default = '')
    parser.add_argument("--patch_sql" , nargs= "*", help = "sql file for patch properties", default = 'patch_property.sql')

    parser.add_argument('--format', '-f', dest='out_format', default='fits',
                        choices=['csv', 'csv.gz', 'sqlite3', 'fits'],
                        help='specify output format')
    parser.add_argument('--release_year', '-r', default = 's21',
                        choices='s15 s16 s17 s18 s19 s20 s21'.split(),
                        help='the release year')
    parser.add_argument('--nomail', '-M', action='store_true',
                        help='suppress email notice')

    parser.add_argument('--preview', '-p', action='store_true',
                        help='quick mode (short timeout)')
    parser.add_argument('--skip-syntax-check', '-S', action='store_true',
                        help='skip syntax check')
    parser.add_argument('--delete-job', '-D', action='store_true',
                        help='delete job after your downloading')
    global args
    args = parser.parse_args()

###################################################################    
    #if no tracts, all tract in HSC database will be downloaded
    tractlist = args.tracts
    sql_list = args.sql
    user = args.user
    patch_sql = args.patch_sql
    
    ngroups = 1
    if (tractlist==''):
        tractname= 'fieldTractInfoS21.csv'
        tracts      =   ascii.read(tractname)['tract']
        tractlist = list(tracts)
        ngroups = 20

    tracts2     =   HQR.chunkNList(list(tractlist),ngroups)

####################################################################
    #download stars and targets if necessary
    if (sql_list!=''):
        if(user is None):
            parser.error("The '--user' argument is required.")
        else:
            Download_HSC(args=args, sqls=sql_list, tracts2=tracts2, Unzip=True)
    
    Download_HSC(args=args, sqls=patch_sql, tracts2=tracts2, Unzip=True)

####################################################################
    #tract_patch 
    autumn = TractPatch("autumn")
    AEGIS = TractPatch("AEGIS")
    hectomap = TractPatch("hectomap")
    spring = TractPatch("spring")
    
####################################################################
    #Get Property for each healpixel
    autumn_property = get_property_all(autumn)
    autumn_property.write('autumn_property.fits', format='fits', overwrite=True)
    AEGIS_property = get_property_all(AEGIS)
    AEGIS_property.write('AEGIS_property.fits', format='fits', overwrite=True)
    hectomap_property = get_property_all(hectomap)
    hectomap_property.write('hectomap_property.fits', format='fits', overwrite=True)
    spring_property = get_property_all(spring)
    spring_property.write('spring_property.fits', format='fits', overwrite=True)
    return

if __name__ == '__main__':
    main()