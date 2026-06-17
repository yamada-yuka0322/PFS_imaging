import astropy.io.fits as fits
import astropy.io.ascii as ascii
from astropy.table import Table, vstack

import numpy as np
from pathlib import Path
import os

from multiprocessing import Pool
from functools import partial

from scipy.spatial import cKDTree

import yaml

import argparse

from pfsimaging import Loader as loader

autumn = None
gal_ra = None
gal_dec = None
gal_id = None

STAR_CACHE = {}

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--kind", required = True,
                        choices=["star", "galaxy", "random"],
                        help="which object to generate mask")
    parser.add_argument("--config", "-c", required = True,
                        help="YAML config file containing paths")

    global args
    args = parser.parse_args()

    tract_data = {}

    autumn = loader.TractPatch("autumn")
    spring = loader.TractPatch("spring")

    tract_data.update(autumn.data)
    tract_data.update(spring.data)

    tract_list = autumn.get_tract()
    tract_list.extend(spring.get_tract())
    #tractname=  './Tracttest.csv'
    #tract_list      =   ascii.read(tractname)['tract']

    config = {}
    if args.config is not None:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
    else:
        parser.error("--config are required")

    outdir = os.path.join(config["Gaia"]["bsmask_dir"] , args.kind)
    Path(outdir).mkdir(parents=True, exist_ok=True)

    StarPath = config["Gaia"]["output_dir"]
    func = partial(process_tract, StarPath = StarPath)
    
    all_cols = []
    all_tables = []
    
    if (args.kind == 'galaxy'):
        Object = loader.Target(config)
        print(f"{len(Object.ra)} targets downloaded")
    elif (args.kind == 'star'):
        Object = loader.Star()
    elif (args.kind == 'random'):
        Object = loader.Random()

    for tract in tract_list:
        filename = f"{tract}.fits"
        path = os.path.join(outdir , filename)
        if os.path.exists(path):
            print(f"bright star mask for tract {tract} exists")
            
            # stack existing bsmasks for galaxies
            if args.kind == "galaxy":
                all_tables.append(Table.read(path))

            continue
            
            
        global _ra, _dec, _id
        _ra, _dec, _id = GetObjects(tract, args.kind, config, Object)

        if (_ra is None) or (len(_ra) == 0):
            print(f"{args.kind} file for tract {tract} does not exist")
            continue
            
        adj_tracts = get_adjacent_tracts(tract_data, tract)
        attributes = [(adj, _ra, _dec) for adj in adj_tracts]
    
        with Pool(len(adj_tracts)) as p:
            results = p.map(func, attributes)
        
        mask_halo = np.zeros(len(_ra), dtype=bool)
        mask_ghost = np.zeros(len(_ra), dtype=bool)
        mask_blooming = np.zeros(len(_ra), dtype=bool)
        
        for h, g, b in results:
            mask_halo |= h
            mask_ghost |= g
            mask_blooming |= b

        print(f"Saving bright star mask for tract {tract}") 

        cols = [
            fits.Column(name='id', format='K', array=_id),
            fits.Column(name='ra', format='D', array=_ra),
            fits.Column(name='dec', format='D', array=_dec),
            fits.Column(name="halo", format="L", array=mask_halo),
            fits.Column(name="ghost", format="L", array=mask_ghost),
            fits.Column(name="blooming", format="L", array=mask_blooming),
        ]

        hdu = fits.BinTableHDU.from_columns(cols)
        hdu.writeto(path, overwrite=True)
        
        if args.kind == "galaxy":
            all_tables.append(Table(hdu.data))
            
    if args.kind == "galaxy" and len(all_tables) > 0:
        combined = vstack(all_tables)
        combined_path = os.path.join(outdir, "all_tracts.fits")
        combined.write(combined_path, overwrite=True)
        print(f"Saved combined galaxy bright star mask: {combined_path}")
                
def process_tract(tasks, StarPath):
    #書き込むのはこのtractだけ
    #星はadjacent tractもとってくる
    tract, _ra, _dec = tasks
    
    star_ra, star_dec, star_imag = GetStars(tract, StarPath)
    
    if (star_ra is None):
        mask = np.zeros(len(_ra), dtype=bool)
        return mask, mask, mask

    _halo, _ghost, _blooming = GetMask(star_ra, star_dec, star_imag, _ra, _dec)

    return _halo, _ghost, _blooming

def GetStars(tract, path):
    StarPath = os.path.join(path, f"{tract}_stars.fits")
    if os.path.exists(StarPath):
        hdu = fits.open(StarPath)

        data = hdu[1].data
        ID = data['source_id']
        ra = data['ra']
        dec = data['dec']
        phot_g = data['phot_g_mean_mag']
        phot_bp = data['phot_bp_mean_mag']
        phot_rp = data['phot_rp_mean_mag']
        hdu.close()
        
        x = phot_bp - phot_rp
        imag = ( phot_g + 4.906063e-02 - 6.084751e-01 * x + 5.999354e-02 * x**2 + 8.071123e-03 * x**3 + 7.058538e-04 * x**4)
        
        return ra, dec, imag
    else:
        print(f"Stellar file for tract {tract} does not exist")
        return None, None, None

    
def GetObjects(tract, kind, config, Object):
    if(kind == 'galaxy'):
        ra, dec, _, _, ID = Object.get_tract(int(tract))
        print(f"{len(ra)} targets in tract{tract}")
    elif(kind=='star'):
        Object.load_stars(tract, config)
        ra = Object.ra
        dec = Object.dec
        ID = Object.objectID
    elif(kind=='random'):
        Object.load_random(tract, config)
        ra = Object.ra
        dec = Object.dec
        ID = Object.objectID
 
    return ra, dec, ID
    
def get_adjacent_tracts(tract_dict, target_tract, tol=1.0):

    if target_tract not in tract_dict:
        raise ValueError(f"{target_tract} is not in tract_dict")

    target_center = np.array(tract_dict[target_tract]['center'])

    tract_ids = list(tract_dict.keys())
    centers = np.array([tract_dict[t]['center'] for t in tract_ids])

    # 差分
    dra_raw = np.abs(centers[:, 0] - target_center[0])
    dra = np.minimum(dra_raw, 360.0 - dra_raw)   # ⭐ ここが重要
    ddec = np.abs(centers[:, 1] - target_center[1])

    # ステップ幅推定
    nonzero_dra = np.sort(np.unique(np.round(dra[dra > 0], 6)))
    nonzero_ddec = np.sort(np.unique(np.round(ddec[ddec > 0], 6)))

    if len(nonzero_dra) == 0 or len(nonzero_ddec) == 0:
        return []

    step_ra = nonzero_dra[0]
    step_dec = nonzero_ddec[0]

    neighbors = []
    for t, dx, dy in zip(tract_ids, dra, ddec):

        cond_ra = (abs(dx - step_ra) < tol) or (dx < tol)
        cond_dec = (abs(dy - step_dec) < tol) or (dy < tol)

        if cond_ra and cond_dec:
            neighbors.append(t)

    return sorted(neighbors)

def GetMask(star_ra, star_dec, star_imag, gal_ra, gal_dec):
    star_ra = star_ra%360
    gal_ra = gal_ra%360
    #raを0~360に入れる
    if(((gal_ra.max() - star_ra.min())>180.0) or ((star_ra.max() - gal_ra.min())>180.0)):
        #ra=0.0を跨いでいる場合 -180~180に折り返す
        star_ra = np.where(star_ra>180.0, star_ra - 360.0, star_ra)
        gal_ra = np.where(gal_ra>180.0, gal_ra - 360.0, gal_ra)
        
    gal_pos = np.vstack([gal_ra, gal_dec]).T
    tree = cKDTree(gal_pos)

    mask_halo = np.zeros(len(gal_ra), dtype=bool)
    mask_ghost = np.zeros(len(gal_ra), dtype=bool)
    mask_blooming = np.zeros(len(gal_ra), dtype=bool)

    for ra, dec, imag in zip(star_ra, star_dec, star_imag):

        # ===== halo =====
        halo = 1.105e3 * np.exp(-0.347 * imag) + 4.950
        r_halo = halo / 3600.0
        idx = tree.query_ball_point([ra, dec], r=r_halo)
        idx = np.array(idx, dtype=int)
        mask_halo[idx] = True

        # ===== ghost =====
        if imag < 6.75:
            ghost = 700.0
        elif imag < 9.5:
            ghost = 13.49 * imag**2 - 338.32 * imag + 2061.89
        else:
            ghost = 0.0

        if ghost > 0:
            r_ghost = ghost / 3600.0
            idx = tree.query_ball_point([ra, dec], r=r_ghost)
            idx = np.array(idx, dtype=int)
            mask_ghost[idx] = True

        # ===== blooming =====
        if imag < 8.75:
            blooming_size = 700.0
        else:
            blooming_size = -200.0 * imag + 2460.0

        if imag < 6.75:
            blooming_width = 0.0
        else:
            blooming_width = -2.63 * imag + 34.93

        if blooming_width > 0:
            r = (blooming_size**2 + blooming_width**2)**0.5 / 3600.0
            idx = tree.query_ball_point([ra, dec], r=r)
            idx = np.array(idx, dtype=int)

            sub_ra = gal_ra[idx]
            sub_dec = gal_dec[idx]

            mask = (
                (np.abs(sub_dec - dec) * 3600.0 < blooming_width) &
                (np.abs(sub_ra - ra) * 3600.0 < blooming_size)
            )

            mask_blooming[idx[mask]] = True

    return mask_halo, mask_ghost, mask_blooming

if __name__ == '__main__':
	main()
