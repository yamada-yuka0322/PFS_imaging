from astroquery.gaia import Gaia
import os
import sys
import csv
import json
import time
import astropy.io.fits as pyfits
import getpass
import argparse
import urllib.request, urllib.error, urllib.parse
import astropy.io.ascii as ascii

from pathlib import Path

from pfsimaging import imaging as Im

import numpy as np

from multiprocessing import Pool
from functools import partial

import yaml

args = None
config=None
outdir = None

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", "-c", default=None,
                        help="YAML config file containing paths and SQL settings")
    global args, config, out_dir
    args = parser.parse_args()
    
    config = {}
    if args.config is not None:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
    else:
        parser.error("--config are required")

    outdir = Path(config["Gaia"]["output_dir"]).expanduser()
    outdir.mkdir(parents=True, exist_ok=True)
    
    tract_data = {}

    autumn = Im.TractPatch("autumn")
    spring = Im.TractPatch("spring")

    tract_data.update(autumn.data)
    tract_data.update(spring.data)

    tract_list = autumn.get_tract()
    tract_list.extend(spring.get_tract())
    
    func = partial(wrapper, tract_data = tract_data, outdir=outdir)
    
    with Pool(processes=20) as pool:  # Adjust number of processes based on your CPU
        results = pool.map(func, tract_list)
    
    #for tract in tract_list:
        #data = tract_data[int(tract)]
        #corner1, corner0, corner3, corner2 = data['corner']
       # tab = query_gaia_dr2_region(corner0, corner1, corner2, corner3)
        #tab.write(f'/lustre/work/YukaYamada/data/Gaia_HSC/{tract}_stars.fits', format="fits", overwrite=True)
        #print(f"downloaded tract: {tract}")
        
def wrapper(tract, tract_data, outdir):
    filename = outdir / f"{tract}_stars.fits"
    if os.path.exists(filename):
        print(f"Stellar file in tract {tract} already exists")
    else:
        print(f"Fetching tract {tract}")
        data = tract_data[int(tract)]
        corner1, corner0, corner3, corner2 = data['corner']
        tab = query_gaia_dr2_region(corner0, corner1, corner2, corner3)
        if(tab is None):
            print("Not crossing ra=0.0")
        else:
            tab.write(filename, format="fits", overwrite=True)
            print(f"downloaded tract: {tract}")
    
    
def query_gaia_dr2_region(corner0, corner1, corner2, corner3, verbose=True):
    """
    Gaia DR2 から、指定した円形領域内で base_where を満たす星をすべて取得する。

    Parameters
    ----------
    ra_center_deg : float
        中心の RA [deg] (ICRS)
    dec_center_deg : float
        中心の Dec [deg] (ICRS)
    radius_deg : float
        円の半径 [deg]
    """

    # 使うテーブルを DR2 に固定
    Gaia.MAIN_GAIA_TABLE = "gaiadr2.gaia_source"

    base_where = """
        gs.phot_g_mean_flux_over_error > 50
        AND gs.phot_bp_mean_flux_over_error > 20
        AND gs.phot_rp_mean_flux_over_error > 20
        AND gs.phot_bp_rp_excess_factor < 1.3 + 0.06 * power(gs.phot_bp_mean_mag - gs.phot_rp_mean_mag, 2)
        AND gs.phot_bp_rp_excess_factor > 1.0 + 0.015 * power(gs.phot_bp_mean_mag - gs.phot_rp_mean_mag, 2)
    """
    
    corners = np.array([corner0, corner1, corner2, corner3], dtype=float)
    ra = corners[:, 0].copy()
    _ra = ra%360
    dec = corners[:, 1].copy()

    ra_min, ra_max = ra.min(), ra.max() #360を跨がない
    dec_min, dec_max = dec.min(), dec.max()
    crosses_zero = (_ra.max() - _ra.min()) > 180.0

    if crosses_zero:
        print("cross 0")
        
        poly = f"""
        POLYGON('ICRS',
        {ra_min}, {dec[0]},
        {360.0}, {dec_min},
        {360.0}, {dec_max},
        {ra_min}, {dec[3]}
        )
        """
        
        poly1 = f"""
        POLYGON('ICRS',
        {0.0}, {dec_min},
        {ra_max%360}, {dec[1]},
        {ra_max%360}, {dec[2]},
        {0.0}, {dec_max}
        )
        """
        
        area = f"1 = CONTAINS(POINT('ICRS', gs.ra, gs.dec), {poly}) OR 1 = CONTAINS(POINT('ICRS', gs.ra, gs.dec), {poly1})"
        
    else:
        poly = f"""
        POLYGON('ICRS',
        {_ra[0]}, {dec[0]},
        {_ra[1]}, {dec[1]},
        {_ra[2]}, {dec[2]},
        {_ra[3]}, {dec[3]}
        )
        """
        
        area = f"1 = CONTAINS(POINT('ICRS', gs.ra, gs.dec), {poly})"

    query = f"""
    SELECT
        gs.source_id,
        gs.ra, gs.dec,
        gs.phot_g_mean_mag,
        gs.phot_bp_mean_mag,
        gs.phot_rp_mean_mag,
        gs.phot_g_mean_flux_over_error,
        gs.phot_bp_mean_flux_over_error,
        gs.phot_rp_mean_flux_over_error,
        gs.phot_bp_rp_excess_factor
    FROM gaiadr2.gaia_source AS gs
    WHERE
        {area}
        AND (
            {base_where}
        )
    """

    if verbose:
        print("Sending ADQL to Gaia (DR2)...")
        print(query)

    job = Gaia.launch_job_async(query, dump_to_file=False)
    result = job.get_results()
    
    if verbose:
        print(f"Fetched {len(result)} rows from Gaia DR2.")

    return result  # astropy.table.Table

if __name__ == '__main__':
    main()