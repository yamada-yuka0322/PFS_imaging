from astroquery.gaia import Gaia
import os
import argparse

from pfsimaging import Loader as loader

import numpy as np

from multiprocessing import Pool
from functools import partial

import yaml

def main():
    """
    function to download the Gaia stars for the bright stellar mask
    
    arguments
    -----------------------------------------------------
    config: path to YAML config file
    The output file directory must be specified under config["Gaia"]["output_dir"]
    
    output
    ------------------------------------------------------
    fits file
    The Gaia star catalog will be saved in a fits file per tract under config["Gaia"]["output_dir"] directory.
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", "-c", default=None,
                        help="YAML config file containing paths and SQL settings")
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

    autumn = loader.TractPatch("autumn")
    spring = loader.TractPatch("spring")

    tract_data.update(autumn.data)
    tract_data.update(spring.data)

    tract_list = autumn.get_tract()
    tract_list.extend(spring.get_tract())
    
    func = partial(wrapper, tract_data = tract_data, outdir=outdir)
    
    with Pool(processes=20) as pool:  # Adjust number of processes based on your CPU
        results = pool.map(func, tract_list)
        
def wrapper(tract, tract_data, outdir):
    """
    wrapper function to pass the corner of each tracts to query_gaia_dr2_region function
    """
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
    Function to get all stars within the tract polygon which passes the selection discribed in base_where

    Parameters
    ----------
    corner0: [ra, dec] corner of tract (lower right)
    corner1: [ra, dec] corner of tract (lower left)
    corner2: [ra, dec] corner of tract (upper left)
    corner3: [ra, dec] corner of tract (upper right)
    
    return
    ---------
    astropy table of the downloaded stars
    """

    Gaia.MAIN_GAIA_TABLE = "gaiadr2.gaia_source"

    #Same as the selection used for the HSC bright stellar mask
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

    #In case where the tract crosses ra = 0deg, the left and the right side would be defined as a different polygon
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