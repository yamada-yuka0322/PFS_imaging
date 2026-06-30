from astroquery.gaia import Gaia
import os
import argparse

import yaml

coordinates = {
    'autumn':[((27.4, 40.0),(-7.5, 0.0)),((27.4, 40.0),(0.0, 6.0)),((13.7, 27.4),(-1.6, 6.0)),((0.0, 13.7),(-1.6, 6.0))
             ,((350.0, 360.0),(-1.6, 7.6)),((340.0, 350.0),(-1.6, 7.6)),((329.0, 340.0),(-1.6, 7.6))],
    'spring': [((127.0, 137.0),(-3.0, 6.0)),((137.0, 147.0),(-3.0, 6.0)),((147.0, 157.0),(-3.0, 6.0)),
              ((157.0, 167.0),(-3.0, 6.0)),((167.0, 177.0),(-3.0, 6.0)),((177.0, 187.0),(-3.0, 6.0)),
              ((187.0, 197.0),(-3.0, 6.0)),((197.0, 207.0),(-3.0, 6.0)),((207.0, 217.0),(-3.0, 6.0)),
              ((217.0, 227.0),(-3.0, 6.0))],
    'hectomap':[((199.0, 220.0),(41.5, 45.0)),((220.0, 230.0),(41.5, 45.0)),((230.0, 251.0),(41.5, 45.0)),]
}
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
        
    for field, coord_list in coordinates.items():
        outdir = os.path.join(config["Gaia"]["star"], f"{field}")
        os.makedirs(outdir, exist_ok=True)
        
        for coordinate in coord_list:
            ra, dec = coordinate
            ra_min, ra_max = ra
            dec_min, dec_max = dec
            tab = query_gaia_dr2_region(coordinate)
            filename = os.path.join(outdir, f"ra{ra_min}_{ra_max}-dec{dec_min}_{dec_max}.fits")
            tab.write(filename, format="fits", overwrite=True)
            
def query_gaia_dr2_region(coordinate, verbose=True):
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
        gs.phot_g_mean_mag > 12.0
        AND gs.phot_g_mean_mag < 17.0
        AND (
            gs.astrometric_excess_noise = 0
            OR LOG10(gs.astrometric_excess_noise) < 0.3 * gs.phot_g_mean_mag - 5.3
        )
    """
    
    ra, dec = coordinate
    ra_min, ra_max = ra
    dec_min, dec_max = dec

    poly = f"""
    POLYGON('ICRS',
    {ra_min}, {dec_min},
    {ra_max}, {dec_min},
    {ra_max}, {dec_max},
    {ra_min}, {dec_max}
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
        gs.phot_bp_rp_excess_factor,
        gs.astrometric_excess_noise
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