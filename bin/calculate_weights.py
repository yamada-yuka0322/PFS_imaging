from pfsimaging import weights

from astropy.table import Table, vstack, join
import astropy.io.fits as fits

import argparse

import os

import yaml

keys = ['gseeing', 'rseeing', 'iseeing', 'zseeing', 'yseeing', 'star', 'g_depth', 'r_depth', 'i_depth', 'z_depth', 'y_depth', 'csfd_desi_extinction']
#keys = ['gseeing', 'rseeing', 'iseeing', 'zseeing', 'yseeing', 'g_depth', 'r_depth', 'i_depth', 'z_depth', 'y_depth', 'csfd_desi_extinction']

def parse_args():
    """
    output structure
    
    configs["out_dir"]["weights"] _______ lin_weights.fits
                                      |
                                      |__ quad_weights.fits
                                      |
                                      |__ nn_weights.fits
                                      |
                                      |__ combined_weights.fits

    """
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', required=True,
                    help='path to the config file which specify necessary path')
    ap.add_argument('--method', '-m', nargs='+', default=['lin', 'quad', 'nn'],
                    choices=['lin', 'quad', 'nn'],
                    help='which method to calculate the imaging systematic weights')
    ap.add_argument('--run_name', '-rn', default = "optuna",
                        help='specify the run name of the optuna')
    return ap.parse_args()

def method_weights(method, Property, args, config):
    if(method == 'lin'):
        table = weights.linear_weights(Property, keys)
        return table
    elif(method == 'quad'):
        table = weights.quadratic_weights(Property, keys)
        return table
    elif(method == 'nn'):
        best_model = os.path.join(config['out_dir']['optuna'], args.run_name, "best_model.pt")
        if os.path.exists(best_model):
            table = weights.nn_weights(Property, keys, best_model)
            return table
        else:
            print(f"file {best_model} does not exist")
            return None
    else: 
        print(f"method {method} not available")
        return None

def main():
    args = parse_args()
    
    config = {}
    path = args.config
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    
    methods = args.method
    
    out_dir = os.path.join(config['out_dir']['weights'], args.run_name)
    os.makedirs(out_dir, exist_ok=True)
    
    property_file = os.path.join(config['out_dir']['imaging'], "all_property_cleaned.fits")

    if os.path.exists(property_file):
        with fits.open(property_file) as hdu:
            data = hdu[1].data
            
        property_table = Table(data)
        table_all = property_table
        
        for m in methods:
            weight_table = method_weights(m, property_table, args, config)
            if (weight_table is not None):
                filename = os.path.join(out_dir, f"{m}_weights.fits")
                weight_table.write(filename, format='fits', overwrite=True)
                
                _weight = Table({'healpix': weight_table['healpix'],
                                f'{m}_weight': weight_table[f'{m}_weight']})
                table_all = join(table_all, _weight, join_type='left', keys='healpix')
                
        filename = os.path.join(out_dir, f"combined_weights.fits")
        table_all.write(filename, format='fits', overwrite=True)
    else:
        print(f"file {property_file} does not exist")
    
if __name__ == '__main__':
    main()