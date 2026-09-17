from pfsimaging import train as Tr

import astropy.io.fits as fits
from astropy.table import Table

import argparse

import os

import yaml

#keys = ['gseeing', 'rseeing', 'iseeing', 'g_depth', 'r_depth', 'i_depth','csfd_desi_extinction']
keys = ['gseeing', 'rseeing', 'iseeing', 'zseeing', 'yseeing', 'g_depth', 'r_depth', 'i_depth', 'z_depth', 'y_depth','star', 'csfd_desi_extinction']

def parse_args():
    """
    output structure
    
    configs["output_dir"]["optuna"] __ run_name ____ trials_log.csv
                                                 |
                                                 |__ best_model.pt
                                                 |
                                                 |__ rank01_trial{}_val{}.pt
                                                 :
                                                 |__ rank{sane_n}_trial{}_val{}.pt
    """
    ap = argparse.ArgumentParser(description=__doc__)
    #ap.add_argument('--config', required=True,
                    #help='path to the config file which specify necessary path')
    ap.add_argument('--trial', default=200,
                    help='specify the number of trials of optuna run')
    ap.add_argument('--save_n', '-n', default=5,
                    help='number of top trials to save')
    ap.add_argument('--input', required = True,
                        help='specify the name of the input file')
    ap.add_argument('--output', required = True,
                        help='specify the name of the output directory')
    return ap.parse_args()

def main():
    args = parse_args()
    
    out_dir = args.output
    os.makedirs(out_dir, exist_ok=True)
    
    property_file = args.input
    if os.path.exists(property_file):
        with fits.open(property_file) as hdu:
            data = hdu[1].data
            
        property_table = Table(data)
        Tr.run_optuna_nn(out_dir, property_table, keys, n_trials=int(args.trial), top_k=int(args.save_n))
    else:
        print(f"file {property_file} does not exist")
    
if __name__ == '__main__':
    main()