from astropy.table import Table, vstack, join
import astropy.io.fits as fits

import argparse

import os

import healpy as hp

import numpy as np

def parse_args():
    """
    """
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input', required=True, 
            help='input path')
    ap.add_argument('--out_dir', required=True, 
            help='output directory path')
    ap.add_argument('--weight-dir', '-w', help='path to the input mock')
    ap.add_argument('--method', '-m', nargs='+', default=['lin', 'quad', 'nn'],
                    choices=['lin', 'quad', 'nn'],
                    help='which method to apply the imaging systematics')
    return ap.parse_args()

def contaminate_mock(weights, method, mock):
    mock = mock.copy()
    
    weight = weights[method + "_weight"]
    
    mock["target"] = mock["target"]/weight
    return mock

def main():
    args = parse_args()
    
    path = args.input
    clean_mock = Table.read(path)
    
    path = args.weight_dir
    filename = os.path.join(path, "combined_weights.fits")
    weights = Table.read(filename)
    
    methods = args.method
    path = args.out_dir
    os.makedirs(path, exist_ok=True)
    for m in methods:
        cont_mock = contaminate_mock(weights, m, clean_mock)
        filename = os.path.join(path, f"{m}_contaminated_mock.fits")
        cont_mock.write(filename, format='fits', overwrite=True)
        print(f"saved {filename}")

if __name__ == "__main__":
    main()
    