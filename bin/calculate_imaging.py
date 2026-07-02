from pfsimaging import imaging as Im
from pfsimaging import Loader as loader

from astropy.table import Table, vstack

import argparse

import os

import yaml

def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', required=True,
                    help='path to the config file which specify necessary path')
    ap.add_argument('--dustmap', default='csfd_desi',
                    choices=['csfd_desi', 'sfd'],
                    help='dust map for dust attenuation (default: csfd_desi)')
    ap.add_argument('--fields', nargs='+', default=['spring', 'autumn'],
                    choices=['spring', 'autumn', 'hectomap'],
                    help='which fields to process (default: spring autumn)')
    ap.add_argument('--clean', '-c', action='store_true',
                        help='remove problematic pixels')
    return ap.parse_args()


def main():
    args = parse_args()
    
    config = {}
    path = args.config
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    
    out_dir = config['out_dir']['imaging']
    os.makedirs(out_dir, exist_ok=True)
    
    dustmap = args.dustmap
    fields = args.fields
    
    tables = []
    cleaned_tables = []
    for field in fields:
        TractPatch = loader.TractPatch(field)
        
        file = os.path.join(out_dir,f'{field}_property.fits')
        im_property = Im.get_property_all(TractPatch, dustmap, config)
        
        if im_property is None:
            print(f"no tract in {field} field")
        else:
            table = Table.from_pandas(im_property)
            table.write(file, format='fits', overwrite=True)
            
            tables.append(table)
            
            if (args.clean):
                cleaned_file = os.path.join(out_dir,f'{field}_property_cleaned.fits')
                cleaned_table = Im.clean_pixels(table, field)
                cleaned_table.write(cleaned_file, format='fits', overwrite=True)
                cleaned_tables.append(cleaned_table)
     
    all_table = vstack(tables)
    file = os.path.join(out_dir,'all_property.fits')
    all_table.write(file, format='fits', overwrite=True)
    
    if (args.clean):
        all_cleaned_table = vstack(cleaned_tables)
        cleaned_file = os.path.join(out_dir,f'all_property_cleaned.fits')
        all_cleaned_table.write(cleaned_file, format='fits', overwrite=True)
        
        
    
if __name__ == '__main__':
    main()