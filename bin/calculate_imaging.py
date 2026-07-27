import os
import yaml
import argparse
from astropy.table import Table, vstack

from pfsimaging import imaging as Im
from pfsimaging import Loader as loader


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', required=True,
                    help='path to the config file which specify necessary path')
    ap.add_argument('--out-dir', required=True, 
            help='output path')
    ap.add_argument('--dustmap', default='csfd_desi',
                    choices=['csfd_desi', 'sfd'],
                    help='dust map for dust attenuation (default: csfd_desi)')
    ap.add_argument('--fields', nargs='+', default=['spring', 'autumn'],
                    choices=['spring', 'autumn', 'hectomap'],
                    help='which fields to process (default: spring autumn)')
    ap.add_argument('--nprocess', type=int, default=1,
                    help='number of processes for multiprocessing')
    ap.add_argument('--clean', '-c', action='store_true',
                        help='remove problematic pixels')
    ap.add_argument('--verbose', action='store_true', 
            help='enable print statements') 
    return ap.parse_args()


def main():
    args = parse_args()
    
    config = {}
    path = args.config
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    
    dustmap = args.dustmap
    fields = args.fields
    
    tables = []
    cleaned_tables = []
    for field in fields:
        if args.verbose: print(f'processing {field}') 
        TractPatch = loader.TractPatch(field)
        
        im_property = Im.get_property_all(TractPatch, dustmap, config, 
                nprocess=args.nprocess, 
                verbose=args.verbose)
        
        if im_property is None:
            print(f"no tract in {field} field")
        else:
            im_property.write(os.path.join(args.out_dir, f'{field}_property.fits'), 
                    format='fits', overwrite=True)
            
            tables.append(im_property)
            
            if (args.clean):
                cleaned_table = Im.clean_pixels(im_property, field, verbose=args.verbose)
                cleaned_table.write(os.path.join(args.out_dir,f'{field}_property_cleaned.fits'), 
                        format='fits', overwrite=True)
                cleaned_tables.append(cleaned_table)
     
    all_table = vstack(tables)
    all_table.write(os.path.join(args.out_dir, 'all_property.fits'), 
            format='fits', overwrite=True)
    
    if (args.clean):
        all_cleaned_table = vstack(cleaned_tables)
        all_cleaned_table.write(os.path.join(args.out_dir, 'all_property_cleaned.fits'), 
                format='fits', overwrite=True)
        
        
    
if __name__ == '__main__':
    main()
