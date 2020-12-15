#!/usr/bin/python3

import argparse, os, time
import numpy as np
from toolkit import smooth
from astropy.io import fits

def main(args):
    
    #%&%&%&%&%&%&%&%&%&%&%&%
    #    Load DataCube
    #%&%&%&%&%&%&%&%&%&%&%&%
    print('Load DataCube')
    with fits.open(args.file_cube) as hdul:
        hdr = hdul[0].header
        cube = hdul[0].data # shape: (CHAN, Dec(y), RA(x))

    with fits.open(args.file_diff) as hdul:
        diff = hdul[0].data # shape: (CHAN, Dec(y), RA(x))

    core = cube-diff
    fits.writeto('fit_cube.fits',core, header=hdr,overwrite=True)
    
    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('file_cube', type=str, help='the input data file')
    parser.add_argument('file_diff', type=str, help='the input data file')
    parser.add_argument('--file_core', type=str, default='core.fits', help='the dendrogram file')
    args = parser.parse_args()
    start_time = time.time()
    main(args)
    print("--- {:.3f} seconds ---".format(time.time() - start_time))
