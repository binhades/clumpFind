#!/usr/bin/python3

import argparse, os, time
import numpy as np
from astropy.io import fits
from astropy import units as u
from astrodendro import Dendrogram, ppv_catalog

from matplotlib import pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib import colors as mcolors

def calc_dendrogram(data,sigma,snr,delta,ppb, save=False,fout='temp'):
    # Survey designs
    # sigma - K, noise level
    # ppb  - pixels/beam
    # delta - flux derivation between sources

    def custom_independent(structure, index=None, value=None):
       peak_index, peak_value = structure.get_peak()
       return peak_value > 1.5*sigma
                   
    d = Dendrogram.compute(data, min_value=snr*sigma, \
                            min_delta=delta*sigma, min_npix=ppb*ppb, \
                            is_independent=custom_independent, \
                            verbose = 1)
    #v=d.viewer()
    #v.show()
    
    #%&%&%&%&%&%&%&%&%&%&%&%&%&%
    #   Generate the catalog
    #%&%&%&%&%&%&%&%&%&%&%&%&%&%
    #print("Generate a catalog of dendrogram structures")
    #metadata = {}
    #metadata['data_unit'] = u.Jy #This should be Kelvin (not yet implemented)!
    #cat = ppv_catalog(d, metadata)
    #print(cat)

    if save:
        d.save_to(fout+'.hdf5')

    return d
 
def main(args):
    
    print('Load data cube from FITS')
    hdu = fits.open(args.fits_file)[0]
    data = hdu.data[args.chan_0:args.chan_1,:,:]

    print('Make dendrogram from the full cube')
    d = calc_dendrogram(data,args.sigma,args.snr,args.delta,args.ppb,save=True,fout=args.file_out)
   
    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('fits_file', type=str, help='the input data file')
    parser.add_argument('--file_out', type=str, default='my_dendrogram',help='the input data file')
    parser.add_argument('--sigma', type=float, default=0.001, help='the noise level')
    parser.add_argument('--ppb', type=float, default=3.0, help='the pixel per beam')
    parser.add_argument('--snr', type=float, default=3., help='snr of sigma to use')
    parser.add_argument('--delta', type=float, default=3., help='delta of sigma to asign a leaf')
    parser.add_argument('--chan_0', type=int, default=0,  help='the channel index to start')
    parser.add_argument('--chan_1', type=int, default=-1, help='the channel index to end')
    args = parser.parse_args()
    start_time = time.time()
    main(args)
    print("--- {:.3f} seconds ---".format(time.time() - start_time))
