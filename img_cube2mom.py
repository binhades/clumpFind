#!/usr/bin/python3

import argparse, os, time
import numpy as np
from matplotlib import pyplot as plt
from astropy.io import fits
from astropy import units as u
import aplpy

# Function to remove the 3rd dimension in a 
# spectroscopic cube header
def hd2d(hd):

    # Create bi-dimensional header
    mhd = fits.PrimaryHDU(np.zeros([hd['NAXIS2'],hd['NAXIS1']])).header
    
    for i in ['1','2']:
            for t in ['CRVAL','CRPIX','CDELT','CTYPE','CROTA','CUNIT']:
                    if hd.get(t+i) != None:
                            mhd[t+i] = hd[t+i]
    
    for t in ['BUNIT','BMAJ','BMIN','BPA','RESTFRQ']:
            if hd.get(t) != None:
                    mhd[t] = hd[t]
    
    return mhd


# Function to generate the integrated intensity map
def mom0map(hdu):

    hd = hdu.header
    data = hdu.data
    
    # Generate moment 0 map in K km/s
    mom0 = np.nansum(data,axis=0)*abs(hd['CDELT3'])/1000.
    
    return fits.PrimaryHDU(mom0,hd2d(hd))


def main(args):
    
    #%&%&%&%&%&%&%&%&%&%&%&%
    #    Make dendrogram
    #%&%&%&%&%&%&%&%&%&%&%&%
    print('Make dendrogram from the full cube')
    hdu = fits.open(args.fits_file)[0]
    data = hdu.data[701:1100,:,:]
    hd = hdu.header
    mhdu = mom0map(hdu)
                
    fig = aplpy.FITSFigure(mhdu, figsize=(8, 6), convention='wells')
    fig.show_colorscale(cmap='gray')#, vmax=36, stretch = 'sqrt')
    
#    fig.tick_labels.set_xformat('dd')
#    fig.tick_labels.set_yformat('dd')
    
    fig.add_colorbar()
    fig.colorbar.set_axis_label_text(r'[(K km/s)$^{1/2}$]')
    fig.save('c.png')

    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('fits_file', type=str, help='the input data file')
    args = parser.parse_args()
    start_time = time.time()
    main(args)
    print("--- {:.3f} seconds ---".format(time.time() - start_time))
