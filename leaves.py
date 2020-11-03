#!/usr/bin/python3

import argparse, os, time
import numpy as np
from astropy.io import fits
from astropy import units as u
import aplpy
import scimes
from astrodendro import Dendrogram, ppv_catalog

from matplotlib import pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib import colors as mcolors


def hd2d(hd):
# Function to remove the 3rd dimension in a 
# spectroscopic cube header

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



def main(args):
    
    #%&%&%&%&%&%&%&%&%&%&%&%
    #    Make dendrogram
    #%&%&%&%&%&%&%&%&%&%&%&%
    print('Make dendrogram from the full cube')
    hdu = fits.open(args.fits_file)[0]
    d0 = hdu.data
    nchan = d0.shape[0]
    ny = d0.shape[1]
    nx = d0.shape[2]
    d1 = d0[801:1000,:,:]
    hdr = hdu.header
    
    # Survey designs
    sigma = args.sigma #K, noise level
    ppb = args.ppb #pixels/beam

    def custom_independent(structure, index=None, value=None):
       peak_index, peak_value = structure.get_peak()
       return peak_value > 3.*sigma
                   
    d = Dendrogram.compute(d1, min_value=sigma, \
                            min_delta=args.delta*sigma, min_npix=1.*ppb, \
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
    
    #%&%&%&%&%&%&%&%&%&%&%&%&%&%
    #     Plot Leaves
    #%&%&%&%&%&%&%&%&%&%&%&%&%&%
    print("Plot Leaves")

    fig = plt.figure()
    for leaf in d.leaves:
        mask = leaf.get_mask()
        mask2d = mask.mean(axis=0)
        mask = np.zeros((ny,nx))
        mask[np.where(mask2d==0)] = True
        mask[np.where(mask2d!=0)] = False
        mask3d = np.repeat(mask[np.newaxis,:,:],nchan,axis=0)
        md = np.ma.masked_array(d0,mask=mask3d)
        sp = md.mean(axis=(1,2))
        plt.plot(sp)
#        break

    plt.show()


    
    
    #%&%&%&%&%&%&%&%&%&%&%&%&%&%
#    fig = plt.figure()
#    plt.imshow(np.nanmax(cube.data,axis=0),origin='lower',\
#                    interpolation='nearest',cmap='jet')
#    title = 'Leaves'
#    plt.title(title+' assignment map')
#    plt.colorbar(label='Structure label')
#    plt.xlabel('X [pixel]')
#    plt.ylabel('Y [pixel]')
    
    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('fits_file', type=str, help='the input data file')
    parser.add_argument('--sigma', type=float, default=0.01, help='the noise level')
    parser.add_argument('--ppb', type=float, default=9.0, help='the pixel per beam')
    parser.add_argument('--iter', type=int, default=10, help='number of k-means iteration')
    parser.add_argument('--delta', type=float, default=1., help='delta of sigma to asign a leaf')
    args = parser.parse_args()
    start_time = time.time()
    main(args)
    print("--- {:.3f} seconds ---".format(time.time() - start_time))
