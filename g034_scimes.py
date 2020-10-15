#!/usr/bin/python3

import argparse, os, time
import numpy as np
from matplotlib import pyplot as plt
from astropy.io import fits
from astropy import units as u
import aplpy
import scimes
from astrodendro import Dendrogram, ppv_catalog

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
    
    # Survey designs
    sigma = args.sigma #K, noise level
    ppb = args.ppb #pixels/beam
                    
    d = Dendrogram.compute(data, min_value=sigma, \
                            min_delta=0.5*sigma, min_npix=1.*ppb, \
                            verbose = 1)
    
    
    #%&%&%&%&%&%&%&%&%&%&%&%&%&%
    #   Generate the catalog
    #%&%&%&%&%&%&%&%&%&%&%&%&%&%
    print("Generate a catalog of dendrogram structures")
    metadata = {}
    metadata['data_unit'] = u.Jy #This should be Kelvin (not yet implemented)!
    cat = ppv_catalog(d, metadata)
    
    
    #%&%&%&%&%&%&%&%&%&%&%&%&%&%
    #     Running SCIMES
    #%&%&%&%&%&%&%&%&%&%&%&%&%&%
    print("Running SCIMES")
    dclust = scimes.SpectralCloudstering(d, cat, hd, rms=sigma, \
                            user_iter=args.iter, \
                            )
    
    
    #%&%&%&%&%&%&%&%&%&%&%&%&%&%
    #     Image the result
    #%&%&%&%&%&%&%&%&%&%&%&%&%&%
    print("Visualize the clustered dendrogram")
    dclust.showdendro()
    
    print("Visualize collapsed maps of the assignment cubes")
    cubes = [dclust.clusters_asgn,\
                    dclust.leaves_asgn,\
                    dclust.trunks_asgn]
    titles = ['Clusters', 'Leaves', 'Trunks']
    
    for cube, title in zip(cubes, titles):
    
        plt.figure()
        plt.imshow(np.nanmax(cube.data,axis=0),origin='lower',\
                        interpolation='nearest',cmap='jet')
        plt.title(title+' assignment map')
        plt.colorbar(label='Structure label')
        plt.xlabel('X [pixel]')
        plt.ylabel('Y [pixel]')
        plt.show()
    
    
    print("Image the results with APLpy")
    
    clusts = dclust.clusters
    colors = dclust.colors
    # Create Orion integrated intensity map
    mhdu = mom0map(hdu)
                
    fig = aplpy.FITSFigure(mhdu, figsize=(8, 6), convention='wells')
    fig.show_colorscale(cmap='gray')#, vmax=36, stretch = 'sqrt')
    
    
    count = 0
    for c in clusts:
    
        mask = d[c].get_mask()
        mask_hdu = fits.PrimaryHDU(mask.astype('short'), hdu.header)
        
        mask_coll = np.amax(mask_hdu.data, axis = 0)
        mask_coll_hdu = fits.PrimaryHDU(mask_coll.astype('short'), hd2d(hdu.header))
                        
        fig.show_contour(mask_coll_hdu, colors=colors[count], linewidths=1, convention='wells', levels = [0])
    
        count = count+1
        print(count)
                    
    fig.tick_labels.set_xformat('dd')
    fig.tick_labels.set_yformat('dd')
    
    fig.add_colorbar()
    fig.colorbar.set_axis_label_text(r'[(K km/s)$^{1/2}$]')
    fig.save('b.png')

    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('fits_file', type=str, help='the input data file')
    parser.add_argument('--sigma', type=float, default=0.01, help='the noise level')
    parser.add_argument('--ppb', type=float, default=9.0, help='the pixel per beam')
    parser.add_argument('--iter', type=int, default=10, help='the pixel per beam')
    args = parser.parse_args()
    start_time = time.time()
    main(args)
    print("--- {:.3f} seconds ---".format(time.time() - start_time))
