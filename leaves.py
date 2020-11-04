#!/usr/bin/python3

import argparse, os, time
import numpy as np
from toolkit import smooth
from astropy.io import fits
from astrodendro import Dendrogram, ppv_catalog
from matplotlib import pyplot as plt


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

def plot_spec(fig,x,y):
    ax = fig.add_subplot(1,2,1)
    ax.plot(x,y,lw=0.5)
    ax.set_ylim(-2.5,10.5)
    ax.set_xlabel('V$_{LSR}$ (km$\,$s$^{-1}$)')
    ax.set_ylabel('Flux (Jy$\,$beam$^{-1}$)')
    return 0

def plot_imag(fig,imag,mask)
    ax = fig.add_subplot(1,2,2)
    ax.imshow(imag,origin='lower',interpolation='nearest',cmap='hot',\
            aspect='equal',vmin=0.0005,vmax=0.005)
    ax.contour(mask,linewidths=2,levels=[0.001],alpha=0.8)

def main(args):
    
    #%&%&%&%&%&%&%&%&%&%&%&%
    #    Make dendrogram
    #%&%&%&%&%&%&%&%&%&%&%&%
    print('Make dendrogram from the full cube')
    hdu = fits.open(args.fits_file)[0]
    hdr = hdu.header
    d0 = hdu.data
    nchan = d0.shape[0]
    ny = d0.shape[1]
    nx = d0.shape[2]
    d = Dendrogram.load_from(args.file_d+'.hdf5')
   
    #%&%&%&%&%&%&%&%&%&%&%&%&%&%
    #     Plot Leaves
    #%&%&%&%&%&%&%&%&%&%&%&%&%&%
    print("Plot Leaves")

    #fig, axs = plt.subplots(1, 2, figsize=(9, 4))
    fig = plt.figure(figsize=(9, 4))
    for leaf in d.leaves:
        print(leaf.idx)
        file_out = 'leaf'+str(leaf.idx)+'.png'
        peak = leaf.get_peak()[0]
        ind_p = args.chan_0+peak[0]

        imag  = d0[ind_p-10:ind_p+10,:,:].mean(axis=0)

        mask2d = leaf.get_mask().mean(axis=0)
        mask = np.zeros((ny,nx))
        mask[np.where(mask2d==0)] = True
        mask[np.where(mask2d!=0)] = False
        mask3d = np.repeat(mask[np.newaxis,:,:],nchan,axis=0)

        md = np.ma.masked_array(d0,mask=mask3d)
        sp = md.mean(axis=(1,2))
        spsm = smooth(sp,window_len=5) * 1000.

        plot_spec(fig,velo,spsm)
        plot_imag(fig,imag,mask2d)
        fig.savefig(file_out,dpi=300,format='png',bbox_inches='tight')
        plt.clf()
#        break

#    plt.show()


    
    
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
    parser.add_argument('--file_d', type=str, default='my_dendrogram', help='the dendrogram file')
    parser.add_argument('--chan_0', type=int, default=0,  help='channel index start')
    parser.add_argument('--chan_1', type=int, default=-1, help='channel index end')
    args = parser.parse_args()
    start_time = time.time()
    main(args)
    print("--- {:.3f} seconds ---".format(time.time() - start_time))
