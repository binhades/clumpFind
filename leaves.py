#!/usr/bin/python3

import argparse, os, time
import numpy as np
from toolkit import smooth
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.visualization import (MinMaxInterval, SqrtStretch, ImageNormalize)
from astrodendro import Dendrogram, ppv_catalog
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

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

def plot_spec(fig,x,y,ftsize='x-large',title=None,vline=None):
    ax = fig.add_subplot(1,2,1)
    ax.plot(x,y,lw=0.5)
    ax.set_xlim(-250,250)
    ax.set_ylim(-2.5,10.5)
    ax.set_xlabel('V$_{LSR}$ (km$\,$s$^{-1}$)',fontsize=ftsize)
    ax.set_ylabel('Flux (mJy$\,$beam$^{-1}$)',fontsize=ftsize)
    if title is not None:
        ax.set_title(title,fontsize=ftsize)
    if vline is not None:
        ax.axvline(x[vline],lw=2.,color='red')
    return 0

def plot_imag(fig,imag,mask,wcs,ftsize='x-large',coor=None,title=None):
    ax = fig.add_subplot(1,2,2,projection=wcs)

    # ------------------------
    # Display the image
    im = ax.imshow(imag,origin='lower',interpolation='nearest',cmap='hot',\
            aspect='equal',vmin=0.)#,norm=LogNorm()) #,vmin=0.0005,vmax=0.005
    ax.contour(mask,linewidths=2,levels=[0.001],alpha=0.8,colors='grey')
    # ------------------------
    # coordinates
    ra = ax.coords['ra']
    de = ax.coords['dec']
    ra.set_axislabel('R.A.',minpad=0.5,size=ftsize)
    de.set_axislabel('Dec.',minpad=0.5,size=ftsize)
    ra.set_separator(('$\mathrm{^h}$','$\mathrm{^m}$'))
    ra.set_ticklabel(size=ftsize)
    de.set_ticklabel(size=ftsize)
    # ------------------------
    if title is not None:
        ax.set_title(title,fontsize=ftsize)
    if coor is not None:
        ax.plot(coor[0],coor[1],marker='*')
    cbar = ax.figure.colorbar(im)
    cbar.ax.set_ylabel('mJy km$\,$s$^{-1}$',fontsize=ftsize)


def main(args):
    
    #%&%&%&%&%&%&%&%&%&%&%&%
    #    Make dendrogram
    #%&%&%&%&%&%&%&%&%&%&%&%
    print('Make dendrogram from the full cube')
    hdu = fits.open(args.fits_file)[0]
    hdr = hdu.header
    wcs = WCS(hdr,'degree').celestial
    data = hdu.data
    nchan = data.shape[0]
    velo = (np.arange(nchan) - hdr['CRPIX3'] + 1) * hdr['CDELT3'] + hdr['CRVAL3']
    ny = data.shape[1]
    nx = data.shape[2]
    d = Dendrogram.load_from(args.file_d+'.hdf5')
   
    #%&%&%&%&%&%&%&%&%&%&%&%&%&%
    #     Plot Leaves
    #%&%&%&%&%&%&%&%&%&%&%&%&%&%
    print("Plot Leaves")

    fig = plt.figure(figsize=(12, 4))
    for leaf in d.leaves:
        print(leaf.idx)
        file_out = 'leaf'+str(leaf.idx)+'.png'
        peak = leaf.get_peak()[0]
        v_p = args.chan_0+peak[0]
        x_p = peak[2]
        y_p = peak[1]

        coor = SkyCoord.from_pixel(x_p,y_p,wcs)
        gc = coor.transform_to('galactic')
        equ_str = coor.to_string(style='hmsdms',precision=0)
        #gal_str = gc.to_string(style='decimal',precision=2)
        gal_str = 'G{:5.2f}{:+5.2f}'.format(gc.l.value,gc.b.value)
        title0 = equ_str + ' @ '+str(velo[v_p]) + ' km/s'
        title1 = gal_str + ' @ '+str(velo[v_p]) + ' km/s'

        imag  = data[v_p-2:v_p+2,:,:].sum(axis=0) * hdr['CDELT3'] * 1000

        mask2d = leaf.get_mask().mean(axis=0)
        mask = np.zeros((ny,nx))
        mask[np.where(mask2d==0)] = True
        mask[np.where(mask2d!=0)] = False
        mask3d = np.repeat(mask[np.newaxis,:,:],nchan,axis=0)

        md = np.ma.masked_array(data,mask=mask3d)
        sp = md.mean(axis=(1,2))
        spsm = smooth(sp,window_len=5) * 1000.

        plot_spec(fig,velo,spsm,vline=v_p,title=title0)
        plot_imag(fig,imag,mask2d,wcs,coor=(x_p,y_p),title=title1)
        fig.savefig(file_out,dpi=300,format='png',bbox_inches='tight')
        plt.clf()
    
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
