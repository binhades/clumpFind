#!/usr/bin/python3

import argparse, os, time
import numpy as np
from toolkit import smooth, colormap
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astrodendro import Dendrogram
from matplotlib import pyplot as plt
import matplotlib.patches as patches

def plot_imag(fig,imag,mask,wcs,ftsize='xx-large',coor=None,title=None,beam=None,cmap='hot',colorbar=False,size=10):
    ax = fig.add_subplot(1,1,1,projection=wcs)
    # ------------------------
    # Display the image
    im = ax.imshow(imag,origin='lower',interpolation='nearest',cmap=colormap(cmap),\
            aspect='equal',vmin=0.)
    ax.contour(mask,linewidths=4,levels=[0.001],colors='blue')
    # ------------------------
    # coordinates
    ra = ax.coords['ra']
    de = ax.coords['dec']
    ra.set_axislabel('R.A. (J2000)',minpad=0.5,size=ftsize)
    de.set_axislabel('Dec. (J2000)',minpad=0.5,size=ftsize)
    ra.set_separator(('$\mathrm{^h}$','$\mathrm{^m}$'))
    ra.set_ticklabel(size=ftsize)
    de.set_ticklabel(size=ftsize)
    # ------------------------
    if title is not None:
        ax.set_title(title,fontsize=ftsize)
    if coor is not None:
        ax.set_xlim(coor[0]-size,coor[0]+size)
        ax.set_ylim(coor[1]-size,coor[1]+size)
    if colorbar:
        cbar = ax.figure.colorbar(im)
        cbar.ax.set_ylabel('mJy$\,$beam$^{-1}$ km$\,$s$^{-1}$',fontsize=ftsize)
    if beam is not None:
        bm = patches.Circle((5,5),radius=args.beam/2,edgecolor='k',facecolor='w',alpha=0.5) # pixel coordinates
        ax.add_patch(bm)

    return ax

def main(args):
    #------------------------
    #    Load DataCube
    #------------------------
    print('Load DataCube')
    hdu = fits.open(args.fits_file)[0]
    hdr = hdu.header
    wcs = WCS(header=hdr).celestial
    data = hdu.data
    nchan = data.shape[0]
    velo = (np.arange(nchan) - hdr['CRPIX3'] + 1) * hdr['CDELT3'] + hdr['CRVAL3']
    ny = data.shape[1]
    nx = data.shape[2]
    #------------------------
    # unit convert: Jy/beam -> mJy/pix
    #------------------------
    beam = 4.7 # arcmin
    pix = 1.0 # arcmin
    pix_over_beam = pix**2/((beam/2)**2*np.pi)
    data = data * 1000 * pix_over_beam # x Jy/beam = (x * pix/beam) Jy/pix
    #------------------------
    #    Load contour data
    #------------------------
    if args.file_contour is not None:
        hdu1 = fits.open(args.file_contour)[0]
        hdr1= hdu1.header
        wcs1 = WCS(header=hdr).celestial
        contour_data = hdu1.data

    #------------------------
    #    Load dendrogram
    #------------------------
    print('Load Dendrogram')
    d = Dendrogram.load_from(args.file_d+'.hdf5')
    print('')
    #------------------------
    # Plot all leaves and branches on full frame of Moment0 map.
    #------------------------
    fig = plt.figure(figsize=(8,8))

    for i, struc in enumerate(d.leaves):

        mask = struc.get_mask().mean(axis=0)
        peak = struc.get_peak()[0]
        x_p = peak[2]
        y_p = peak[1]
        v_p = args.chan_0+peak[0]
        imag  = data[v_p-10:v_p+10,:,:].sum(axis=0) * hdr['CDELT3'];

        coor = SkyCoord.from_pixel(x_p,y_p,wcs)
        gc = coor.transform_to('galactic')
        gname = 'G{:5.2f}{:+5.2f}'.format(gc.l.value,gc.b.value)
        file_out = '{}_{}.png'.format(gname,args.cmap)

        ax = plot_imag(fig,imag,mask,wcs,coor=(x_p,y_p),cmap=args.cmap,size=args.size,beam=args.beam,ftsize=20)
        if args.file_contour is not None:
            ax.contour(contour_data,linewidths=1.5,levels=args.levels,colors=args.contour_color)
        fig.savefig(file_out,dpi=300,format='png',bbox_inches='tight')
        fig.clear(True)


    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('fits_file', type=str, help='the input data file')
    parser.add_argument('--file_d', type=str, default='my_dendrogram', help='the dendrogram file')
    parser.add_argument('--file_contour', type=str, help='the hna moment0 fits file as contour')
    parser.add_argument('--contour_color', type=str, default='#d0d0d0', help='the contour color')
    parser.add_argument('--levels', nargs='+',type=float, help='contour levels')
    parser.add_argument('--chan_0', type=int, default=0,  help='channel index start')
    parser.add_argument('--chan_1', type=int, default=-1, help='channel index end')
    parser.add_argument('--cmap', type=str, default='hot', help='the colormap, batlow or lajolla')
    parser.add_argument('--plot_spec', action='store_true', help='set to plot spectra of individual leaves')
    parser.add_argument('--beam', type=float, help='set to add beam to image, size in pixel. 4.7 for FAST rrl cube')
    parser.add_argument('--size', type=int, help='half size in pixel')
    args = parser.parse_args()
    start_time = time.time()
    main(args)
    print("--- {:.3f} seconds ---".format(time.time() - start_time))
