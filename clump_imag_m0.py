#!/usr/bin/python3
# copy from clump_plot_all.py
# to only create the overall image with leaves and branches on m0 map
# choose to overlay sources in csv catalog

import argparse, os, time
import numpy as np
from toolkit import smooth
from toolkit import colormap
from astropy.io import fits
from astropy.wcs import WCS
from astropy.visualization import simple_norm
from astropy import units as u
from astropy.coordinates import SkyCoord
from astrodendro import Dendrogram, ppv_catalog
from matplotlib import pyplot as plt
from matplotlib import cm
import matplotlib.patches as patches
from matplotlib.colors import LogNorm

def add_leaf_label(ax, idx_arr, struc):
    leaf_label = np.argwhere(idx_arr == struc.idx)[0][0]+1
    peak = struc.get_peak()[0]
    x_p = peak[2]
    y_p = peak[1]
    ax.text(x_p-1,y_p+1,str(leaf_label),fontsize='x-large',color='k')

    return leaf_label

def load_csv(file_csv):
    import csv
    catalog=[]
    with open(file_csv,'rt') as fcsv:
        reader = csv.DictReader(fcsv)
        for row in reader:
            catalog.append({'GName':row['GName'],'RA':row['RA'],'Dec':row['Dec']})
    return catalog

def add_source(ax, file_csv, wcs):

    catalog = load_csv(file_csv)

    for item in catalog:
        ra = item['RA']
        dec = item['Dec']
        pos = SkyCoord(ra,dec,frame='fk5',unit=(u.hourangle, u.deg))
        ax.scatter(pos.ra.degree, pos.dec.degree,200,marker='*',color='#00FFFF',transform=ax.get_transform('fk5'))
        #ax.text(pos.ra.degree, pos.dec.degree,item['GName'],color='#00FFFF',transform=ax.get_transform('fk5'))

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

    # unit convert: Jy/beam -> mJy/pix
    beam = 4.7 # arcmin
    pix = 1.0 # arcmin
    pix_over_beam = pix**2/((beam/2)**2*np.pi)
    data = data * 1000 * pix_over_beam # x Jy/beam = (x * pix/beam) Jy/pix

    #------------------------
    #    Load dendrogram
    #------------------------
    print('Load Dendrogram')
    d = Dendrogram.load_from(args.file_d+'.hdf5')
   
    #------------------------
    # Plot all leaves and branches on full frame of Moment0 map.
    #------------------------
    imag0 = data[args.chan_0:args.chan_1,:,:].mean(axis=0)
    norm = simple_norm(imag0,stretch='asinh',asinh_a=0.18,min_percent=5,max_percent=100)

    fig0 = plt.figure(figsize=(8,8))
    ax0  = fig0.add_subplot(1,1,1,projection=wcs)
    im0 = ax0.imshow(imag0,origin='lower',interpolation='nearest',cmap=colormap(args.cmap),\
            aspect='equal',norm=norm) 
    # coordinates
    if args.proj == 'equ':
        ra0 = ax0.coords['ra']
        de0 = ax0.coords['dec']
        ra0.set_axislabel('R.A. (J2000)',minpad=0.5,size="xx-large")
        de0.set_axislabel('Dec. (J2000)',minpad=0.5,size="xx-large")
        ra0.set_separator(('$\mathrm{^h}$','$\mathrm{^m}$'))
        ra0.set_ticklabel(size="xx-large")
        de0.set_ticklabel(size="xx-large")
    elif args.proj == 'gal':
        l0 = ax0.coords['glon']
        b0 = ax0.coords['glat']
        l0.set_axislabel('l deg',minpad=0.5,size="xx-large")
        b0.set_axislabel('b deg',minpad=0.5,size="xx-large")
        l0.set_ticklabel(size="xx-large")
        b0.set_ticklabel(size="xx-large")

    # beam
    if args.beam is not None:
        beam = patches.Circle((5,5),radius=args.beam/2,edgecolor='k',facecolor='w',alpha=0.5) # pixel coordinates
        ax0.add_patch(beam)
    # ------------------------
    # add sources in catalog
    if args.file_c is not None:
        add_source(ax0,args.file_c,wcs)
    # ------------------------
    # leaf label
    list_idx = [] # raw index
    list_idv = [] # sorted index
    list_peak= [] # raw peaks
    for i, struc in enumerate(d.leaves):
        peak = struc.get_peak()[1]
        list_peak.append(peak)
        list_idx.append(struc.idx)
    peak_ind = np.argsort(np.array(list_peak))[::-1]
    leaves_idx_arr = np.array(list_idx)[peak_ind]
    # ------------------------

    colors = ['blue','red','green']
    c=1
    for i, struc in enumerate(d.trunk):
        if struc.is_leaf:
            leaf_label = add_leaf_label(ax0,leaves_idx_arr,struc)
            file_out = 'leaf_{:d}_{}.png'.format(leaf_label,args.cmap)
            color = colors[0]
            subtree = []
            line = 'solid'
        elif struc.is_branch: # branch
            file_out = 'branch_{:d}_{}.png'.format(struc.idx,args.cmap)
            color=colors[c]
            c=c+1
            subtree = struc.descendants
            line = 'dashed'

        mask = struc.get_mask().mean(axis=0)
        ax0.contour(mask,linewidths=1.5,levels=[0.001],alpha=0.8,colors=[color],linestyles=line)

        for j, sub_struc in enumerate(subtree):
            if sub_struc.is_leaf:
                leaf_label = add_leaf_label(ax0,leaves_idx_arr,sub_struc)
                file_out = 'leaf_{:d}_{}.png'.format(leaf_label,args.cmap)
                mask = sub_struc.get_mask().mean(axis=0)
                ax0.contour(mask,linewidths=1.5,levels=[0.001],alpha=0.8,colors=[color])

    fig0.savefig('m0-clumps_{}.png'.format(args.cmap),dpi=300,format='png',bbox_inches='tight')

    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('fits_file', type=str, help='the input data file')
    parser.add_argument('--file_d', type=str, default='my_dendrogram', help='the dendrogram file')
    parser.add_argument('--file_c', type=str, help='the catalog file to add source on map')
    parser.add_argument('--chan_0', type=int, default=0,  help='channel index start')
    parser.add_argument('--chan_1', type=int, default=-1, help='channel index end')
    parser.add_argument('--cmap', type=str, default='hot', help='the colormap, batlow or lajolla')
    parser.add_argument('--beam', type=float, help='set to add beam to image, size in pixel. 4.7 for FAST rrl cube')
    parser.add_argument('--proj', type=str, default='equ', help='equ or gal, projection of the map coordinate system')
    args = parser.parse_args()
    start_time = time.time()
    main(args)
    print("--- {:.3f} seconds ---".format(time.time() - start_time))
