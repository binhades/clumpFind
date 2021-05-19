#!/usr/bin/python3
# copy from clump_plot_all.py
# to only create the overall image with leaves and branches on m0 map
# choose to overlay sources in csv catalog

import numpy as np
import aplpy
from astropy.coordinates import Galactic
import matplotlib.pyplot as plt
import argparse, os
from toolkit import colormap

#import matplotlib as mpl
#mpl.rcParams['xtick.labelsize']='xx-large'
#mpl.rcParams['ytick.labelsize']='xx-large'

import argparse, os, time

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

def plot_image(file_in,file_out=None,file_contour=None,file_reg=None,resize=None,contour=False,levels=None,contour_color='#d0d0d0',plot=False,oformat='png',skycoor='equ',vmin=None,vmax=None,pmin=0.25,pmax=99.75,stretch='linear',vmid=None,exponent=2,cmap='hot',beam=None,colorbar=None,dendro=None,file_catalog=None,addgal=True,save=True):
    fig = aplpy.FITSFigure(file_in)
    fig.show_colorscale(vmin=vmin,vmax=vmax,pmin=pmin,pmax=pmax,cmap=colormap(cmap),aspect='equal',smooth=1,stretch=stretch,vmid=vmid,exponent=exponent)

#    fig.hide_colorscale()
    if resize is not None:
        fig.recenter(resize[0],resize[1],width=resize[2],height=resize[3])
#-------------------------------------------------------------------
#   ADD beam
    if beam is not None:
        fig.add_beam(major=beam/60,minor=beam/60,angle=0,corner='bottom left')
        fig.beam.set_alpha(0.5)
        fig.beam.set_edgecolor('blue')
        fig.beam.set_facecolor('white')
        fig.beam.set_linewidth(3)
#-------------------------------------------------------------------
#    fig.show_circles([50.05,], [-0.85,], 0.028,color='white',linewidth=1)
#    fig.add_label(50.05, -0.95,"FWHM 3.4'",color='white')
#-------------------------------------------------------------------
    if addgal:
        l = np.arange(30,70,0.05)
        b0 = np.zeros(int((70-30)/0.05))
        bp = np.zeros(int((70-30)/0.05)) + 0.5
        bm = np.zeros(int((70-30)/0.05)) - 0.5

        gal_c0 = SkyCoord(l,b0,frame=Galactic,unit="deg")
        gal_cp = SkyCoord(l,bp,frame=Galactic,unit="deg")
        gal_cm = SkyCoord(l,bm,frame=Galactic,unit="deg")

        ra0 = gal_c0.icrs.ra.value
        rap = gal_cp.icrs.ra.value
        ram = gal_cm.icrs.ra.value
        dec0 = gal_c0.icrs.dec.value
        decp = gal_cp.icrs.dec.value
        decm = gal_cm.icrs.dec.value

        fig.show_lines([np.array([ra0,dec0]),np.array([rap,decp]),np.array([ram,decm])], \
                color='darkgrey', linewidths=2, linestyle=['-','--','--'])
        fig.add_label(283.95, 2.18,r"b=$0^{\circ}$",color='black',fontsize=14)
        fig.add_label(283.38, 2.18,r"b=$0.5^{\circ}$",color='black',fontsize=14)
        fig.add_label(284.42, 1.8,r"b=$-0.5^{\circ}$",color='black',fontsize=14)


    if skycoor == 'gal':
        fig.axis_labels.set_xtext('Galactic Longitude')
        fig.axis_labels.set_ytext('Galactic Latitude')
        fig.tick_labels.set_xformat('dd.d')
        fig.tick_labels.set_yformat('dd.d')
        fig.ticks.set_xspacing(0.5)
        fig.ticks.set_yspacing(0.5)
        #fig.ticks.set_minor_frequency(5)
    else:
        z=0

    fig.axis_labels.set_font(size='xx-large')
    fig.tick_labels.set_font(size='xx-large')

    fig.ticks.show_x()
    fig.ticks.show_y()

    fig.ticks.set_linewidth(1)
    fig.ticks.set_length(2)
    fig.ticks.set_color('black')
    fig.ticks.show()
    fig.set_nan_color('white')
    if colorbar is not None:
        fig.add_colorbar()
        fig.colorbar.show()
        fig.colorbar.set_axis_label_text(colorbar)
        fig.colorbar.set_axis_label_font(size='xx-large')
        fig.colorbar.set_font(size='xx-large')
#-------------------------------------------
# DS9 regions
    if file_reg is not None:
        fig.show_regions(file_reg)

#-------------------------------------------
# add contour 
#-------------------------------------------
    if contour:
        if levels is not None:
            if file_contour is None:
                file_contour = file_in
            fig.show_contour(file_contour,levels=levels,smooth=1,colors=contour_color, linewidths=1)
        else:
            print("contour levels is not set")

#------------------------
# Plot all leaves and branches on full frame of Moment0 map.
#------------------------
    # add sources in catalog
#    add_source(ax0,catalog,wcs)
    # ------------------------
    # leaf label
    list_idx = [] # raw index
    list_idv = [] # sorted index
    list_peak= [] # raw peaks
    for i, struc in enumerate(dendro.leaves):
        peak = struc.get_peak()[1]
        list_peak.append(peak)
        list_idx.append(struc.idx)
    peak_ind = np.argsort(np.array(list_peak))[::-1]
    leaves_idx_arr = np.array(list_idx)[peak_ind]
    # ------------------------

    colors = ['blue','red','green']
    c=1
    for i, struc in enumerate(dendro.trunk):
        if struc.is_leaf:
            leaf_label = add_leaf_label(fig,leaves_idx_arr,struc)
            color = colors[0]
            subtree = []
            line = 'solid'
        elif struc.is_branch: # branch
            color=colors[c]
            c=c+1
            subtree = struc.descendants
            line = 'dashed'

        mask = struc.get_mask().mean(axis=0)

        fig.show_contour(mask,linewidths=1.5,levels=[0.001],alpha=0.8,colors=[color],linestyles=line)#,smooth=1)
        fig.add_label(283.95, 2.18,r"b=$0^{\circ}$",color='black',fontsize=14)

        print(color)

        for j, sub_struc in enumerate(subtree):
            if sub_struc.is_leaf:
                leaf_label = add_leaf_label(fig,leaves_idx_arr,sub_struc)
                mask = sub_struc.get_mask().mean(axis=0)
                fig.show_contour(mask,linewidths=1.5,levels=[0.001],alpha=0.8,colors=[color])#,smooth=1)

    if file_catalog is not None:
        add_source(fig,file_catalog)

    if save:
        if file_out is None:
            file_out='m0-clumps_{}.png'.format(cmap)
        fig.save(file_out,dpi=300,format=oformat,adjust_bbox=True)

    if plot:
        plt.show()
    fig.close()
    return 0

def add_leaf_label(fig, idx_arr, struc):
    leaf_label = np.argwhere(idx_arr == struc.idx)[0][0]+1
    peak = struc.get_peak()[0]
    x_p = peak[2]
    y_p = peak[1]
    xw, yw = fig.pixel2world(x_p-1,y_p+1)
    fig.add_label(float(xw),float(yw),str(leaf_label),fontsize='x-large',color='k')

    return leaf_label

def load_csv(file_csv):
    import csv
    catalog=[]
    with open(file_csv,'rt') as fcsv:
        reader = csv.DictReader(fcsv)
        for row in reader:
            catalog.append({'GName':row['GName'],'RA':row['RA'],'Dec':row['Dec']})
    return catalog

def add_source(fig, file_catalog):
    catalog = load_csv(file_catalog)
    for item in catalog:
        ra = item['RA']
        dec = item['Dec']
        pos = SkyCoord(ra,dec,frame='fk5',unit=(u.hourangle, u.deg))
        fig.show_markers(pos.ra.degree, pos.dec.degree,marker='*',s=200,edgecolor='#00FFFF',facecolor='#00FFFF')

def main(args):

    #------------------------
    #    Load dendrogram
    #------------------------
    print('Load Dendrogram')
    d = Dendrogram.load_from(args.file_d+'.hdf5')
 
    plot_image(args.file_map,args.file_out,file_reg=args.file_reg,file_contour=args.file_contour,\
            plot=args.plot,oformat=args.format,resize=args.resize,\
            contour=args.contour,contour_color=args.contour_color,\
            levels=args.levels,skycoor=args.skycoor,beam=args.beam,\
            vmin=args.vmin,vmax=args.vmax,pmin=args.pmin,pmax=args.pmax,\
            stretch=args.stretch,cmap=args.cmap,colorbar=args.colorbar,\
            dendro=d,file_catalog=args.file_catalog)

    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('file_map', type=str, help='the input data file')
    parser.add_argument('--file_d', type=str, default='my_dendrogram', help='the dendrogram file')
    parser.add_argument('--file_catalog', type=str, help='the catalog file to add source on map')
    parser.add_argument('--chan_0', type=int, default=0,  help='channel index start')
    parser.add_argument('--chan_1', type=int, default=-1, help='channel index end')
    parser.add_argument('--cmap', type=str, default='hot', help='the colormap, batlow or lajolla')
    parser.add_argument('--beam', type=float, help='set to add beam to image, size in pixel. 4.7 for FAST rrl cube')
    parser.add_argument('--proj', type=str, default='equ', help='equ or gal, projection of the map coordinate system')

    parser.add_argument('--file_out', type=str, help='Output file name for the figure')
    parser.add_argument('--file_contour', type=str, help='File name for the contour, use file_in if not set')
    parser.add_argument('--file_reg', type=str, help='DS9 region file to plot')
    parser.add_argument('--format', type=str, default='png', help='output file format')
    parser.add_argument('--skycoor', type=str, default='equ', help='sky coordinate system')
    parser.add_argument('--resize', metavar='Deg', type=float, nargs=4, help='resize the map: x_center, y_center, x_size, y_size')
    parser.add_argument('--contour', action='store_true', help='set to add contour')
    parser.add_argument('--levels', nargs='+', type=float, help='set contour levels')
    parser.add_argument('--contour_color', type=str, default='#d0d0d0', help='contour color')
    parser.add_argument('--plot', action='store_true', help='set to show plot')
    parser.add_argument('--vmin', type=float, help='Minimum pixel value to use for the colorscale.')
    parser.add_argument('--vmax', type=float, help='Maximum pixel value to use for the colorscale.')
    parser.add_argument('--colorbar', type=str, help='the colorbar title if is to show')
    parser.add_argument('--pmin', type=float, help='Percentile value used to determine the Minimum pixel value to use for the colorscale.')
    parser.add_argument('--pmax', type=float, help='Percentile value used to determine the Maximum pixel value to use for the colorscale.')
    parser.add_argument('--stretch', type=str, default='linear', help='the stretch function to use')


    args = parser.parse_args()
    start_time = time.time()
    main(args)
    print("--- {:.3f} seconds ---".format(time.time() - start_time))



