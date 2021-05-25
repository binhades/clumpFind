#!/usr/bin/python3
# load the image of vlsr 
# make mask with NaN values outside branches and inside leaves.

import argparse, os, time, copy
import aplpy
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord, Galactic
from astropy.io import fits
#from astropy import units as u
from astrodendro import Dendrogram
#from toolkit import smooth

#mpl.rcParams['xtick.labelsize']='xx-large'
#mpl.rcParams['ytick.labelsize']='xx-large'


def plot_image(file_fits,file_velo,file_out=None,file_sour=None,file_contour=None,file_reg=None,\
               resize=None,contour=False,levels=None,contour_color='#d0d0d0',plot=False,oformat='png',\
               skycoor='equ',vmin=None,vmax=None,pmin=0.25,pmax=99.75,stretch='linear',vmid=None,\
               exponent=2,beam=None,colorbar=None,dendro=None,addgal=True,save=True):
    fig = aplpy.FITSFigure(file_fits)
    fig.show_colorscale(vmin=vmin,vmax=vmax,pmin=pmin,pmax=pmax,aspect='equal',smooth=1,stretch=stretch,vmid=vmid,exponent=exponent)
    fig.hide_colorscale()

#    fig.hide_colorscale()
    if resize is not None:
        fig.recenter(resize[0],resize[1],width=resize[2],height=resize[3])
#-------------------------------------------------------------------
#   ADD beam
    if beam is not None:
        fig.add_beam(major=beam/60,minor=beam/60,angle=0,corner='bottom left')
        #fig.beam.set_alpha(0.5)
        fig.beam.set_edgecolor('black')
        fig.beam.set_facecolor('white')
        fig.beam.set_linewidth(1.5)
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
                file_contour = file_fits
            fig.show_contour(file_contour,levels=levels,smooth=1,colors=contour_color, linewidths=1)
        else:
            print("contour levels is not set")

#------------------------
# Plot all leaves and branches on full frame of Moment0 map.
#------------------------
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
    v_cmap = copy.copy(mpl.cm.get_cmap('viridis'))
    v_cmap = copy.copy(mpl.cm.get_cmap('plasma'))
    v_cmap = copy.copy(mpl.cm.get_cmap('cividis'))
    v_cmap = copy.copy(mpl.cm.get_cmap('seismic'))
    v_cmap = copy.copy(mpl.cm.get_cmap('Reds'))
    v_cmap = copy.copy(mpl.cm.get_cmap('coolwarm'))


    cata_velo = load_csv(file_velo)
    add_source_velo(fig,cata_velo)
    norm = mpl.colors.Normalize(vmin=45,vmax=60)
    count_branch=0

    for i, struc in enumerate(dendro.trunk):
        if struc.is_leaf:
            leaf_label = add_leaf_label(fig,leaves_idx_arr,struc,draw=False)
            leaf_velo = get_source_velo(leaf_label,cata_velo)
            color = colors[0]
            subtree = []
            line = 'solid'
            mask = struc.get_mask().mean(axis=0)
            if leaf_label == 19:
                continue
            fig.show_contour(mask,levels=[0.001,1000],alpha=0.95,filled=True,colors=[v_cmap(norm(leaf_velo))],linestyles=line)#,smooth=1)
        elif struc.is_branch: # branch
            count_branch=count_branch+1
            color=colors[count_branch]
            subtree = struc.descendants
            line = 'dashed'

        for j, sub_struc in enumerate(subtree):
            if sub_struc.is_leaf:
                leaf_label = add_leaf_label(fig,leaves_idx_arr,sub_struc,draw=False)
                leaf_velo = get_source_velo(leaf_label,cata_velo)
                if leaf_label == 14 or leaf_label == 15:
                    continue
                mask = sub_struc.get_mask().mean(axis=0)
                fig.show_contour(mask,levels=[0.001,1000],alpha=0.95,filled=True,colors=[v_cmap(norm(leaf_velo))])#,smooth=1)

    # add sources in catalog
    if file_sour is not None:
        add_source(fig,file_sour,v_cmap,norm)


    if save:
        if file_out is None:
            file_out='clumps_velo_dist.png'
        fig.save(file_out,dpi=300,format=oformat,adjust_bbox=True)

    if plot:
        plt.show()
    fig.close()
    return 0

def add_leaf_label(fig, idx_arr, struc,draw=True):
    leaf_label = np.argwhere(idx_arr == struc.idx)[0][0]+1
    peak = struc.get_peak()[0]
    x_p = peak[2]
    y_p = peak[1]
    xw, yw = fig.pixel2world(x_p-1,y_p+1)
    if draw:
        fig.add_label(float(xw),float(yw),str(leaf_label),fontsize='x-large',color='k')

    return leaf_label

def load_csv(file_csv):
    import csv
    catalog=[]
    with open(file_csv,'rt') as fcsv:
        reader = csv.DictReader(fcsv)
        for row in reader:
            catalog.append({'Index':int(row['Index']),'GLon':float(row['GLon']),'GLat':float(row['GLat']),'VLSR':float(row['VLSR'])})
    return catalog

def get_source_velo(ind0, catalog):
    for item in catalog:
        v = item['VLSR']
        ind = item['Index']
        if ind == ind0:
            return v

    print('Index not matching with Catalog, -1')
    return -1


def add_source(fig, file_catalog,cmap,norm):
    catalog = load_csv(file_catalog)
    for item in catalog:
        l = item['GLon']
        b = item['GLat']
        v = item['VLSR']
        ind = item['Index']
        if ind == 3:
            offset = -0.03
        else:
            offset = 0.03
        pos = SkyCoord(l,b,frame=Galactic,unit='deg')
        #fig.show_markers(pos.icrs.ra.value, pos.icrs.dec.value,marker='o',s=200,facecolor=cmap(norm(v)))
        fig.show_markers(pos.icrs.ra.value, pos.icrs.dec.value,marker='o',s=100,edgecolor='gray',facecolor='green')
        fig.add_label(pos.icrs.ra.value, pos.icrs.dec.value+offset,'{:4.1f}'.format(v),fontsize=12,color='green')

def add_source_velo(fig, catalog):
    for item in catalog:
        l = item['GLon']
        b = item['GLat']
        v = item['VLSR']
        ind = item['Index']
        pos = SkyCoord(l,b,frame=Galactic,unit='deg')
        if ind == 14 or ind == 15 or ind == 19:
            continue

        fig.add_label(pos.icrs.ra.value, pos.icrs.dec.value,'{:4.1f}'.format(v),fontsize='x-large',color='k')

def mask_data(mask,data,mode='inside'):
    d_ma = np.copy(data)
    if mode == 'inside':
        ind = np.where(mask != 0)
    elif mode == 'outside':
        ind = np.where(mask == 0)
    else:
        print('mask mode not correct')
        return d_ma

    d_ma[ind] = np.nan
    return d_ma

def main(args):
    if not os.path.isfile(args.file_fits):
        print("FITS File not found")
        return 0
    if not os.path.isfile(args.file_dend):
        print("Dend File not found")
        return 0
    #------------------------
    #    Load dendrogram
    #------------------------
    print('Load Dendrogram')
    dendro = Dendrogram.load_from(args.file_dend)
    print('')
    #------------------------
    #    Load vlsr image
    #------------------------
    f_base = os.path.splitext(args.file_fits)[0]
    f_mask = f_base+'_mask.fits'

    with fits.open(args.file_fits,mode='readonly') as hdul:
        hdr = hdul[0].header
        data = np.copy(np.squeeze(hdul[0].data))
        count_branch=0
        for i, struc in enumerate(dendro.trunk):
            if struc.is_branch: # branch
                mask = struc.get_mask().mean(axis=0)
                d_ma = mask_data(mask,data,mode='outside')
                count_branch=count_branch+1
                subtree = struc.descendants
                for j, sub_struc in enumerate(subtree):
                    if sub_struc.is_leaf:
                        mask = sub_struc.get_mask().mean(axis=0)
                        d_ma = mask_data(mask,d_ma,mode='inside')
                break
        fits.writeto(f_mask,d_ma,hdr,overwrite=True)


    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('file_fits', type=str, help='the input data file')
    parser.add_argument('file_dend', type=str, default='my_dendrogram', help='the dendrogram file')


    args = parser.parse_args()
    start_time = time.time()
    main(args)
    print("--- {:.3f} seconds ---".format(time.time() - start_time))

