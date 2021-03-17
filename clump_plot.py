#!/usr/bin/python3

import argparse, os, time
import numpy as np
from toolkit import smooth
from toolkit import colormap
from astropy.io import fits
from astropy.wcs import WCS
from astropy.visualization import simple_norm
from astropy.coordinates import SkyCoord
from astrodendro import Dendrogram, ppv_catalog
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.colors import LogNorm

def plot_spec(fig,x,ym,yp,ftsize='x-large',title=None,vline=None):
    ax = fig.add_subplot(1,2,1)

    if vline is not None:
        ax.axvline(x[vline],lw=1.5,color='red')

    ax.plot(x,yp,lw=0.5, label='peak')
    ax.plot(x,ym,'--',lw=1, label='average')
    ax.legend()
    ax.set_xlim(-200,250)
    #ax.set_ylim(-2.5,10.5)
    ax.set_xlabel('V$_{LSR}$ (km$\,$s$^{-1}$)',fontsize=ftsize)
    ax.set_ylabel('Flux (mJy$\,$beam$^{-1}$)',fontsize=ftsize)
    if title is not None:
        ax.set_title(title,fontsize=ftsize)
    return ax

def plot_imag(fig,imag,mask,wcs,ftsize='x-large',coor=None,title=None,cmap='hot'):
    ax = fig.add_subplot(1,2,2,projection=wcs)

    # ------------------------
    # Display the image
    im = ax.imshow(imag,origin='lower',interpolation='nearest',cmap=colormap(cmap),\
            aspect='equal',vmin=0.)#,norm=LogNorm()) #,vmin=0.0005,vmax=0.005
    ax.contour(mask,linewidths=2,levels=[0.001],alpha=0.8,colors='grey')
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
        ax.plot(coor[0],coor[1],marker='*')
    cbar = ax.figure.colorbar(im)
    cbar.ax.set_ylabel('mJy$\,$beam$^{-1}$ km$\,$s$^{-1}$',fontsize=ftsize)

    return ax

def plot_struc(fig, file_out,struc,velo,wcs,hdr,data,nx,ny,nchan,cmap):
    peak = struc.get_peak()[0]
    v_p = args.chan_0+peak[0]
    x_p = peak[2]
    y_p = peak[1]

    coor = SkyCoord.from_pixel(x_p,y_p,wcs)
    gc = coor.transform_to('galactic')
    equ_str = coor.to_string(style='hmsdms',precision=0)
    gal_str = 'G{:5.2f}{:+5.2f}'.format(gc.l.value,gc.b.value)
    title0 = equ_str + ' @ '+str(velo[v_p]) + ' km/s'
    title1 = gal_str + ' @ '+str(velo[v_p]) + ' km/s'

    imag  = data[v_p-2:v_p+2,:,:].sum(axis=0) * hdr['CDELT3'] * 1000

    mask2d = struc.get_mask().mean(axis=0)
    mask = np.zeros((ny,nx))
    mask[np.where(mask2d==0)] = True
    mask[np.where(mask2d!=0)] = False
    mask3d = np.repeat(mask[np.newaxis,:,:],nchan,axis=0)

    md = np.ma.masked_array(data,mask=mask3d)
    spm = md.mean(axis=(1,2))
    spp = data[:,y_p,x_p]
    spmsm = smooth(spm,window_len=5) * 1000.
    sppsm = smooth(spp,window_len=5) * 1000.

    plot_spec(fig,velo,spmsm,sppsm,vline=v_p,title=title0)
    plot_imag(fig,imag,mask2d,wcs,coor=(x_p,y_p),title=title1,cmap=cmap)
    fig.savefig(file_out,dpi=300,format='png',bbox_inches='tight')
    plt.clf()

    return fig

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
    ra0 = ax0.coords['ra']
    de0 = ax0.coords['dec']
    ra0.set_axislabel('R.A. (J2000)',minpad=0.5,size="xx-large")
    de0.set_axislabel('Dec. (J2000)',minpad=0.5,size="xx-large")
    ra0.set_separator(('$\mathrm{^h}$','$\mathrm{^m}$'))
    ra0.set_ticklabel(size="xx-large")
    de0.set_ticklabel(size="xx-large")

    fig1 = plt.figure(figsize=(12, 4))

    colors = ['blue','red','green']
    c=1
    for i, struc in enumerate(d.trunk):
        if struc.is_leaf:
            file_out = 'leaf_{:d}_{}.png'.format(struc.idx,args.cmap)
            color = colors[0]
            subtree = []
            line = 'solid'
        elif struc.is_branch: # branch
            file_out = 'branch_{:d}_{}.png'.format(struc.idx,args.cmap)
            color=colors[c]
            c=c+1
            subtree = struc.descendants
            line = 'dashed'

        plot_struc(fig1,file_out,struc,velo,wcs,hdr,data,nx,ny,nchan,args.cmap)
        mask = struc.get_mask().mean(axis=0)
        ax0.contour(mask,linewidths=1.5,levels=[0.001],alpha=0.8,colors=[color],linestyles=line)

        for j, sub_struc in enumerate(subtree):
            if sub_struc.is_leaf:
                file_out = 'leaf_{:d}_{}.png'.format(sub_struc.idx,args.cmap)
                plot_struc(fig1,file_out,sub_struc,velo,wcs,hdr,data,nx,ny,nchan,args.cmap)
                mask = sub_struc.get_mask().mean(axis=0)
                ax0.contour(mask,linewidths=1.5,levels=[0.001],alpha=0.8,colors=[color])

    fig0.savefig('m0-clumps_{}.png'.format(args.cmap),dpi=300,format='png',bbox_inches='tight')

    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('fits_file', type=str, help='the input data file')
    parser.add_argument('--file_d', type=str, default='my_dendrogram', help='the dendrogram file')
    parser.add_argument('--chan_0', type=int, default=0,  help='channel index start')
    parser.add_argument('--chan_1', type=int, default=-1, help='channel index end')
    parser.add_argument('--cmap', type=str, default='hot', help='the colormap, batlow or lajolla')
    args = parser.parse_args()
    start_time = time.time()
    main(args)
    print("--- {:.3f} seconds ---".format(time.time() - start_time))
