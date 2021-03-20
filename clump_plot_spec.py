#!/usr/bin/python3

import argparse, os, time
import numpy as np
from toolkit import smooth
from pyrrl.spec import fit
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astrodendro import Dendrogram
from matplotlib import pyplot as plt

def plot_spec(fig,x,ypeak,ysum,yfit,ftsize='xx-large',title=None,vline=None):
    ax=fig.add_subplot(1,1,1)

    if vline is not None:
        ax.axvline(vline,lw=1,color='blue')
        ax.axvline(vline-122,lw=1,color='green')
        ax.text(vline+8,0.9*max(yfit),r'Hn$\alpha$',fontsize=ftsize)
        ax.text(vline-122+5,0.9*max(yfit),r'Hen$\alpha$',fontsize=ftsize)

#    ax.plot(x,ypeak,'-',lw=1.5, color='b', label='Peak')
    ax.plot(x,ysum, '-',lw=1.5, color='k', label='Sum')
    ax.plot(x,yfit,'--',lw=2.5, color='r', label='Fit')
    #ax.legend(fontsize=ftsize)
    ax.set_xlim(-100,120)
    ax.set_xlabel('V$_{LSR}$ (km$\,$s$^{-1}$)',fontsize=ftsize)
    ax.set_ylabel('Flux (mJy)',fontsize=ftsize)
    ax.tick_params(labelsize=ftsize)
    if title is not None:
        ax.set_title(title,fontsize=ftsize)
    return ax

def struc_info(struc,wcs,idx_arr=None,label=None,stype='leaf'):
    if label is None:
        if idx_arr is None:
            label=0
        else:
            label = np.argwhere(idx_arr == struc.idx)[0][0]+1
    file_out = 'spec_{}_{:d}.png'.format(stype,label)

    peak = struc.get_peak()[0]
    x_p = peak[2]
    y_p = peak[1]

    coor = SkyCoord.from_pixel(x_p,y_p,wcs)
    gc = coor.transform_to('galactic')
    gal_str = 'G{:5.2f}{:+5.2f}'.format(gc.l.value,gc.b.value)
    title = 'Index:{:d}; {}'.format(label,gal_str)
    return title, file_out

def struc_spec(struc,data,velo,chan0,nchan,nx,ny,wbounds=[8.0,29.0]):
    peak = struc.get_peak()[0]
    x_p = peak[2]
    y_p = peak[1]
    v_p = chan0+peak[0]


    mask2d = struc.get_mask().mean(axis=0)
    mask = np.zeros((ny,nx))
    mask[np.where(mask2d==0)] = True
    mask[np.where(mask2d!=0)] = False
    mask3d = np.repeat(mask[np.newaxis,:,:],nchan,axis=0)

    md = np.ma.masked_array(data,mask=mask3d)
    sps = md.sum(axis=(1,2))
    spp = data[:,y_p,x_p]


    sp_peak = smooth(spp,window_len=5) # spectrum - peak - smooth
    sp_sum = smooth(sps,window_len=5) # spectrum - sum - smooth
    sp_fit,peak,vlsr,fwhm,e1,e2,e3 = fit(velo,sps,init=[sp_sum[v_p],velo[v_p],15],\
            vbounds=[velo[v_p]-5,velo[v_p]+5],wbounds=wbounds)
    print(peak,vlsr,fwhm)

    return sp_peak, sp_sum, sp_fit, vlsr

def main(args):

    #------------------------
    #    Load DataCube
    #------------------------
    print('Load DataCube')
    hdu = fits.open(args.fits_file)[0]
    hdr = hdu.header
    wcs = WCS(header=hdr).celestial
    data = hdu.data
    ny = data.shape[1]
    nx = data.shape[2]
    nchan = data.shape[0]
    velo = (np.arange(nchan) - hdr['CRPIX3'] + 1) * hdr['CDELT3'] + hdr['CRVAL3']

    # ------------------------
    # unit convert: Jy/beam -> mJy/pix
    # ------------------------
    beam = 4.7 # arcmin
    pix = 1.0 # arcmin
    pix_over_beam = pix**2/((beam/2)**2*np.pi)
    data = data * 1000 * pix_over_beam # x Jy/beam = (x * pix/beam) Jy/pix

    #------------------------
    #    Load dendrogram
    #------------------------
    print('Load Dendrogram')
    d = Dendrogram.load_from(args.file_d+'.hdf5')
    print('')
    # ------------------------
    # leaf label
    # ------------------------
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
    fig = plt.Figure(figsize=(8, 4))

    for i, struc in enumerate(d.leaves):
        title,file_out = struc_info(struc,wcs,idx_arr=leaves_idx_arr,stype='leaf')
        print(title,struc.idx)
        if struc.idx == 30:
            wbounds = [10,30]
        else:
            wbounds = [5,30]
            
        sp_peak,sp_sum,sp_fit,vlsr = struc_spec(struc,data,velo,args.chan_0,nchan,nx,ny,wbounds=wbounds)
        plot_spec(fig,velo,sp_peak,sp_sum,sp_fit,vline=vlsr,title=title,ftsize=25)
        fig.savefig(file_out,dpi=300,format='png',bbox_inches='tight')
        fig.clear(True)

    j = 0
    for i, struc in enumerate(d.trunk):
        if struc.is_branch: # branch
            j=j+1
            title,file_out = struc_info(struc,wcs,stype='branch',label=j)
            sp_peak,sp_sum,sp_fit,vlsr = struc_spec(struc,data,velo,args.chan_0,nchan,nx,ny)
            plot_spec(fig,velo,sp_peak,sp_sum,sp_fit,vline=vlsr,title=title)
            fig.savefig(file_out,dpi=300,format='png',bbox_inches='tight')
            fig.clear(True)
    plt.close()
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