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

def plot_spec(fig,x,y,ftsize='xx-large',vlim=None,title=None,vline=None,method=None,legend=False):
    ax=fig.add_subplot(1,1,1)

    if vline is not None:
        ax.axvline(vline,lw=1,color='red')
        ax.text(vline+8,0.9*max(y),r'H$n\alpha$',fontsize=ftsize)

        ax.axvline(vline-122,lw=1,color='green')
        ax.text(vline-122+5,0.9*max(y),r'He$n\alpha$',fontsize=ftsize)

        ax.axvline(vline-150,lw=1,color='blue')
        ax.text(vline-150-35,0.9*max(y),r'C$n\alpha$',fontsize=ftsize)

    ax.plot(x,y, '-',lw=1.5, color='k', label=method)
    ax.set_xlabel('$\Delta$ V$_{LSR}$ (km$\,$s$^{-1}$)',fontsize=ftsize)
    ax.set_ylabel('Relative Intensity',fontsize=ftsize)
    ax.tick_params(labelsize=ftsize)

    if legend:
        ax.legend(fontsize=ftsize)
    if vlim is not None:
        ax.set_xlim(vlim[0],vlim[1])
    if title is not None:
        ax.set_title(title,fontsize=ftsize)
    return ax

def struc_info(struc,wcs,idx_arr=None,label=None,stype='leaf',method='sum'):
    if label is None:
        if idx_arr is None:
            label=0
        else:
            label = np.argwhere(idx_arr == struc.idx)[0][0]+1
    file_out = 'spec_{}_{:d}_{}.png'.format(stype,label,method)

    peak = struc.get_peak()[0]
    x_p = peak[2]
    y_p = peak[1]

    coor = SkyCoord.from_pixel(x_p,y_p,wcs)
    gc = coor.transform_to('galactic')
    gal_str = 'G{:5.2f}{:+5.2f}'.format(gc.l.value,gc.b.value)
    title = 'Index:{:d}; {}'.format(label,gal_str)
    return title, file_out

def struc_spec(struc,data,velo,chan0,nchan,nx,ny,wbounds=[8.0,29.0],method='sum'):
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
    spm = md.mean(axis=(1,2))
    spp = data[:,y_p,x_p]

    if method == 'sum':
        spec = sps
    elif method == 'peak':
        spec = spp
    elif method == 'mean':
        spec = spm

#    sp_peak = smooth(spp,window_len=5) # spectrum - peak - smooth
#    sp_sum = smooth(sps,window_len=5) # spectrum - sum - smooth
#    sp_fit,peak,vlsr,fwhm,e1,e2,e3 = fit(velo,spec,init=[spec[v_p],velo[v_p],15],\
#            vbounds=[velo[v_p]-1,velo[v_p]+1],wbounds=wbounds)

    return spec[v_p-400:v_p+200]

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
    d = Dendrogram.load_from(args.file_d)
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
    fig = plt.Figure(figsize=(6, 4.5))

    spec_arr = []
    rms_arr =[]
    for i, struc in enumerate(d.leaves):
        title,file_out = struc_info(struc,wcs,idx_arr=leaves_idx_arr,stype='leaf',method=args.method)
        print(title,struc.idx)
        if args.type == 'hydrogen':
            if struc.idx == 30:
                wbounds = [10,30]
            else:
                wbounds = [5,30]
        elif args.type == 'carbon':
                wbounds = [3,8]
            
        spec = struc_spec(struc,data,velo,args.chan_0,nchan,nx,ny,wbounds=wbounds,method=args.method)
        rms = np.nanstd(spec[-100:-1])
        spec_arr.append(spec)
        rms_arr.append(rms)

    
    for i, spec in enumerate(spec_arr):
        #weight = 1.
        #weight = 1./rms_arr[i]
        weight = 1./(rms_arr[i]**2)
        spec_i = spec*weight
        if i == 0:
            spec_ave = spec_i
        else:
            spec_ave = spec_ave + spec_i

    velo = np.arange(0,len(spec_ave))*0.5 - 200
    print(spec_ave.shape)
    print('rms',np.std(spec_ave[0:300]))
    print('max',np.max(spec_ave))
    plot_spec(fig,velo,spec_ave,vline=0,ftsize=12,method=args.method)
    fig.savefig('clump_spec_aver_w1.png',dpi=300,format='png',bbox_inches='tight')
    plt.close()
    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('fits_file', type=str, help='the input data file')
    parser.add_argument('--file_d', type=str, default='my_dendrogram', help='the dendrogram file')
    parser.add_argument('--method', type=str, default='sum', help='the way of extracting spectra')
    parser.add_argument('--type', type=str, default='hydrogen', help='the line type')
    parser.add_argument('--chan_0', type=int, default=0,  help='channel index start')
    parser.add_argument('--chan_1', type=int, default=-1, help='channel index end')
    args = parser.parse_args()
    start_time = time.time()
    main(args)
    print("--- {:.3f} seconds ---".format(time.time() - start_time))
