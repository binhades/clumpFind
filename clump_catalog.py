#!/usr/bin/python3

import argparse, os, time
import numpy as np
from toolkit import smooth
from pyrrl.spec import fit
from astropy.io import fits
from astropy.wcs import WCS
from astropy.visualization import simple_norm
from astropy.coordinates import SkyCoord
from astrodendro import Dendrogram, ppv_catalog

def main(args):

    #%&%&%&%&%&%&%&%&%&%&%&%
    #    Load DataCube
    #%&%&%&%&%&%&%&%&%&%&%&%
    print('Load DataCube')
    hdu = fits.open(args.fits_file)[0]
    hdr = hdu.header
    wcs = WCS(header=hdr).celestial
    data = hdu.data * 0.05764 * 1000 # unit convert: Jy/beam -> mJy/pixel
    nchan = data.shape[0]
    velo = (np.arange(nchan) - hdr['CRPIX3'] + 1) * hdr['CDELT3'] + hdr['CRVAL3']
    ny = data.shape[1]
    nx = data.shape[2]

    #%&%&%&%&%&%&%&%&%&%&%&%
    #    Load dendrogram
    #%&%&%&%&%&%&%&%&%&%&%&%
    print('Load Dendrogram')
    d = Dendrogram.load_from(args.file_d+'.hdf5')

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
    print('')
    # ------------------------
    for i in range(len(d.leaves)):
        ind = peak_ind[i]
        leaf = d.leaves[ind]
        leaf_label = np.argwhere(leaves_idx_arr == leaf.idx)[0][0]+1
        #leaf_label = leaf.idx

        file_out = 'leaf'+str(leaf.idx)+'.png'
        peak = leaf.get_peak()[0]
        v_p = args.chan_0+peak[0]
        x_p = peak[2]
        y_p = peak[1]

        coor = SkyCoord.from_pixel(x_p,y_p,wcs)
        gc = coor.transform_to('galactic')
        equ_str = coor.to_string(style='hmsdms',precision=0)
        gal_str = 'G{:5.2f}{:+5.2f}'.format(gc.l.value,gc.b.value)

        mask2d = leaf.get_mask().mean(axis=0)
        mask = np.zeros((ny,nx))
        mask[np.where(mask2d==0)] = True
        mask[np.where(mask2d!=0)] = False
        mask3d = np.repeat(mask[np.newaxis,:,:],nchan,axis=0)

        md = np.ma.masked_array(data,mask=mask3d)
        spm = md.sum(axis=(1,2))
        spp = data[:,y_p,x_p]
        spmsm = smooth(spm,window_len=5)
        sppsm = smooth(spp,window_len=5)

        yfit,peak,vlsr,fwhm,err1,err2,err3 = fit(velo,spmsm,paras=[spmsm[v_p],velo[v_p],15.])
        print("{index:02d} {Gname} {Coor} {peak:5.2f}$\pm${perr:4.2f} \
                {vlsr:4.1f}$\pm${verr:3.1f} {fwhm:4.1f}$\pm${werr:3.1f} \
                {integ:8.3f} {area:f}".format(leaf_label,gal_str,equ_str,peak,err1,\
                vlsr,err2,fwhm,err3,peak*fwhm,leaf.get_npix()))


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
