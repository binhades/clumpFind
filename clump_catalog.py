#!/usr/bin/python3

import argparse, os, time
import numpy as np
from toolkit import smooth, vlsr_distance
from pyrrl.spec import fit
from astropy.io import fits
from astropy.wcs import WCS
from astropy.visualization import simple_norm
from astropy.coordinates import SkyCoord
from astrodendro import Dendrogram, ppv_catalog

def main(args):
    # -------------------------------------------
    #    Load DataCube
    # -------------------------------------------
    print('Load DataCube')
    hdu = fits.open(args.fits_file)[0]
    hdr = hdu.header
    wcs = WCS(header=hdr).celestial
    data = hdu.data
    nchan = data.shape[0]
    velo = (np.arange(nchan) - hdr['CRPIX3'] + 1) * hdr['CDELT3'] + hdr['CRVAL3']
    ny = data.shape[1]
    nx = data.shape[2]

    # unit convert: Jy/beam -> mJy/pixel
    beam=4.7 # arcmin
    pix = 1.0 # arcmin
    pix_over_beam = pix**2/((beam/2)**2*np.pi)
    print(pix_over_beam)
    data = data * 1000 * pix_over_beam # x Jy/beam = (x * pix/beam) Jy/pix 

    # -------------------------------------------
    #    Load dendrogram
    # -------------------------------------------
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
    if args.file_csv is not None:
        import csv
        fcsv = open(args.file_csv,'w')
        fieldnames = ['Index', 'GName', 'Coordinate', 'Peak', 'Peak_err',\
                      'Vlsr', 'vlsr_err', 'fwhm', 'fwhm_err', 'Flux_int',\
                      'Area', 'D_far', 'D_near']
        writer = csv.DictWriter(fcsv,fieldnames=fieldnames,quoting=csv.QUOTE_NONE)
        writer.writeheader()
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

        size = np.where(mask2d>0)[0].shape[0]

        md = np.ma.masked_array(data,mask=mask3d)
        spm = md.sum(axis=(1,2))
        spp = data[:,y_p,x_p]
        spmsm = smooth(spm,window_len=5)
        sppsm = smooth(spp,window_len=5)

        yfit,peak,vlsr,fwhm,err1,err2,err3 = fit(velo,spmsm,paras=[spmsm[v_p],velo[v_p],15.])
        d_far,d_near = vlsr_distance(gc.l.value,vlsr)
        print("{index:02d} {Gname} {Coor} {peak:5.2f}$\pm${perr:4.2f} {vlsr:4.1f}$\pm${verr:3.1f} {fwhm:4.1f}$\pm${werr:3.1f} {integ:8.2f} {area:3d} {d1:4.1f} {d2:4.1f}".format(index=leaf_label,Gname=gal_str,Coor=equ_str,peak=peak,perr=err1,vlsr=vlsr,verr=err2,fwhm=fwhm,werr=err3,integ=peak*fwhm,area=size,d1=d_far,d2=d_near))
        if args.file_csv is not None:
            row = {'Index':leaf_label, 'GName':gal_str, 'Coordinate':equ_str,\
                    'Peak':peak, 'Peak_err':err1,'Vlsr':vlsr, 'vlsr_err':err2,\
                    'fwhm':fwhm, 'fwhm_err':err3, 'Flux_int':peak*fwhm,\
                    'Area':size, 'D_far':d_far, 'D_near':d_near}

            writer.writerow(row)
    if args.file_csv is not None:
        fcsv.close()

    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('fits_file', type=str, help='the input data file')
    parser.add_argument('--file_d', type=str, default='my_dendrogram', help='the dendrogram file')
    parser.add_argument('--file_csv', type=str, help='the csv file for output')
    parser.add_argument('--chan_0', type=int, default=0,  help='channel index start')
    parser.add_argument('--chan_1', type=int, default=-1, help='channel index end')
    args = parser.parse_args()
    start_time = time.time()
    main(args)
    print("--- {:.3f} seconds ---".format(time.time() - start_time))
