#!/usr/bin/python3

import argparse, os, time
import numpy as np
from toolkit import smooth, vlsr_distance
from pyrrl.spec import fit
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astrodendro import Dendrogram

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

    size = np.where(mask2d>0)[0].shape[0]
    md = np.ma.masked_array(data,mask=mask3d)
    sps = md.sum(axis=(1,2))
    spm = md.mean(axis=(1,2))
    spp = data[:,y_p,x_p]

    if method == 'peak':
        spec = spp
    elif method == 'mean':
        spec = spm
    elif method == 'sum':
        spec = sps
    else:
        print('method error')
        return 0

    sp_fit,peak,vlsr,fwhm,e1,e2,e3 = fit(velo,spec,init=[spec[v_p],velo[v_p],15],\
            vbounds=[velo[v_p]-5,velo[v_p]+5],wbounds=wbounds)

    return peak, vlsr, fwhm, e1,e2,e3, size

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

    # ------------------------
    # unit convert: Jy/beam -> mJy/pix
    # ------------------------
    if args.unit == 'Jy':
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
    # init CSV 
    # ------------------------
    if args.file_csv is not None:
        import csv
        fcsv = open(args.file_csv,'w')
        fieldnames = ['Index', 'GName', 'GLon', 'GLat', 'Coordinate', \
                      'Peak', 'Peak_err', 'VLSR', 'VLSR_err', 'FWHM', 'FWHM_err',\
                      'Flux_int', 'Flux_int_err','Area', 'D_far', 'D_near']
        writer = csv.DictWriter(fcsv,fieldnames=fieldnames,quoting=csv.QUOTE_NONE)
        writer.writeheader()
    # ------------------------
    for i in range(len(d.leaves)):
        ind = peak_ind[i]
        struc = d.leaves[ind]
        leaf_label = np.argwhere(leaves_idx_arr == struc.idx)[0][0]+1
        peak = struc.get_peak()[0]
        v_p = args.chan_0+peak[0]
        x_p = peak[2]
        y_p = peak[1]

        coor = SkyCoord.from_pixel(x_p,y_p,wcs)
        equ_str = coor.to_string(style='hmsdms',precision=0)
        gc = coor.transform_to('galactic')
        gal_str = 'G{:5.2f}{:+5.2f}'.format(gc.l.value,gc.b.value)

        if struc.idx == 30:
            wbounds = [10,30]
        else:
            wbounds = [5,30]

        peak,vlsr,fwhm,err1,err2,err3,size = struc_spec(struc,data,velo,args.chan_0,nchan,nx,ny,wbounds=wbounds,method=args.method)
        flux_int = peak*fwhm
        err4 = np.sqrt(peak**2*err3**2+fwhm**2*err1**2)

        d_far,d_near = vlsr_distance(gc.l.value,vlsr)
        print("{index:02d} {Gname} {Coor} {peak:5.2f}$\pm${perr:4.2f} {vlsr:4.1f}$\pm${verr:3.1f} {fwhm:4.1f}$\pm${werr:3.1f} {integ:8.1f}$\pm${int_err:4.1f} {area:3d} {d1:4.1f} {d2:4.1f}".format(index=leaf_label,Gname=gal_str,Coor=equ_str,peak=peak,perr=err1,vlsr=vlsr,verr=err2,fwhm=fwhm,werr=err3,integ=flux_int,int_err=err4,area=size,d1=d_far,d2=d_near))
        if args.file_csv is not None:
            row = {'Index':leaf_label, 'GName':gal_str,\
                    'GLon':gc.l.value, 'GLat':gc.b.value,'Coordinate':equ_str,\
                    'Peak':peak, 'Peak_err':err1,'VLSR':vlsr, 'VLSR_err':err2,\
                    'FWHM':fwhm, 'FWHM_err':err3, 'Flux_int':flux_int,'Flux_int_err':err4,\
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
    parser.add_argument('--method',type=str, default='sum', help='method to extracting spectra: sum, mean, peak')
    parser.add_argument('--unit',type=str, default='Jy', help='default unit')

    args = parser.parse_args()
    start_time = time.time()
    main(args)
    print("--- {:.3f} seconds ---".format(time.time() - start_time))


