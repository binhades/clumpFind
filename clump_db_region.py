#!/usr/bin/python3

# Aim: to load the dendro file, make polygon from the Clump mask, 
#      and store into sqlite database

import argparse, os, time, sqlite3
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astrodendro import Dendrogram
import matplotlib.pyplot as plt

# =============================================================================
# Functions to process clumps

def get_leaf_mask(struc, leaf_index_list,wcs,group):

    leaf_ind = np.argwhere(leaf_index_list == struc.idx)[0][0]+1
    peak = struc.get_peak()[0]
    x_p = peak[2]
    y_p = peak[1]

    coor = SkyCoord.from_pixel(x_p,y_p,wcs)
    equ_j2000 = coor.to_string(style='hmsdms',precision=0)
    ra = coor.ra.value
    dec = coor.dec.value
    gc = coor.transform_to('galactic')
    gname = 'G{:5.2f}{:+5.2f}'.format(gc.l.value,gc.b.value)

    mask = struc.get_mask().mean(axis=0)
    mask[np.nonzero(mask)] = 1
    poly_x,poly_y = get_polygon_from_mask(mask,wcs)

    data = {'GName':gname,'Id':leaf_ind, 'Gid':group, \
            'RA':ra, 'Dec':dec,'Equ_J2000':equ_j2000, \
            'PolyX':poly_x, 'PolyY':poly_y}

    return data

def get_branch_mask(struc, index,wcs):

    mask = struc.get_mask().mean(axis=0)
    mask[np.nonzero(mask)] = 1
    poly_x,poly_y = get_polygon_from_mask(mask,wcs)
    data = {'Gid':index,'PolyX':poly_x, 'PolyY':poly_y}

    return data

def get_polygon_from_mask(mask,wcs):
    pos = None
    try:
        plt.subplot(projection=wcs)
        contour = plt.contour(mask,levels=[1.0])
        plt.close()
    except UserWarning:
        0
    for seg in contour.collections[0].get_segments():
        if pos is None:
            pos = seg
        else:
            pos = np.concatenate((pos,seg))
    coor = SkyCoord.from_pixel(pos[:,0],pos[:,1],wcs)
    poly_x = coor.ra.value
    poly_y = coor.dec.value

    return poly_x,poly_y

def get_leaf_index_list(dendro):

    list_idx = [] # raw index
    list_idv = [] # sorted index
    list_peak= [] # raw peaks
    for i, struc in enumerate(dendro.leaves):
        peak = struc.get_peak()[1]
        list_peak.append(peak)
        list_idx.append(struc.idx)
    peak_ind = np.argsort(np.array(list_peak))[::-1]
    peak_arr = np.array(list_peak)[peak_ind]
    leaf_index_list = np.array(list_idx)[peak_ind]

    return leaf_index_list, peak_arr

# ------------------------------------------------------------------
# Functions for database

def db_init(file_sql):
    try:
        conn = sqlite3.connect(args.file_sql)
        conn.row_factory = sqlite3.Row
    except sqlite3.Error as err:
        print('Init Sqlite3 connection Error:', err)
        return None
    table1 = 'Regions'
    
    table1_hdr = ''' GName TEXT PRIMARY KEY,
                     Equ_J2000 TEXT,
                     RA REAL,
                     Dec REAL,
                     Id INTEGER,
                     Gid INTEGER,
                     PolyX BLOB,
                     PolyY BLOB
                 '''

    table2 = 'Groups'
    table2_hdr = ''' Gid INT PRIMARY KEY,
                     PolyX BLOB,
                     PolyY BLOB
                 '''
    table1_exec ='CREATE TABLE if not exists {} ({})'.format(table1,table1_hdr)
    table2_exec ='CREATE TABLE if not exists {} ({})'.format(table2,table2_hdr)

    try:
        cur = conn.cursor()
        cur.execute(table1_exec)
        conn.commit()
    except sqlite3.Error as err:
        print('Init Sqlite Creat Regions Table Error:', err)
        return 0

    try:
        cur = conn.cursor()
        cur.execute(table2_exec)
        conn.commit()
    except sqlite3.Error as err:
        print('Init Sqlite Creat Groups Table Error:', err)
        return 0


    return conn

def db_add_leaf(conn,leaf):
    try:
        cur = conn.cursor()
        poly_x = leaf['PolyX'].astype(np.float32,order='C',casting='same_kind')
        poly_y = leaf['PolyY'].astype(np.float32,order='C',casting='same_kind')
        polyXStr = poly_x.tobytes(order='C')
        polyYStr = poly_y.tobytes(order='C')

        data = (leaf['GName'],leaf['Equ_J2000'],leaf['RA'],leaf['Dec'], \
                int(leaf['Id']),int(leaf['Gid']),polyXStr,polyYStr)
        cur.execute("INSERT INTO Regions (GName,Equ_J2000,RA,Dec,Id,Gid,PolyX,PolyY) \
                     VALUES (?,?,?,?,?,?,?,?)",data)
        conn.commit()
    except sqlite3.Error as err:
        print(err)
        return None
    return conn

def db_add_branch(conn,branch):
    try:
        cur = conn.cursor()
        poly_x = branch['PolyX'].astype(np.float32,order='C',casting='same_kind')
        poly_y = branch['PolyY'].astype(np.float32,order='C',casting='same_kind')
        polyXStr = poly_x.tobytes(order='C')
        polyYStr = poly_y.tobytes(order='C')

        data = (int(branch['Gid']),polyXStr,polyYStr)
        cur.execute("INSERT INTO Groups (Gid,PolyX,PolyY) \
                     VALUES (?,?,?)",data)
        conn.commit()
    except sqlite3.Error as err:
        print(err)
        return None

    return conn

# =============================================================================

def main(args):
    if not os.path.isfile(args.file_fits):
        print('File not found: ', args.file_fits)
        return 0
    if not os.path.isfile(args.file_dend):
        print('File not found: ', args.file_dend)
        return 0
    if os.path.isfile(args.file_sql):
        print('Database file exist, use another name', args.file_sql)
        return 0

    # -------------------------------------------
    #    Load DataCube
    # -------------------------------------------
    print('Load FITS header from image')
    hdu = fits.open(args.file_fits)[0]
    hdr = hdu.header
    wcs = WCS(header=hdr).celestial

    #------------------------
    #    Load dendrogram
    #------------------------
    print('Load Dendrogram')
    dendro = Dendrogram.load_from(args.file_dend)
    leaf_index_list, peak_arr = get_leaf_index_list(dendro)
    branch_index_list = []

    # ------------------------
    # init sqlite3 database 
    # ------------------------
    db_conn = db_init(args.file_sql)

    print(type(db_conn))
    # ------------------------
    # Major Run 
    # ------------------------
    for i, struc in enumerate(dendro.trunk):

        if struc.is_leaf: # individual leaf, not belongs to branches
            group=0
            data = get_leaf_mask(struc,leaf_index_list,wcs,group)
            db_conn = db_add_leaf(db_conn, data)

        elif struc.is_branch:
            branch_index_list=np.append(branch_index_list,struc.idx)
            group = np.argwhere(branch_index_list == struc.idx)[0][0]+1
            data = get_branch_mask(struc,group,wcs)
            db_conn = db_add_branch(db_conn, data)

            for j, sub_struc in enumerate(struc.descendants):
                if sub_struc.is_leaf: # individual leaf, not belongs to branches
                   data = get_leaf_mask(sub_struc,leaf_index_list,wcs,group)
                   db_conn = db_add_leaf(db_conn, data)
               
    return 0
#----------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('file_fits', type=str, help='the FITS image for WCS')
    parser.add_argument('file_dend', type=str, help='the dendrogram file')
    parser.add_argument('file_sql',  type=str, help='the output sql database')

    args = parser.parse_args()
    start_time = time.time()
    main(args)
    print("--- {:.3f} seconds ---".format(time.time() - start_time))
