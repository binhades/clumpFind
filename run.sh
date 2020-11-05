#!/bin/bash
#./g034_scimes.py G034.39+00.22_Halpha_Jy_cube_aver_fix_cov_H184a_smooth.fits --sigma 0.001 --ppb 3
#./calc_dendrogram.py G034.39+00.22_Halpha_Jy_cube_aver_fix_cov_H184a_smooth.fits --sigma 0.0005 --snr 5 --ppb 2. --delta 0.8 --chan_0 800 --chan_1 1000 --file_out my_dendrogram
./leaves.py G034.39+00.22_Halpha_Jy_cube_aver_fix_cov_H184a.fits --file_d my_dendrogram --chan_0 800 --chan_1 1000 
