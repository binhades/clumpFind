#!/bin/bash
#./g034_scimes.py G034.39+00.22_Halpha_Jy_cube_aver_fix_cov_H184a_smooth.fits --sigma 0.001 --ppb 3
./leaves.py G034.39+00.22_Halpha_Jy_cube_aver_fix_cov_H184a_smooth.fits --sigma 0.0005 --ppb 4 --delta 2  --iter 100
