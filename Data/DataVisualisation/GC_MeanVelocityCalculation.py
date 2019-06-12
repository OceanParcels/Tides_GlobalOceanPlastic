# -*- coding: utf-8 -*-
"""
Computing the temporal mean zonal and meridional GlobCurrent velocities

Author: Miriam Sterl
"""

from netCDF4 import Dataset
import glob

filenames = glob.glob('/science/projects/oceanparcels/input_data/DataPlasticTides/GlobCurrent/20*.nc')
sumU = 0
sumV = 0
for f in filenames:
    dataset = Dataset(f)
    sumU += dataset.variables['eastward_eulerian_current_velocity'][0,:,:]
    sumV += dataset.variables['northward_eulerian_current_velocity'][0,:,:]

meanU = sumU/len(filenames)
meanV = sumV/len(filenames)

saveFile_U = '/science/projects/oceanparcels/output_data/data_Miriam/MeanU'
saveFile_V = '/science/projects/oceanparcels/output_data/data_Miriam/MeanV'
meanU.dump(saveFile_U)
meanV.dump(saveFile_V)