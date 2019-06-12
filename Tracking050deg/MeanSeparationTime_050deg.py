# -*- coding: utf-8 -*-
"""
Computing the mean separation (of all particle pairs) as a function of time

@author: Miriam Sterl
"""

from netCDF4 import Dataset
import numpy as np
import math

# Only GlobCurrent
File1 = '/science/projects/oceanparcels/output_data/data_Miriam/Results050deg/Results_TrackingGC_050deg.nc'
dataset1 = Dataset(File1)
lat1 = dataset1.variables['lat'][:]
lon1 = dataset1.variables['lon'][:]
lon1[lon1>180]-=360
lon1[lon1<-180]+=360

# GlobCurrent + FES
File2 = '/science/projects/oceanparcels/output_data/data_Miriam/Results050deg/Results_TrackingGCFES_050deg.nc'
dataset2 = Dataset(File2)
lat2 = dataset2.variables['lat'][:]
lon2 = dataset2.variables['lon'][:]
lon2[lon2>180]-=360
lon2[lon2<-180]+=360


def Haversine(lat1, lon1, lat2, lon2):
    """
    Haversine formula: determines the great-circle distance in km between two points on the (spherical) Earth given in lon/lat
    """
    deg2rad = math.pi / 180.0
    R = 6371 # Earth radius in km
    dlat = (lat2 - lat1) * deg2rad # latitudinal distance in radians
    dlon = (lon2 - lon1) * deg2rad # longitudinal distance in radians
    h = (math.sin(dlat/2))**2 + math.cos(lat1 * deg2rad) * math.cos(lat2 * deg2rad) * (math.sin(dlon/2))**2
    d = 2 * R * math.asin(min(1,math.sqrt(h))) # min of 1 and sqrt(h) to protect from roundoff errors for antipodal points
    return d


Particles = lon1.shape[0] # number of particle pairs
Times = lon1.shape[1] # number of times

meanSep = np.zeros(Times) # mean separation for each time
for t in range(Times): # for each time
    for i in range(Particles): # for each particle pair
        meanSep[t] += Haversine(lat1[i,t], lon1[i,t], lat2[i,t], lon2[i,t])
    meanSep[t] = meanSep[t]/Particles
    
savefile = '/science/projects/oceanparcels/output_data/data_Miriam/Results050deg/MeanSeparationTime_050deg'
meanSep.dump(savefile)