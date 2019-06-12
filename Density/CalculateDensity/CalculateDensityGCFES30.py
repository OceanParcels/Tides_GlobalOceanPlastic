# -*- coding: utf-8 -*-
"""
Calculating the particle density in nr. of particles/km^2

Code written by Victor Onink, modified by Miriam Sterl
"""

from netCDF4 import Dataset
import numpy as np

def AreaGridCells(sizeLon, sizeLat):
    """
    Function that calculates the surface area of grid cells on the Earth.
    
    :param sizeLon: the grid cell size in longitudinal direction
    :param sizeLat: the grid cell size in latitudinal direction
    :returns: matrix with the surface area (in km^2) of each grid cell
    """
    radians = np.pi/180.
    r = 6378.1 # the Earth radius in km
    lon_bins = np.linspace(-180, 180, sizeLon+1) # the bins in longitudinal direction
    lat_bins = np.linspace(-80, 80, sizeLat+1) # the bins in latitudinal direction
    Area = np.array([[radians*(lon_bins[i+1]-lon_bins[i])*
                      (np.sin(radians*lat_bins[j+1]) - np.sin(radians*lat_bins[j])) 
                      for j in range(len(lat_bins)-1)]
                      for i in range(len(lon_bins)-1)])
    Area = r*r*Area
    return Area

def CalculateDensityHistogram(londata, latdata, binsLon, binsLat):
    """
    Function that gives a histogram with the number of particles per km^2 in global grid cells.
    
    :param londata: the longitudes of all the particles at a certain time
    :param latdata: the latitudes of all the particles at a certain time
    :param binsLon: the bins in longitudinal direction
    :param binsLat: the bins in latitudinal direction
    :returns: the particle density in number of particles per km^2 for each global grid cell
    """
    londata, latdata = londata.reshape(np.size(londata)), latdata.reshape(np.size(latdata))
    density = np.zeros((len(binsLon), len(binsLat)))
    # For every particle, we check to which lon/lat in binsLon/binsLat its longitude/latitude is closest and put it in that bin
    for i in range(np.array(londata).shape[0]):
        density[np.argmin(np.abs(londata[i]-binsLon)),np.argmin(np.abs(latdata[i]-binsLat))]+=1
    # Now, normalize it by area
    area = AreaGridCells(len(binsLon),len(binsLat))
    density /= area
    return density

File = '/science/projects/oceanparcels/output_data/data_Miriam/Results_TrackingGCFES30.nc'
saveFile = '/science/projects/oceanparcels/output_data/data_Miriam/DensityGCFES30'
dataset = Dataset(File)
lat = dataset.variables['lat'][:]
lon = dataset.variables['lon'][:]
time = dataset.variables['time'][:]
lon[lon>180]-=360
lon[lon<-180]+=360

binsLon = np.arange(-180, 181, 1)
binsLat = np.arange(-80, 81, 1)
density = np.zeros((time.shape[1], len(binsLon), len(binsLat)))
for i in range(time.shape[1]):
    density[i,:,:] = CalculateDensityHistogram(lon[:,i], lat[:,i], binsLon, binsLat)
np.save(saveFile, density)

