# -*- coding: utf-8 -*-
"""
Tracking particles across the globe with GlobCurrent data

@author: Miriam Sterl
Based on code written by Victor Onink
"""

from parcels import Field, FieldSet, ParticleSet, Variable, JITParticle, AdvectionRK4, ErrorCode
from netCDF4 import Dataset
import numpy as np
import math
import datetime
from datetime import timedelta
from operator import attrgetter


""" ----- Functions to generate uniform initial particle distribution ----- """

def GlobCurrentLand(landfilename, fieldname):
    """
    Function to return a Field with True at land points and False at ocean points
    
    :param landfilename: name of the file which contains the velocity fields
    :param fieldname: name of field to use to determine where the land is
    :returns: a Field with True at land points and False at ocean points
    """
    landfile = Dataset(landfilename, 'r')
    Lon = landfile.variables['lon'][:] # longitudes from dataset
    Lat = landfile.variables['lat'][:] # latitudes from dataset
    f = landfile.variables[fieldname][:] # velocity field at grid points
    f = f[0] # the northward velocity at each grid point
    L = np.ma.getmask(f) # all land points are masked
    Land = Field('Land', L, transpose = False, lon=Lon, lat=Lat)
    return Land

def GenerateDistribution(landfilename, lonmin, lonmax, latmin, latmax, spacing):
    """
    Function that generates a particle distribution with no particles on land
    
    :param landfilename: name of the file which contains the velocity fields
    :param lonmin, lonmax, latmin, latmax: the minimum/maximum longitude/latitude values where we want particles
    :param spacing: the spacing between two neighbouring particles
    :returns: the longitudes and latitudes corresponding to the positions of the particles in this distribution
    """
    Land = GlobCurrentLand(landfilename, 'northward_eulerian_current_velocity')
    grid = np.mgrid[lonmin:lonmax:spacing, latmin:latmax:spacing] # lon/lat grid
    n = grid[0].size
    lons = np.reshape(grid[0], n) # all the particle longitudes
    lats = np.reshape(grid[1], n) # all the particle latitudes
    [lons,lats] = [np.array([lo for lo, la in zip(lons,lats) if Land[0,lo,la,0]==0.0]),
                   np.array([la for lo, la in zip(lons,lats) if Land[0,lo,la,0]==0.0])]
    return lons, lats

def GenerateParticles(landfilename, fieldset, pclass, lonmin, lonmax, latmin, latmax, spacing, starttime):
    """
    Function that generates a ParticleSet with particle distribution produces by GenerateDistribution
    
    :param landfilename: name of the file which contains the velocity fields
    :param fieldset: the fieldset to advect the ParticleSet on
    :param lonmin, lonmax, latmin, latmax: the minimum/maximum longitude/latitude values where we want particles
    :param spacing: the spacing between two neighbouring particles
    :returns: a ParticleSet with particles distributed at the positions generated by GenerateDistribution
    """
    [lons, lats] = GenerateDistribution(landfilename, lonmin, lonmax, latmin, latmax, spacing)
    pset = ParticleSet.from_list(fieldset=fieldset,
                                pclass=pclass,
                                lon=lons,
                                lat=lats,
                                time=starttime)
    return pset

    
""" ----- Creating the FieldSet ----- """

# We use the daily mean geostrophic + Ekman current from January 1st, 2002 until December 31st, 2014,
# together with an artificial anti-beaching boundary current
filenames = {'U': "/science/projects/oceanparcels/input_data/DataPlasticTides/GlobCurrent/20*.nc",
             'V': "/science/projects/oceanparcels/input_data/DataPlasticTides/GlobCurrent/20*.nc",
             'borU': "/science/projects/oceanparcels/input_data/DataPlasticTides/GlobCurrent/boundary_velocitiesTotal.nc",
             'borV': "/science/projects/oceanparcels/input_data/DataPlasticTides/GlobCurrent/boundary_velocitiesTotal.nc"}
variables = {'U': 'eastward_eulerian_current_velocity',
             'V': 'northward_eulerian_current_velocity',
             'borU': 'MaskUvel',
             'borV': 'MaskVvel'}
dimensions = {'lat': 'lat',
              'lon': 'lon',
              'time': 'time'}
fieldset = FieldSet.from_netcdf(filenames, variables, dimensions, allow_time_extrapolation=True)
fieldset.add_periodic_halo(zonal=True)


""" ----- Creating the ParticleSet ----- """

# We want to keep track of the distance travelled by the particles, so we create a new class:
class DistParticle(JITParticle):
    distance = Variable('distance', initial=0., dtype=np.float32) # the distance travelled
    prev_lon = Variable('prev_lon', dtype=np.float32, to_write=False,
                        initial=attrgetter('lon')) # the previous longitude
    prev_lat = Variable('prev_lat', dtype=np.float32, to_write=False,
                        initial=attrgetter('lat')) # the previous longitude

# random GC file to create initial distribution:
landfilename = "/science/projects/oceanparcels/input_data/DataPlasticTides/GlobCurrent/20120528-GLOBCURRENT-L4-CUReul_hs-ALT_SUM-v03.0-fv01.0.nc"
lonmin, lonmax = -179.75, 179.75
latmin, latmax = -75, 75.
spacing = 1
starttime = datetime.datetime(2002,1,1,0,0)
endtime = datetime.datetime(2014,12,31,21,0)
pset = GenerateParticles(landfilename, fieldset, DistParticle, lonmin, lonmax, latmin, latmax, spacing, starttime)
for p in pset:
    p.dt = 1 # to include the first date in the outputdata


""" ----- Functions and kernels for during advection ----- """

# Keeping track of the total distance travelled by a particle:
def TotalDistance(particle, fieldset, time, dt):
    """
    Function to calculate the total distance travelled by a particle (reported in km)
    """
    # Calculate the distance in latitudinal direction (using 1.11e2 kilometer per degree latitude)
    lat_dist = (particle.lat - particle.prev_lat) * 1.11e2
    # Calculate the distance in longitudinal direction, using cosine(latitude) - spherical earth
    lon_dist = (particle.lon - particle.prev_lon) * 1.11e2 * math.cos(particle.lat * math.pi / 180)
    # Calculate the total Euclidean distance travelled by the particle in km
    particle.distance += math.sqrt(math.pow(lon_dist, 2) + math.pow(lat_dist, 2))

    particle.prev_lon = particle.lon  # set the stored values for next iteration
    particle.prev_lat = particle.lat
    
# Anti-beaching boundary current:
def AntiBeach(particle, fieldset, time, dt):
    bu = fieldset.borU[time, particle.lon, particle.lat, particle.depth]
    bv = fieldset.borV[time, particle.lon, particle.lat, particle.depth]
    particle.lon -= bu*dt*0.00001
    particle.lat -= bv*dt*0.00001
    
# Casting the functions into Kernel objects and defining the final Kernel:
totalKernel = AdvectionRK4 + pset.Kernel(TotalDistance) + pset.Kernel(AntiBeach)

# Deleting particles that are out of bounds (for the recovery kernel):
def DeleteParticle(particle, fieldset, time, dt):
    particle.delete()
    

""" ----- Advecting the particles ----- """

# We save data for every 48 hours. The total run is from start 2002 to end 2014.
Time = starttime
steps = 0 # the number of steps at 2-day-intervals
while Time <= endtime:
    steps += 1
    Time += timedelta(hours=48) # 2 days

outputfile = pset.ParticleFile(name = "/science/projects/oceanparcels/output_data/data_Miriam/Results_TrackingGC.nc")
outputfile.write(pset, pset[0].time)

for nr in range(steps):
    # Advect the particles for 48 hours
    pset.execute(totalKernel,
                 runtime = timedelta(hours=48),
                 dt = timedelta(minutes=30), 
                 recovery = {ErrorCode.ErrorOutOfBounds: DeleteParticle}) # the recovery kernel
    outputfile.write(pset, pset[0].time)

