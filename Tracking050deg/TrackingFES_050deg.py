# -*- coding: utf-8 -*-
"""
Tracking particles across the globe with FES data for M2+S2+K1+O1 tidal field, particle spacing 0.5 degrees

@author: Miriam Sterl
"""

from parcels import Field, FieldSet, ParticleSet, JITParticle, Variable, ErrorCode
from netCDF4 import Dataset
import numpy as np
import math
import datetime
from datetime import timedelta
from operator import attrgetter


""" ----- Computation of tidal currents ----- """

t0 = datetime.datetime(1900,1,1,0,0) # origin of time = 1 January 1900, 00:00:00 UTC
starttime = datetime.datetime(2002,1,1,0,0) # time when the simulation starts = 1 January 2002, 00:00:00 UTC
endtime = datetime.datetime(2014,12,31,21,0) # time when the simulation ends = 31 December 2014, 21:00:00 UTC


def TidalMotionM2S2K1O1(particle, fieldset, time, dt):
    """
    Kernel that calculates tidal currents U and V due to M2, S2, K1 and O1 tide at particle location and time
    and advects the particle in these currents (using Euler forward scheme)
    Calculations based on Doodson (1921) and Schureman (1958)
    """        
    # Number of Julian centuries that have passed between t0 and time
    t = ((time + fieldset.t0rel)/86400.0)/36525.0
    
    # Define constants to compute astronomical variables T, h, s, N (all in degrees) (source: FES2014 code)
    cT0 = 180.0
    ch0 = 280.1895
    cs0 = 277.0248
    cN0 = 259.1568; cN1 = -1934.1420
    deg2rad = math.pi/180.0
    
    # Calculation of factors T, h, s at t0 (source: Doodson (1921))
    T0 = math.fmod(cT0, 360.0) * deg2rad
    h0 = math.fmod(ch0, 360.0) * deg2rad
    s0 = math.fmod(cs0, 360.0) * deg2rad
    
    # Calculation of V(t0) (source: Schureman (1958))
    V_M2 = 2*T0 + 2*h0 - 2*s0
    V_S2 = 2*T0
    V_K1 = T0 + h0 - 0.5*math.pi
    V_O1 = T0 + h0 - 2*s0 + 0.5*math.pi
    
    # Calculation of factors N, I, nu, xi at time (source: Schureman (1958))
    # Since these factors change only very slowly over time, we take them as constant over the time step dt
    N = math.fmod(cN0 + cN1*t, 360.0) * deg2rad
    I = math.acos(0.91370 - 0.03569*math.cos(N))
    tanN = math.tan(0.5*N)
    at1 = math.atan(1.01883 * tanN)
    at2 = math.atan(0.64412 * tanN)
    nu = at1 - at2
    xi = -at1 - at2 + N
    nuprim = math.atan(math.sin(2*I) * math.sin(nu)/(math.sin(2*I)*math.cos(nu) + 0.3347))
    
    # Calculation of u, f at current time (source: Schureman (1958))
    u_M2 = 2*xi - 2*nu
    f_M2 = (math.cos(0.5*I))**4/0.9154
    u_S2 = 0
    f_S2 = 1
    u_K1 = -nuprim
    f_K1 = math.sqrt(0.8965*(math.sin(2*I))**2 + 0.6001*math.sin(2*I)*math.cos(nu) + 0.1006)
    u_O1 = 2*xi - nu
    f_O1 = math.sin(I)*(math.cos(0.5*I))**2/0.3800
    
    # Fourth-order Runge-Kutta methode to advect particle in tidal currents
    
    # ----------------------- STEP 1 -----------------------
    # Tidal fields have longitudes defined from 0...360 degrees (so -180...0 --> 180...360)
    if particle.lon < 0:
        lon = particle.lon + 360
    else:
        lon = particle.lon

    # Zonal amplitudes and phaseshifts at particle location and time
    Uampl_M2_1 = f_M2 * fieldset.UaM2[time, lon, particle.lat, particle.depth]
    Upha_M2_1 = V_M2 + u_M2 - fieldset.UgM2[time, lon, particle.lat, particle.depth]
    Uampl_S2_1 = f_S2 * fieldset.UaS2[time, lon, particle.lat, particle.depth]
    Upha_S2_1 = V_S2 + u_S2 - fieldset.UgS2[time, lon, particle.lat, particle.depth] 
    Uampl_K1_1 = f_K1 * fieldset.UaK1[time, lon, particle.lat, particle.depth]
    Upha_K1_1 = V_K1 + u_K1 - fieldset.UgK1[time, lon, particle.lat, particle.depth]
    Uampl_O1_1 = f_O1 * fieldset.UaO1[time, lon, particle.lat, particle.depth]
    Upha_O1_1 = V_O1 + u_O1 - fieldset.UgO1[time, lon, particle.lat, particle.depth]   
    # Meridional amplitudes and phaseshifts at particle location and time
    Vampl_M2_1 = f_M2 * fieldset.VaM2[time, lon, particle.lat, particle.depth]
    Vpha_M2_1 = V_M2 + u_M2 - fieldset.VgM2[time, lon, particle.lat, particle.depth] 
    Vampl_S2_1 = f_S2 * fieldset.VaS2[time, lon, particle.lat, particle.depth] 
    Vpha_S2_1 = V_S2 + u_S2 - fieldset.VgS2[time, lon, particle.lat, particle.depth] 
    Vampl_K1_1 = f_K1 * fieldset.VaK1[time, lon, particle.lat, particle.depth]
    Vpha_K1_1 = V_K1 + u_K1 - fieldset.VgK1[time, lon, particle.lat, particle.depth]
    Vampl_O1_1 = f_O1 * fieldset.VaO1[time, lon, particle.lat, particle.depth]
    Vpha_O1_1 = V_O1 + u_O1 - fieldset.VgO1[time, lon, particle.lat, particle.depth]   
    # Zonal and meridional tidal currents; time + fieldset.t0rel = number of seconds elapsed between t0 and time
    Uvel_M2_1 = Uampl_M2_1 * math.cos(fieldset.omegaM2 * (time + fieldset.t0rel) + Upha_M2_1)
    Uvel_S2_1 = Uampl_S2_1 * math.cos(fieldset.omegaS2 * (time + fieldset.t0rel) + Upha_S2_1)
    Uvel_K1_1 = Uampl_K1_1 * math.cos(fieldset.omegaK1 * (time + fieldset.t0rel) + Upha_K1_1)
    Uvel_O1_1 = Uampl_O1_1 * math.cos(fieldset.omegaO1 * (time + fieldset.t0rel) + Upha_O1_1)
    Vvel_M2_1 = Vampl_M2_1 * math.cos(fieldset.omegaM2 * (time + fieldset.t0rel) + Vpha_M2_1)
    Vvel_S2_1 = Vampl_S2_1 * math.cos(fieldset.omegaS2 * (time + fieldset.t0rel) + Vpha_S2_1)
    Vvel_K1_1 = Vampl_K1_1 * math.cos(fieldset.omegaK1 * (time + fieldset.t0rel) + Vpha_K1_1)
    Vvel_O1_1 = Vampl_O1_1 * math.cos(fieldset.omegaO1 * (time + fieldset.t0rel) + Vpha_O1_1)
    # Total zonal and meridional velocity
    U1 = Uvel_M2_1 + Uvel_S2_1 + Uvel_K1_1 + Uvel_O1_1 # total zonal velocity
    V1 = Vvel_M2_1 + Vvel_S2_1 + Vvel_K1_1 + Vvel_O1_1 # total meridional velocity
    # New lon + lat
    lon1, lat1 = (particle.lon + U1*0.5*dt, particle.lat + V1*0.5*dt)
    
    # ----------------------- STEP 2 -----------------------
    if lon1 < 0:
        lon1 += 360

    # Zonal amplitudes and phaseshifts at particle location and time
    Uampl_M2_2 = f_M2 * fieldset.UaM2[time + 0.5*dt, lon1, lat1, particle.depth]
    Upha_M2_2 = V_M2 + u_M2 - fieldset.UgM2[time + 0.5*dt, lon1, lat1, particle.depth]
    Uampl_S2_2 = f_S2 * fieldset.UaS2[time + 0.5*dt, lon1, lat1, particle.depth]
    Upha_S2_2 = V_S2 + u_S2 - fieldset.UgS2[time + 0.5*dt, lon1, lat1, particle.depth] 
    Uampl_K1_2 = f_K1 * fieldset.UaK1[time + 0.5*dt, lon1, lat1, particle.depth]
    Upha_K1_2 = V_K1 + u_K1 - fieldset.UgK1[time + 0.5*dt, lon1, lat1, particle.depth]
    Uampl_O1_2 = f_O1 * fieldset.UaO1[time + 0.5*dt, lon1, lat1, particle.depth]
    Upha_O1_2 = V_O1 + u_O1 - fieldset.UgO1[time + 0.5*dt, lon1, lat1, particle.depth]   
    # Meridional amplitudes and phaseshifts at particle location and time
    Vampl_M2_2 = f_M2 * fieldset.VaM2[time + 0.5*dt, lon1, lat1, particle.depth]
    Vpha_M2_2 = V_M2 + u_M2 - fieldset.VgM2[time + 0.5*dt, lon1, lat1, particle.depth] 
    Vampl_S2_2 = f_S2 * fieldset.VaS2[time + 0.5*dt, lon1, lat1, particle.depth] 
    Vpha_S2_2 = V_S2 + u_S2 - fieldset.VgS2[time + 0.5*dt, lon1, lat1, particle.depth] 
    Vampl_K1_2 = f_K1 * fieldset.VaK1[time + 0.5*dt, lon1, lat1, particle.depth]
    Vpha_K1_2 = V_K1 + u_K1 - fieldset.VgK1[time + 0.5*dt, lon1, lat1, particle.depth]
    Vampl_O1_2 = f_O1 * fieldset.VaO1[time + 0.5*dt, lon1, lat1, particle.depth]
    Vpha_O1_2 = V_O1 + u_O1 - fieldset.VgO1[time + 0.5*dt, lon1, lat1, particle.depth]   
    # Zonal and meridional tidal currents; time + fieldset.t0rel = number of seconds elapsed between t0 and time
    Uvel_M2_2 = Uampl_M2_2 * math.cos(fieldset.omegaM2 * (time + 0.5*dt + fieldset.t0rel) + Upha_M2_2)
    Uvel_S2_2 = Uampl_S2_2 * math.cos(fieldset.omegaS2 * (time + 0.5*dt + fieldset.t0rel) + Upha_S2_2)
    Uvel_K1_2 = Uampl_K1_2 * math.cos(fieldset.omegaK1 * (time + 0.5*dt + fieldset.t0rel) + Upha_K1_2)
    Uvel_O1_2 = Uampl_O1_2 * math.cos(fieldset.omegaO1 * (time + 0.5*dt + fieldset.t0rel) + Upha_O1_2)
    Vvel_M2_2 = Vampl_M2_2 * math.cos(fieldset.omegaM2 * (time + 0.5*dt + fieldset.t0rel) + Vpha_M2_2)
    Vvel_S2_2 = Vampl_S2_2 * math.cos(fieldset.omegaS2 * (time + 0.5*dt + fieldset.t0rel) + Vpha_S2_2)
    Vvel_K1_2 = Vampl_K1_2 * math.cos(fieldset.omegaK1 * (time + 0.5*dt + fieldset.t0rel) + Vpha_K1_2)
    Vvel_O1_2 = Vampl_O1_2 * math.cos(fieldset.omegaO1 * (time + 0.5*dt + fieldset.t0rel) + Vpha_O1_2)
    # Total zonal and meridional velocity
    U2 = Uvel_M2_2 + Uvel_S2_2 + Uvel_K1_2 + Uvel_O1_2 # total zonal velocity
    V2 = Vvel_M2_2 + Vvel_S2_2 + Vvel_K1_2 + Vvel_O1_2 # total meridional velocity
    # New lon + lat
    lon2, lat2 = (particle.lon + U2*0.5*dt, particle.lat + V2*0.5*dt)  
    
    # ----------------------- STEP 3 -----------------------
    if lon2 < 0:
        lon2 += 360

    # Zonal amplitudes and phaseshifts at particle location and time
    Uampl_M2_3 = f_M2 * fieldset.UaM2[time + 0.5*dt, lon2, lat2, particle.depth]
    Upha_M2_3 = V_M2 + u_M2 - fieldset.UgM2[time + 0.5*dt, lon2, lat2, particle.depth]
    Uampl_S2_3 = f_S2 * fieldset.UaS2[time + 0.5*dt, lon2, lat2, particle.depth]
    Upha_S2_3 = V_S2 + u_S2 - fieldset.UgS2[time + 0.5*dt, lon2, lat2, particle.depth] 
    Uampl_K1_3 = f_K1 * fieldset.UaK1[time + 0.5*dt, lon2, lat2, particle.depth]
    Upha_K1_3 = V_K1 + u_K1 - fieldset.UgK1[time + 0.5*dt, lon2, lat2, particle.depth]
    Uampl_O1_3 = f_O1 * fieldset.UaO1[time + 0.5*dt, lon2, lat2, particle.depth]
    Upha_O1_3 = V_O1 + u_O1 - fieldset.UgO1[time + 0.5*dt, lon2, lat2, particle.depth]   
    # Meridional amplitudes and phaseshifts at particle location and time
    Vampl_M2_3 = f_M2 * fieldset.VaM2[time + 0.5*dt, lon2, lat2, particle.depth]
    Vpha_M2_3 = V_M2 + u_M2 - fieldset.VgM2[time + 0.5*dt, lon2, lat2, particle.depth] 
    Vampl_S2_3 = f_S2 * fieldset.VaS2[time + 0.5*dt, lon2, lat2, particle.depth] 
    Vpha_S2_3 = V_S2 + u_S2 - fieldset.VgS2[time + 0.5*dt, lon2, lat2, particle.depth] 
    Vampl_K1_3 = f_K1 * fieldset.VaK1[time + 0.5*dt, lon2, lat2, particle.depth]
    Vpha_K1_3 = V_K1 + u_K1 - fieldset.VgK1[time + 0.5*dt, lon2, lat2, particle.depth]
    Vampl_O1_3 = f_O1 * fieldset.VaO1[time + 0.5*dt, lon2, lat2, particle.depth]
    Vpha_O1_3 = V_O1 + u_O1 - fieldset.VgO1[time + 0.5*dt, lon2, lat2, particle.depth]   
    # Zonal and meridional tidal currents; time + fieldset.t0rel = number of seconds elapsed between t0 and time
    Uvel_M2_3 = Uampl_M2_3 * math.cos(fieldset.omegaM2 * (time + 0.5*dt + fieldset.t0rel) + Upha_M2_3)
    Uvel_S2_3 = Uampl_S2_3 * math.cos(fieldset.omegaS2 * (time + 0.5*dt + fieldset.t0rel) + Upha_S2_3)
    Uvel_K1_3 = Uampl_K1_3 * math.cos(fieldset.omegaK1 * (time + 0.5*dt + fieldset.t0rel) + Upha_K1_3)
    Uvel_O1_3 = Uampl_O1_3 * math.cos(fieldset.omegaO1 * (time + 0.5*dt + fieldset.t0rel) + Upha_O1_3)
    Vvel_M2_3 = Vampl_M2_3 * math.cos(fieldset.omegaM2 * (time + 0.5*dt + fieldset.t0rel) + Vpha_M2_3)
    Vvel_S2_3 = Vampl_S2_3 * math.cos(fieldset.omegaS2 * (time + 0.5*dt + fieldset.t0rel) + Vpha_S2_3)
    Vvel_K1_3 = Vampl_K1_3 * math.cos(fieldset.omegaK1 * (time + 0.5*dt + fieldset.t0rel) + Vpha_K1_3)
    Vvel_O1_3 = Vampl_O1_3 * math.cos(fieldset.omegaO1 * (time + 0.5*dt + fieldset.t0rel) + Vpha_O1_3)
    # Total zonal and meridional velocity
    U3 = Uvel_M2_3 + Uvel_S2_3 + Uvel_K1_3 + Uvel_O1_3 # total zonal velocity
    V3 = Vvel_M2_3 + Vvel_S2_3 + Vvel_K1_3 + Vvel_O1_3 # total meridional velocity
    # New lon + lat
    lon3, lat3 = (particle.lon + U3*dt, particle.lat + V3*dt) 
    
    # ----------------------- STEP 4 -----------------------
    if lon3 < 0:
        lon3 += 360

    # Zonal amplitudes and phaseshifts at particle location and time
    Uampl_M2_4 = f_M2 * fieldset.UaM2[time + dt, lon3, lat3, particle.depth]
    Upha_M2_4 = V_M2 + u_M2 - fieldset.UgM2[time + dt, lon3, lat3, particle.depth]
    Uampl_S2_4 = f_S2 * fieldset.UaS2[time + dt, lon3, lat3, particle.depth]
    Upha_S2_4 = V_S2 + u_S2 - fieldset.UgS2[time + dt, lon3, lat3, particle.depth] 
    Uampl_K1_4 = f_K1 * fieldset.UaK1[time + dt, lon3, lat3, particle.depth]
    Upha_K1_4 = V_K1 + u_K1 - fieldset.UgK1[time + dt, lon3, lat3, particle.depth]
    Uampl_O1_4 = f_O1 * fieldset.UaO1[time + dt, lon3, lat3, particle.depth]
    Upha_O1_4 = V_O1 + u_O1 - fieldset.UgO1[time + dt, lon3, lat3, particle.depth]   
    # Meridional amplitudes and phaseshifts at particle location and time
    Vampl_M2_4 = f_M2 * fieldset.VaM2[time + dt, lon3, lat3, particle.depth]
    Vpha_M2_4 = V_M2 + u_M2 - fieldset.VgM2[time + dt, lon3, lat3, particle.depth] 
    Vampl_S2_4 = f_S2 * fieldset.VaS2[time + dt, lon3, lat3, particle.depth] 
    Vpha_S2_4 = V_S2 + u_S2 - fieldset.VgS2[time + dt, lon3, lat3, particle.depth] 
    Vampl_K1_4 = f_K1 * fieldset.VaK1[time + dt, lon3, lat3, particle.depth]
    Vpha_K1_4 = V_K1 + u_K1 - fieldset.VgK1[time + dt, lon3, lat3, particle.depth]
    Vampl_O1_4 = f_O1 * fieldset.VaO1[time + dt, lon3, lat3, particle.depth]
    Vpha_O1_4 = V_O1 + u_O1 - fieldset.VgO1[time + dt, lon3, lat3, particle.depth]   
    # Zonal and meridional tidal currents; time + fieldset.t0rel = number of seconds elapsed between t0 and time
    Uvel_M2_4 = Uampl_M2_4 * math.cos(fieldset.omegaM2 * (time + dt + fieldset.t0rel) + Upha_M2_4)
    Uvel_S2_4 = Uampl_S2_4 * math.cos(fieldset.omegaS2 * (time + dt + fieldset.t0rel) + Upha_S2_4)
    Uvel_K1_4 = Uampl_K1_4 * math.cos(fieldset.omegaK1 * (time + dt + fieldset.t0rel) + Upha_K1_4)
    Uvel_O1_4 = Uampl_O1_4 * math.cos(fieldset.omegaO1 * (time + dt + fieldset.t0rel) + Upha_O1_4)
    Vvel_M2_4 = Vampl_M2_4 * math.cos(fieldset.omegaM2 * (time + dt + fieldset.t0rel) + Vpha_M2_4)
    Vvel_S2_4 = Vampl_S2_4 * math.cos(fieldset.omegaS2 * (time + dt + fieldset.t0rel) + Vpha_S2_4)
    Vvel_K1_4 = Vampl_K1_4 * math.cos(fieldset.omegaK1 * (time + dt + fieldset.t0rel) + Vpha_K1_4)
    Vvel_O1_4 = Vampl_O1_4 * math.cos(fieldset.omegaO1 * (time + dt + fieldset.t0rel) + Vpha_O1_4)
    # Total zonal and meridional velocity
    U4 = Uvel_M2_4 + Uvel_S2_4 + Uvel_K1_4 + Uvel_O1_4 # total zonal velocity
    V4 = Vvel_M2_4 + Vvel_S2_4 + Vvel_K1_4 + Vvel_O1_4 # total meridional velocity
    
    # Finally, the new particle location:
    particle.lon += (U1 + 2*U2 + 2*U3 + U4)/6. * dt
    particle.lat += (V1 + 2*V2 + 2*V3 + V4)/6. * dt
    




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





""" ----- Creating the FieldSet with GlobCurrent data ----- """

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
fieldset.add_constant('t0rel', (starttime - t0).total_seconds()) # number of seconds elapsed between t0 and starttime





""" ----- Creating the tidal Fields ----- """

files_eastward = "/science/projects/oceanparcels/input_data/DataPlasticTides/FES/eastward_velocity/"
files_northward = "/science/projects/oceanparcels/input_data/DataPlasticTides/FES/northward_velocity/"

dimensions_Ua = {'data': 'Ua', 'lat': 'lat', 'lon': 'lon'}
dimensions_Ug = {'data': 'Ug', 'lat': 'lat', 'lon': 'lon'}
dimensions_Va = {'data': 'Va', 'lat': 'lat', 'lon': 'lon'}
dimensions_Vg = {'data': 'Vg', 'lat': 'lat', 'lon': 'lon'}

deg2rad = math.pi/180.0 # factor to convert degrees to radians

""" --- M2 component --- """

filename_UM2 = files_eastward + 'm2.nc'
filename_VM2 = files_northward + 'm2.nc'

UaM2 = Field.from_netcdf(filename_UM2, 'UaM2', dimensions_Ua, fieldtype='U')
UaM2.set_scaling_factor(1e-2) # convert from cm/s to m/s
UgM2 = Field.from_netcdf(filename_UM2, 'UgM2', dimensions_Ug)
UgM2.set_scaling_factor(deg2rad) # convert from degrees to radians
VaM2 = Field.from_netcdf(filename_VM2, 'VaM2', dimensions_Va, fieldtype='V')
VaM2.set_scaling_factor(1e-2)
VgM2 = Field.from_netcdf(filename_VM2, 'VgM2', dimensions_Vg)
VgM2.set_scaling_factor(deg2rad)

fieldset.add_field(UaM2)
fieldset.add_field(UgM2)
fieldset.add_field(VaM2)
fieldset.add_field(VgM2)

omega_M2 = 28.9841042 # angular frequency of M2 in degrees per hour
fieldset.add_constant('omegaM2', (omega_M2 * deg2rad) / 3600.0) # angular frequency of M2 in radians per second

""" --- S2 component --- """

filename_US2 = files_eastward + 's2.nc'
filename_VS2 = files_northward + 's2.nc'

UaS2 = Field.from_netcdf(filename_US2, 'UaS2', dimensions_Ua, fieldtype='U')
UaS2.set_scaling_factor(1e-2)
UgS2 = Field.from_netcdf(filename_US2, 'UgS2', dimensions_Ug)
UgS2.set_scaling_factor(deg2rad)
VaS2 = Field.from_netcdf(filename_VS2, 'VaS2', dimensions_Va, fieldtype='V')
VaS2.set_scaling_factor(1e-2)
VgS2 = Field.from_netcdf(filename_VS2, 'VgS2', dimensions_Vg)
VgS2.set_scaling_factor(deg2rad)

fieldset.add_field(UaS2)
fieldset.add_field(UgS2)
fieldset.add_field(VaS2)
fieldset.add_field(VgS2)

omega_S2 = 30.0000000 # angular frequency of S2 in degrees per hour
fieldset.add_constant('omegaS2', (omega_S2 * deg2rad) / 3600.0) # angular frequency of S2 in radians per second

""" --- K1 component --- """

filename_UK1 = files_eastward + 'k1.nc'
filename_VK1 = files_northward + 'k1.nc'

UaK1 = Field.from_netcdf(filename_UK1, 'UaK1', dimensions_Ua, fieldtype='U')
UaK1.set_scaling_factor(1e-2)
UgK1 = Field.from_netcdf(filename_UK1, 'UgK1', dimensions_Ug)
UgK1.set_scaling_factor(deg2rad)
VaK1 = Field.from_netcdf(filename_VK1, 'VaK1', dimensions_Va, fieldtype='V')
VaK1.set_scaling_factor(1e-2)
VgK1 = Field.from_netcdf(filename_VK1, 'VgK1', dimensions_Vg)
VgK1.set_scaling_factor(deg2rad)

fieldset.add_field(UaK1)
fieldset.add_field(UgK1)
fieldset.add_field(VaK1)
fieldset.add_field(VgK1)

omega_K1 = 15.0410686 # angular frequency of K1 in degrees per hour
fieldset.add_constant('omegaK1', (omega_K1 * deg2rad) / 3600.0) # angular frequency of K1 in radians per second

""" --- O1 component --- """

filename_UO1 = files_eastward + 'o1.nc'
filename_VO1 = files_northward + 'o1.nc'

UaO1 = Field.from_netcdf(filename_UO1, 'UaO1', dimensions_Ua, fieldtype='U')
UaO1.set_scaling_factor(1e-2)
UgO1 = Field.from_netcdf(filename_UO1, 'UgO1', dimensions_Ug)
UgO1.set_scaling_factor(deg2rad)
VaO1 = Field.from_netcdf(filename_VO1, 'VaO1', dimensions_Va, fieldtype='V')
VaO1.set_scaling_factor(1e-2)
VgO1 = Field.from_netcdf(filename_VO1, 'VgO1', dimensions_Vg)
VgO1.set_scaling_factor(deg2rad)

fieldset.add_field(UaO1)
fieldset.add_field(UgO1)
fieldset.add_field(VaO1)
fieldset.add_field(VgO1)

omega_O1 = 13.9430356 # angular frequency of O1 in degrees per hour
fieldset.add_constant('omegaO1', (omega_O1 * deg2rad) / 3600.0) # angular frequency of O1 in radians per second



# Add a zonal periodic halo to all the fields in fieldset
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
spacing = 0.5
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
    lat_dist = (particle.lat - particle.prev_lat) * 1.11e2
    lon_dist = (particle.lon - particle.prev_lon) * 1.11e2 * math.cos(particle.lat * math.pi / 180)
    particle.distance += math.sqrt(math.pow(lon_dist, 2) + math.pow(lat_dist, 2))

    particle.prev_lon = particle.lon
    particle.prev_lat = particle.lat

# Anti-beaching boundary current:
def AntiBeach(particle, fieldset, time, dt):
    bu = fieldset.borU[time, particle.lon, particle.lat, particle.depth]
    bv = fieldset.borV[time, particle.lon, particle.lat, particle.depth]
    particle.lon -= bu*dt*0.00001
    particle.lat -= bv*dt*0.00001

# Casting the functions into Kernel objects and defining the final Kernel:
totalKernel = pset.Kernel(TotalDistance) + pset.Kernel(AntiBeach) + pset.Kernel(TidalMotionM2S2K1O1)

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

outputfile = pset.ParticleFile(name = "/science/projects/oceanparcels/output_data/data_Miriam/Results050deg/Results_TrackingFES_050deg.nc")
outputfile.write(pset, pset[0].time)

for nr in range(steps):
    # Advect the particles for 48 hours
    pset.execute(totalKernel,
                 runtime = timedelta(hours=48),
                 dt = timedelta(minutes=30),
                 recovery = {ErrorCode.ErrorOutOfBounds: DeleteParticle}) # the recovery kernel
    outputfile.write(pset, pset[0].time)


