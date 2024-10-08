#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 14:31:15 2024

@author: dimitrisherrera
"""
# =======================================================================
# Mask area out of circle
# =======================================================================
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as patches
import pandas as pd
import cartopy.crs as ccrs
import shapely.geometry as sgeom
from cartopy.geodesic import Geodesic
import cartopy.feature as cf
from tropycal import tracks
import matplotlib.path as mpath
import datetime

# =======================================================================
# Define basic functions
# =======================================================================
# Masking function
# Description:
    
def mask_tc(data, center_x, center_y, radius):
    
    # Calculate the distance from the center of the circle
    ny, nx = data.shape
    ix, iy = np.meshgrid(np.arange(nx), np.arange(ny))
    distance = np.sqrt((ix - center_x)**2 + (iy - center_y)**2)

    # Mask portions of the data array outside of the circle
    data_masked = np.ma.masked_where(distance > radius_grids, data)

    return data_masked # A masked array showing TC-associated precipitation. 


# Find TC's center indices function
# Description:
    
def cent_indices(lats, lat, lons, lon):    
    a = abs(lats-lat)+abs(lons-lon)
    i,j = np.unravel_index(a.argmin(),a.shape)
    return i,j


# Storm case function
# Description: 
def storm_case(storm, pr):
    storm = storm.to_dataframe()
    #storm = storm[storm.vmax >= 34] # 
    storm = storm.drop(['extra_obs', 'special', 'type','vmax', 'mslp', 'wmo_basin'], axis=1)
    storm = storm.set_index('time')
    storm.index = pd.to_datetime(storm.index)
    storm = storm.between_time('00:00', '18:00')
    storm = storm.resample('D').mean() # Calculate the positional mean for each day
    N = list(storm.index.shape)
    las = np.zeros((N))
    los = np.zeros((N))
    dates = storm.index
    date_index = dates.strftime('%j')
    date_index_list = date_index.to_list()
    data = None
    data = pr[int(date_index[0])-1 : int(date_index[-1]), :, :]    
    
    lalon = []
    for values in storm.index:
        lalon.append(storm.loc[values])
            
    for i in range(len(data)):
        las[i], los[i] = lalon[i]
    
    I, J = np.zeros((2, len(data)))
    for i in range(len(data)):
        I[i], J[i] = cent_indices(lats, las[i], lons, los[i])
           
    for i in range(len(data)):
        data[i,:,:] = mask_tc(data[i,:,:], J[i], I[i], radius_grids)
    
    return data#, los, las, dates


# Convert radius from km to number of grid cells
# Description: 
    
def radius_grid(DATA_res, radius_distance_km, latshape, lonshape):
    
    one_deg_distance = 111.0  #km approximately over the tropics
    #DATA_res = 0.05 #deg lat/lon
    DATA_distance = one_deg_distance * DATA_res
    #radius_distance_km = 500 #km
    latshape = lats.shape
    lonshape = lons.shape
    radius_grids = radius_distance_km/DATA_distance
    
    return radius_grids


# Create TC symbol function
# Description: 
    
def get_hurricane():
    u = np.array([  [2.444,7.553],
                    [0.513,7.046],
                    [-1.243,5.433],
                    [-2.353,2.975],
                    [-2.578,0.092],
                    [-2.075,-1.795],
                    [-0.336,-2.870],
                    [2.609,-2.016]  ])
    u[:,0] -= 0.098
    codes = [1] + [2]*(len(u)-2) + [2] 
    u = np.append(u, -u[::-1], axis=0)
    codes += codes

    return mpath.Path(3*u, codes, closed=False)


# Plotting TC function
# Plot the masked data using Cartopy to evaluate whether
# the masking process worked as expected

def plot_swath(masked, lons, lats, date, title):#lon, lat, marker, date, title):
    #plt.style.use('dark_background')
    gd = Geodesic()
    src_crs = ccrs.PlateCarree()
    lcc = ccrs.Robinson() 
    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(111, projection=lcc)
    ax.coastlines(resolution='50m')
    cmap = mpl.cm.winter_r
    #bounds = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0]
    bounds = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0, 130.0, 140.0, 150]
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N, extend='both')
    cax = ax.pcolor(lons, lats, masked, transform=ccrs.PlateCarree(), cmap='winter_r', norm=norm)
    #geoms = []
    #for lon, lat in zip(stn['longitude'], stn['latitude']):
    #cp = gd.circle(lon=lon, lat=lat, radius=500000.) # showing a 500 km radius
    #geoms.append(sgeom.Polygon(cp))
      
       # ax.add_geometries(geoms, crs=src_crs, edgecolor='r', alpha=0.1)
    ax.set_extent([-125.0, -52.0, 5.0, 36.5])
    #ax.scatter(lon, lat, s=250, marker=marker, c='red',transform=ccrs.PlateCarree())
    ax.add_feature(cf.BORDERS, linewidth=0.5)    
    cbar = fig.colorbar(cax, orientation='horizontal', extendfrac='auto', fraction=0.06, pad=0.02)
    cbar.ax.tick_params(labelsize=15)
    cbar.set_label('mm/day', fontsize=15, labelpad=5)
    plt.title(title,fontsize=20)
    
    plt.suptitle(date, x=0.28, y=0.22)
    plt.text(0.14, 0.11, 'Date',
        verticalalignment='bottom', horizontalalignment='left',
        transform=ax.transAxes,
        color='black', fontsize=15)
    
    #plt.savefig("Season_2005"+str(date)+".png")
    return(plt.show())



# =======================================================================
# Using functions
# =======================================================================

# Load daily CHIRPS data over the study area
chirp = xr.open_dataset('/Users/dimitrisherrera/Desktop/CHIRPS_daily/chirps-v2.0.2023.days_p05.nc')
pr = chirp.precip[:,1100:1800,1000:3000]
lats = chirp.latitude[1100:1800]
lons = chirp.longitude[1000:3000]

# Define a circle in the center of the data with a radius of X pixels
# We can get the radius in pixels using the radius_grid function.
radius_grids = radius_grid(0.05, 500, lats.shape, lons.shape)

# Load TC tracks from Hurdat2 using the Tropycal package
#basin = tracks.TrackDataset(basin='both',include_btk=False)
basin = tracks.TrackDataset(basin='both',include_btk=False)

# Get the storms for the year from Hurdat2 using Tropycal
season0 = basin.get_season(2023)
season1 = season0.to_dataframe()

# Select TCs with at least 34 mph (e.g., TC that reached at least the tropical storm category)
season1 = season1[season1.vmax >= 34]
tcs = season1['id'].values.tolist()

# In case the TC list has 'UNNAMED' disturbances included, we remove all those disturbances
#tcs = [i for i in tcs if i != 'UNNAMED']

tcs_season = []
for tc in tcs:
    one_tc = basin.get_storm((str(tc)))#,1990))
    tcs_season.append(one_tc)

trop_cyc = []
for i in range(len(tcs)):
    one_trop_cyc = storm_case(tcs_season[i], pr)
    trop_cyc.append(one_trop_cyc)
 
# Combine all TCS of the year for the NA and E Pacific basins
Season = xr.merge(trop_cyc) 
                                                             
# Create an empty array with the dimensions of the original precip. dataset
PR = pr.copy()
PR = PR.where(PR > -1, other=np.nan)
PR = PR.where(PR.isnull())

# Combine the empty precip array with TCs of the year
PRR = PR.combine_first(Season)

# Aggregate the resulting TC precipitation data to monthly 
# totals. This will give us the total rainfall from TV for a given month.
monthly =PRR.resample(time='1M').sum()

# Mask all values equal/below zero
M = monthly.where(monthly > 0.0, other=np.nan)

M.to_netcdf('/Users/dimitrisherrera/Desktop/TC_contributions/TC_monthly_2023.nc')

'''# =======================================================================
# Plotting 
# =======================================================================

# For plotting purposes, we mask rainfall rates 
# below 1 mm/day
PRR2 = PRR.precip
PRR2 = PRR2.where(PRR2 > 1.0, other=np.nan)

# Create a TC symbol
hurricane = get_hurricane()

#Notes to create datetime for plots
start = datetime.datetime.strptime("01-01-0000", "%d-%m-%Y")
end = datetime.datetime.strptime("01-01-0000", "%d-%m-%Y")
date_generated = [start + datetime.timedelta(days=x) for x in range(0, (end-start).days)]
datest = pd.to_datetime(date_generated)

for i in range(len(PRR2)):
    plot_swath(PRR2[i,:,:], lons, lats, datest[i], "Hurricane Season 1998")

path = '/Users/dimitrisherrera/Desktop/CHIRPS_daily/chirps-v2.0.%s.days_p05.nc'
chirp = xr.open_dataset(path %(2000))

for i in range(len(PRR2)):
    plot_swath(PRR2[i,:,:], lons, lats, datest[i], "Hurricane Season 2005")
    #day+=1
    plt.savefig('/Users/dimitrisherrera/Downloads/IRMA/2005/'+ list(datest[i)] +'.png')'''
    
    
    
    
