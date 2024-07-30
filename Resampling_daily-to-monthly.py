#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 27 23:43:50 2024

@author: dimitrisherrera
"""
# =============================================================================
# # Load the necessary packages
# =============================================================================
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

# Load daily CHIRPS data over the study area
chirp = xr.open_dataset('/Users/dimitrisherrera/Desktop/CHIRPSv2/CHIRPS-daily/chirps-v2.0.2023.days_p05.nc')
pr = chirp.precip[:,1100:1800,1000:3000]
lats = chirp.latitude[1100:1800]
lons = chirp.longitude[1000:3000]


# Resample daily data to monthly totals
monthly = pr.resample(time='1M').sum()

# Mask all values equal/below zero
M = monthly.where(monthly > 0.0, other=np.nan)

# Save as netCDF file
M.to_netcdf('/Users/dimitrisherrera/Desktop/CHIRPSv2/CHIRPS-monthly-from-daily/CHIRPS_monthly_2023.nc')
