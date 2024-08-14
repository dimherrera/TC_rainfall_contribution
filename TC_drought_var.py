#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 21:18:28 2024

@author: dimitrisherrera
"""

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
import pandas as pd

# =============================================================================
# drought_change function:
# This function counts the number of months where a specific drought metric 
# (e.g., SPEI, PDSI) improved at least one unit at i-month and j-lat, k-lon
# It also conducts a t-test to estimate whether the difference between 
# TC-rain drought index is statistically significant to non-TC rain metric.
# =============================================================================
def drought_change(tc_masked, drought_index_o, drought_index_r):  
    '''
    
    Parameters
    ----------
    tc_masked : array
        array with precipitation solely from TC. It will be used to mask
        the areas under TC's tracks.
    drought_index_o : array
        Darray with the drought metric calculated before the removal
        of TC precipitation (e.g., total rainfall in the dataset).
    drought_index_r : array
        array with the drought metric calculated after the removal of
        TC-associated precipitation..

    Returns
    -------
    count_plus_one : array
        DESCRIPTION.
    pval : array
        DESCRIPTION.
    selected_o : array
        DESCRIPTION.
    selected_r : array
        DESCRIPTION.

    ''' 
    
    # Mask the drought_index values not associated to TC rainfall.
    # That would allow us to see the drought_index values associated with TCs
    # We begin masking the drought index calculated with total precipitation
    selected_o = drought_index_o * tc_masked
    
    # We do the same with the drought index calculated after removing TC-precipitation
    selected_r = drought_index_r * tc_masked
    
    # Now, estimate how TC decreased drought each month, by substracting 
    # selected_r from selected_o. This will show the difference between the 
    # drought index with total precip minus the same metric calculated after
    # the removal of TC precipitation. This yields the change
    # of the index solely related to TC precipitation on monthly time sacles.
    drought_change_mon = selected_o - selected_r
     
    # Now, let's count the number of months where the drought index changed
    # at least +1.0, which indicates a significant drought improvement. We do
    # that on each grid cell (lat/lon)
    count_plus_one = np.zeros((len(tc_masked[0]), len(tc_masked[0][0])))
    count_end_drought = np.zeros((len(tc_masked[0]), len(tc_masked[0][0])))

    for i in range(len(tc_masked[0])):
        for j in range(len(tc_masked[0][0])):
            count_plus_one[i,j] = (drought_change_mon[:, i, j] > 1.0).sum()
            count_end_drought[i,j] = ((selected_r[:, i, j] < -0.9) & (selected_o[:, i, j] >= 0.9)).sum()

    # Now, let's see if the differences between TC-rainfall removed and total rainfall 
    # are statistically-significant
    stat, pval = ttest_ind(selected_r, selected_o)
    
    # Mask p-values higher than 0.05
    pval = np.ma.masked_greater(pval, 0.05)
    
    
    return count_end_drought, count_plus_one, pval, selected_o, selected_r
    
# =============================================================================
# tc_drought function extract drought indices regional-mean
# masking values where precipitation is not associated with TCs.
# This function is applied *after* the drought metric is calculated.
# and therefore, accounts for non-TC precipitation to estimate the reference water
# balance (e.g., CAFEC in PDSI)
# =============================================================================
def tc_drought(dif_masked, selectedr, selectedo):
    '''
    Parameters
    ----------
    dif_masked : Tarray
        DESCRIPTION.
    selectedr : array
        DESCRIPTION.
    selectedo : array
        DESCRIPTION.

    Returns
    -------
    selectedots : array
        DESCRIPTION.
    selectedrts : array
        DESCRIPTION.

    '''
    dif_masked_ts = np.ma.mean(dif_masked, axis=(1, 2))
    index = np.ma.where(dif_masked_ts > 0.0)
     
    selectedr = np.ma.masked_invalid(selectedr)
    selectedr = np.ma.masked_equal(selectedr, 0.0)
    selectedrts = np.ma.mean(selectedr, axis=(1, 2))
    selectedrts = selectedrts[index]
    
    selectedo = np.ma.masked_invalid(selectedo)
    selectedo = np.ma.masked_equal(selectedo, 0.0)
    selectedots = np.ma.mean(selectedo, axis=(1, 2))
    selectedots = selectedots[index]
    
    return selectedots, selectedrts 

# =============================================================================
# TC-drought contribution stats function
# =============================================================================
def drought_stats(count, pval):
# The area of the Hurricane Region of the Americas in number of gridcells and km2
    ref_area_grid = 253998#167085 #gridcells
    count_grids = np.ma.count(count)
    pval_grids = np.ma.count(pval)
    
    # Since CHIRPS is 0.05 degrees lat/lon, then, 0.05 * 111 = ~5.55 km:
    # 5.55 * 5.55 = 30.80 km**2
    ref_area_km = ref_area_grid * 30.80
    
    # Let's calculate the percentage of the area where TCs ameliorated drought
    count_percentage = (count_grids * 100)/ref_area_grid
    count_km2 = count_grids * 30.80
    countmax = np.nanmax(count)
    countmean = np.nanmean(count)
    
    # Let's do repeat this for p-values
    pval_percentage = (pval_grids * 100)/ref_area_grid
    pval_km2 = pval_grids * 30.80
    
    return count_grids, count_percentage, count_km2, countmax, countmean, pval_percentage, pval_km2



