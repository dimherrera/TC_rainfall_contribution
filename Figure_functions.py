#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 00:50:12 2024

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
import geopandas
import cartopy
import cartopy.crs as ccrs
from cartopy.geodesic import Geodesic
import cartopy.feature as cf
from cartopy.io import shapereader
import cartopy.io.img_tiles as cimgt
from cartopy.feature import ShapelyFeature
import shapely.geometry as sgeom
from shapely.geometry import Polygon
from tropycal import tracks
import matplotlib.path as mpath
import datetime
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib import rcParams
import seaborn as sns
import matplotlib.dates as mdates
import cartopy.io.shapereader as shapereader



# =============================================================================
#  Define a single-panel plotting function
# =============================================================================
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

cmap = plt.get_cmap('terrain_r')
new_cmap = truncate_colormap(cmap, 0.2, 1.0)

#def plot_swath(masked, lons, lats, track, lont, latt, mode, title):#lon, lat, marker, date, title):'/Users/dimitrisherrera/Downloads/IRMA/2005/'
#def plot_swath(masked, lons, lats, pval, new_cmap, mode, title):#lon, lat, marker, date, title):'/Users/dimitrisherrera/Downloads/IRMA/2005/'
def plot_swath(masked, lons, lats, new_cmap, mode):#lon, lat, marker, date, title):'/Users/dimitrisherrera/Downloads/IRMA/2005/'
   
    if mode=='dark':
        plt.style.use('dark_background')
    else:
        plt.style.use('default')
    
    gd = Geodesic()
    src_crs = ccrs.PlateCarree()
    lcc = ccrs.Robinson() 
    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(111, projection=lcc)
    
    #cmap = mpl.cm.cmap
    cmap2 = mpl.cm.Greys_r
   
    boundst = list(np.arange(-10, 41, 0.2))
    bounds = list(np.arange(1.0, 15.5, 0.5))
    
    norm = mpl.colors.BoundaryNorm(bounds, new_cmap.N, extend='both')
    normt = mpl.colors.BoundaryNorm(boundst, cmap2.N, extend='both')
    
    #ax.pcolormesh(lont, latt, track, transform=ccrs.PlateCarree(), cmap='Greys_r', norm=normt)#vmin=-10, vmax=40)
    cax = ax.pcolor(lons, lats, masked, transform=ccrs.PlateCarree(), cmap=new_cmap, norm=norm)
    ax.contourf(lons, lats, pval, colors='none', transform=ccrs.PlateCarree(), hatches=[7*'/',7*'/'])
    ax.set_extent([-130.0, -45.0, 5.0, 40], crs=ccrs.PlateCarree())
    
    cbar_ticks = list(np.arange(1.0, 16.0, 2.0))
    cbar = fig.colorbar(cax, ticks=cbar_ticks, orientation='horizontal', aspect=30, extendfrac='auto', fraction=0.06, pad=0.01)
    #cbar.ax.set_xticklabels([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label('Number of months scPDSI changed > 0.99', fontsize=12, labelpad=3)

    if mode=='dark':
        ax.coastlines(resolution='50m', color='white')
        ax.add_feature(cf.BORDERS, linewidth=0.5, color='white') 
        ax.add_patch(patches.Rectangle(xy=[-125, 7], ls='--', width=75, height=30,
                                    facecolor='none', edgecolor='white',
                                    transform=ccrs.PlateCarree()))

    else:
        ax.coastlines(resolution='50m', color='black')
        ax.add_feature(cf.BORDERS, linewidth=0.5, color='black')
        ax.add_feature(cf.LAND.with_scale('50m'), facecolor='lightgrey')
        ax.add_patch(patches.Rectangle(xy=[-125, 7], ls='--', width=75, height=30,
                                    facecolor='none', edgecolor='k',
                                    transform=ccrs.PlateCarree()))

    
    #plt.title(title,fontsize=17)

    axins = inset_axes(ax, width="50%", height="60%", loc="upper right", 
                   axes_class=cartopy.mpl.geoaxes.GeoAxes, 
                   axes_kwargs=dict(map_projection=ccrs.PlateCarree()))
    axins.set_aspect('equal', anchor='NE')
    #axins.contourf(lont[1290:1440], latt[100:290], track[100:290,1290:1440], transform=ccrs.PlateCarree(), cmap='Greys_r', vmin=-10, vmax=40)
    axins.pcolor(lons[1290:1420], lats[100:280], masked[100:280,1290:1420], transform=ccrs.PlateCarree(), cmap='terrain_r', norm=norm)
    axins.contour(lons[1290:1420], lats[100:280], pval[100:280,1290:1420], colors='none', transform=ccrs.PlateCarree(), hatches=[7*'/',7*'/'])
    
    if mode=='dark':
        axins.add_feature(cf.COASTLINE,linewidth=0.8, color='white')
        ax.add_patch(patches.Rectangle(xy=[-125, 7], ls='--', width=75, height=30,
                               facecolor='none', edgecolor='white',
                               transform=ccrs.PlateCarree()))

    else:
        axins.add_feature(cf.COASTLINE,linewidth=0.8, color='black')
        ax.add_patch(patches.Rectangle(xy=[-125, 7], ls='--', width=75, height=30,
                                facecolor='none', edgecolor='k',
                                transform=ccrs.PlateCarree()))

    #plt.suptitle(date, x=0.28, y=0.22)
    #plt.text(0.14, 0.11, 'Date',
     #   verticalalignment='bottom', horizontalalignment='left',
     #   transform=ax.transAxes,
     #   color='white', fontsize=15)
     
    #plt.savefig('/Users/dimitrisherrera/Desktop/TC_contributions/Figures_TC_contributions/Figure_S5.png', dpi=400)
    #plt.savefig("Season_2005"+str(date)+".png")
   
    return(plt.show())


# =============================================================================
# # Multipanel plot
# =============================================================================
# Data: if different datasets are used, they should be concatenated before using this
# function. For example, to plot temp, precip, data.
# Example: multipanel(data, lon, lat, 2, 2, cmaps, norm_final, 20, 15, "default", titles_dict, Letters)

def multipanel(data, lons, lats, nrows, ncols, cmaps, norm, fig_width, fig_height, mode, titles_dict, letters):
    rcParams['font.family'] = 'sans-serif'
    rcParams['font.sans-serif'] = ['Tahoma']
    
    if mode=='dark':
        plt.style.use('dark_background')
    else:
        plt.style.use('default')
    
    # set number of rows/columns
    #rows, cols = 2,2
    
    gs = gridspec.GridSpec(nrows, ncols)
    
    # Set figsize using fig_width and fig_height (in inches)
    fig = plt.figure(figsize=(fig_width, fig_height))
    #fig = plt.figure(figsize=(20, 18))
    
    # cmaps is a string list with the colormaps to be used on each panel
    cmaps = tuple(cmaps)
    #letters = tuple(letters)
    
    for i in range(0, nrows*ncols):
        ax = plt.subplot(gs[i], projection = ccrs.Robinson())  
        cax = ax.pcolor(lons, lats, data[i,:,:], transform=ccrs.PlateCarree(), cmap=cmaps[i], norm=norm[i])
        # any of these projections are OK to use
        # PlateCarree() Robinson() Mercator() Orthographic()
        ax.set_extent([-125.0, -45.0, 5.0, 34])
        
        if mode=='dark':
            ax.coastlines(resolution='50m', linewidth=2.3, color='white')
            ax.add_feature(cf.BORDERS, linewidth=0.8, color='white') 
            #ax.annotate(letters[i],
             #       xy=(0, 1), xycoords='axes fraction',
            #        xytext=(+0.2, -0.2), textcoords='offset fontsize',
             #       fontsize=55, verticalalignment='top', weight='bold')
                    #bbox=dict(facecolor='0.7', edgecolor='none', pad=3.0))
            ax.add_patch(patches.Rectangle(xy=[-125, 7], ls='--', linewidth=2.3, width=75, height=30,
                                facecolor='none', edgecolor='white',
                                transform=ccrs.PlateCarree()))

        else:
            ax.coastlines(resolution='50m', linewidth=2.3, color='black')
            ax.add_feature(cf.BORDERS, linewidth=0.8, color='black')
            ax.add_feature(cf.LAND.with_scale('50m'), facecolor='lightgrey')            
            #ax.annotate(letters[i],
             #       xy=(0, 1), xycoords='axes fraction',
              #      xytext=(+0.2, -0.2), textcoords='offset fontsize',
               #     fontsize=55, verticalalignment='top', weight='bold')
                    #bbox=dict(facecolor='0.7', edgecolor='none', pad=3.0))
            ax.add_patch(patches.Rectangle(xy=[-125, 7], ls='--', linewidth=2.3, width=75, height=30,
                                    facecolor='none', edgecolor='k',
                                    transform=ccrs.PlateCarree()))
        
        cbar = fig.colorbar(cax, ax=ax, orientation='horizontal', extendfrac='auto', fraction=0.06, pad=0.02)
        cbar.ax.tick_params(labelsize=30)
        cbar.set_label(cbar_labels[i], fontsize=35, labelpad=5)
        #ax.set_title(titles_dict[i],fontsize=40)
       
        # ADD square showing the Lesser Antilles
        axins = inset_axes(ax, width="50%", height="60%", loc="upper right", 
                       axes_class=cartopy.mpl.geoaxes.GeoAxes, 
                       axes_kwargs=dict(map_projection=ccrs.PlateCarree()))
        axins.set_aspect('equal', anchor='NE')
        # For CHIRPS
        #axins.pcolor(lons[1290:1420], lats[100:280], data[i,100:280,1290:1420], transform=ccrs.PlateCarree(), cmap=cmaps[i], norm=norm[i])
        # For MSWEP
        axins.pcolor(lons[645:709], lats[210:299], data[i,210:299,645:709], transform=ccrs.PlateCarree(), cmap=cmaps[i], norm=norm[i])
        
        if mode=='dark':
            axins.add_feature(cf.COASTLINE,linewidth=1.0, color='white')
            axins.coastlines(resolution='50m', linewidth=1.0, color='white')
        else:
            axins.add_feature(cf.COASTLINE,linewidth=1.0, color='black')
            axins.coastlines(resolution='50m', linewidth=1.0, color='black')
        #ax.gridlines() # add gridlines (e.g., lat/lon lines)
    
    # Do this to get updated positions/dimensions   
    #plt.draw() 
    gs.tight_layout(fig)
    plt.savefig('/Users/dimitrisherrera/Desktop/Figure_3A-Ricardo.png', dpi=400, bbox_inches='tight')
    
    return(plt.show())

# =============================================================================

titles_dict = {
    0: 'TC seasonal contribution',             #Panel A
    1: "TC annual contribution",                #Panel B
    2: 'TC seasonal mean precipitation',           #Panel C
    3: 'TC annual mean precipitation'        #Panel D
}

cbar_labels = {
    0: 'Difference (%)',             #Panel A
    1: "Difference (%)",                #Panel B
    2: 'Difference (mm/month)',           #Panel C
    3: 'Difference (mm/month)'        #Panel D
}

Letters = {
    0: "a",                                     #Panel A
    1: "b",                                     #Panel B
    2: "c",                                     #Panel C
    3: "d"                                      #Panel D
}
#cmaps = ['rainbow', 'rainbow', 'viridis_r','viridis_r']
#cmaps = [new_cmap, new_cmap, new_cmap, new_cmap]
cmaps = ['BrBG', 'BrBG', 'BrBG','BrBG']

cmap = mpl.cm.BrBG
cmap1 = mpl.cm.viridis_r
cmap2 = mpl.cm.viridis_r

#bounds3 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]
bounds3 = [-10.0, -9.0, -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
bounds = [-5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0]

norm1 = mpl.colors.BoundaryNorm(bounds3, cmap.N, extend='both')
#norm2 = mpl.colors.BoundaryNorm(bounds3, cmap2.N, extend='both')
#norm3 = mpl.colors.BoundaryNorm(bounds3, cmap3.N, extend='both')
norm = mpl.colors.BoundaryNorm(bounds, cmap.N, extend='both')

norm_final = [norm, norm, norm1, norm1]
#norm_final = [norm1, norm1, norm2, norm2]

#letters = ['A', 'B', 'C', 'D']

# =============================================================================
# Time series 
# =============================================================================


def timeseries_plot(time_axis_data, data_one, data_two, data_three, mode):

    if mode=='dark':
        plt.style.use('dark_background')
    else:
        plt.style.use('default')
        
    x = time_axis_data
    y = data_one
    z = data_two
    contm = data_three
    yi = list(np.arange(0,11,2))
    yj = list(np.arange(0,60,20))
    
    fig, tc_ax = plt.subplots()
    fig.set_size_inches(15, 3)
    
    cont_ax = tc_ax.twinx()
    tc_ax.set_ylabel("TC Frequency", fontsize=25)
    cont_ax.set_ylabel("Contribution (%)", fontsize=25)
    
    if mode=='dark':
        #plt.style.use('dark_background')
        cont_lines = cont_ax.plot(x, contm[1:40],'white', lw=2.5, alpha=0.7)
        tc_lines = tc_ax.plot(x, y, color='white', alpha=0.3)
        h_lines = tc_ax.plot(x, z, color='white', alpha=0.4)
        
        cont_ax.fill_between(x, contm[1:40],color='white', alpha=0.7)
        tc_ax.fill_between(x, y, color='white', alpha=0.3)
        tc_ax.fill_between(x, z, color='white', alpha=0.4)
        
    else:
        #plt.style.use('default')
        cont_lines = cont_ax.plot(x, contm[1:40],'black', lw=2.5, alpha=0.7)
        tc_lines = tc_ax.plot(x, y, color='black', alpha=0.3)
        h_lines = tc_ax.plot(x, z, color='black', alpha=0.4)
        
        cont_ax.fill_between(x, contm[1:40],color='black', alpha=0.7)
        tc_ax.fill_between(x, y, color='black', alpha=0.3)
        tc_ax.fill_between(x, z, color='black', alpha=0.4)
    
    
    tc_ax.set_xlim([1986, 2023])
    tc_ax.set_ylim([0, 73])
    plt.yticks(yj)
    cont_ax.set_ylim([0, 10])
    plt.yticks(yi)
    #cont_ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    
    cont_ax.tick_params(axis='y',labelsize=27)
    tc_ax.tick_params(axis='y', labelsize=27)
    tc_ax.tick_params(axis='x', labelsize=27)
    
    
    all_lines = tc_lines + h_lines + cont_lines
    cont_ax.legend(all_lines, ["All TCs", "Hurricanes", "Contribution"],loc='upper right', ncol = 3, prop=dict(size=25))
    
    #plt.savefig('/Users/dimitrisherrera/Desktop/TC_contributions/Figures_TC_contributions/Figure_1C_dark.png', dpi=600, bbox_inches='tight')
    #plt.savefig("Season_2005"+str(date)+".png")
    return(plt.show())

# =============================================================================
# Time series single
# =============================================================================


def timeseries_plots(time_axis_data, data_one, data_two, mode):

    if mode=='dark':
        plt.style.use('dark_background')
    else:
        plt.style.use('default')
        
    x = time_axis_data
    y = data_one
    z = data_two
    
    ypd0 = pd.DataFrame(y).rolling(5).mean()
    zpd0 = pd.DataFrame(z).rolling(5).mean()
    
    ypd = np.array(ypd0)
    ypd = np.reshape(ypd, len(y))
    zpd = np.array(zpd0)
    zpd = np.reshape(zpd, len(z))
    
    miny = (ypd - np.std(ypd))
    maxy = (ypd + np.std(ypd))
    
    minz = (zpd - np.std(zpd))
    maxz = (zpd + np.std(zpd))
    
    #miny = y - np.std(y)
    #maxy = y + np.std(y)
    
    yi = list(np.arange(-1.5,3.1,1.5))
    #yi = list(np.arange(-0.7,1.7,0.5))
    #yj = list(np.arange(0,60,20))
    
    fig, ax = plt.subplots()
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(2.0)
    fig.set_size_inches(8, 6)
    
    #cont_ax = tc_ax.twinx()
    #ax.set_ylabel("TC Frequency", fontsize=25)
    #cont_ax.set_ylabel("Contribution (%)", fontsize=25)
    #ax.tick_params(left = False, right = False , labelleft = False , 
    #            labelbottom = False, bottom = False) 
    if mode=='dark':
        #total_lines = ax.plot(x[1::], y[1::],'sienna', lw=2.5, alpha=0.9, linestyle="", marker='.', markersize=15, markerfacecolor='orange')
        total_lines = ax.plot(x[1::], ypd[1::], lw=2.5, alpha=1.0, linestyle="-", color='orange')
        ax.plot(x[1::], y[1::], lw=6.5, alpha=0.4, linestyle="-", color='sienna')
      
        #ax.plot(x[1::], y[1::], lw=3.5, alpha=0.2, linestyle="-", color='sienna')
        #ax.fill_between(x[1::], miny[1::], maxy[1::], alpha=1)
        
        #tc_lines =  ax.plot(x[1::], z[1::], color='black', alpha=0.1,linestyle="", marker='.', markersize=15, markerfacecolor='sienna')
        tc_lines = ax.plot(x[1::], zpd[1::], lw=2.5, alpha=1.0, linestyle="-", color='olivedrab')
        ax.plot(x[1::], z[1::], lw=6.5, alpha=0.4, linestyle="-", color='olivedrab')
        
        ax.plot(x, np.zeros(len(z)), linestyle=(0,(5,10)), color='white', alpha=0.9)
        ax.plot(x, np.zeros(len(y)) + 0.5, linestyle=(0,(5,10)), color='white', alpha=0.9)
        ax.plot(x, np.zeros(len(y)) - 0.5, linestyle=(0,(5,10)), color='white', alpha=0.9)
     
        
    else:
        #total_lines = ax.plot(x[1::], y[1::],'sienna', lw=2.5, alpha=0.9, linestyle="", marker='.', markersize=15, markerfacecolor='orange')
        total_lines = ax.plot(x[1::], ypd[1::], lw=2.5, alpha=0.9, linestyle="-", color='sienna')
        ax.plot(x[1::], y[1::], lw=6.5, alpha=0.2, linestyle="-", color='sienna')
      
        #ax.plot(x[1::], y[1::], lw=3.5, alpha=0.2, linestyle="-", color='sienna')
        #ax.fill_between(x[1::], miny[1::], maxy[1::], alpha=1)
        
        #tc_lines =  ax.plot(x[1::], z[1::], color='black', alpha=0.1,linestyle="", marker='.', markersize=15, markerfacecolor='sienna')
        tc_lines = ax.plot(x[1::], zpd[1::], lw=2.5, alpha=0.9, linestyle="-", color='olivedrab')
        ax.plot(x[1::], z[1::], lw=6.5, alpha=0.2, linestyle="-", color='olivedrab')
        
        ax.plot(x, np.zeros(len(z)), linestyle=(0,(5,10)), color='black', alpha=0.9)
        ax.plot(x, np.zeros(len(y)) + 0.5, linestyle=(0,(5,10)), color='black', alpha=0.9)
        ax.plot(x, np.zeros(len(y)) - 0.5, linestyle=(0,(5,10)), color='black', alpha=0.9)
     
        
        #ax.fill_between(x[1::], minz[1::], maxz[1::], alpha=1)
       
        #tc_lines =  ax.plot(x[1::], z[1::], color='white', alpha=0.9,linestyle="-")#, marker='.', markersize=15, markerfacecolor='greenyellow')
        
        #total_lines = ax.plot(x, y, color='sienna', lw=2.5, alpha=0.5)
        #tc_lines = ax.plot(x, z, color='olivedrab', lw=2.5, alpha=0.7)
       
      #  ax.fill_between(x, y,color='black', alpha=0.7)
        #ax.fill_between(x, y - np.std(y), y + np.std(y), color='sienna', alpha=0.2)
        #ax.fill_between(x, z - np.std(z), ypd + np.std(z), color='olivedrab', alpha=0.2)
        #tc_ax.fill_between(x, z, color='white', alpha=0.4)
        #ds.time.values[0]
    
    #ax.fill_between(x[1::], miny, maxy, facecolor = "red", alpha=1)
    #ax.set_xlim(y.time.values[8],y.time.values[-1])
    
    ax.set_xlim(time_axis_data.values[8],time_axis_data.values[-1])
    ax.xaxis.set_major_locator(mdates.YearLocator(10, month = 1, day = 28))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.set_ylabel('SPEI Units', fontsize=30)
    ax.set_ylim([-0.6, 1.1])
    #ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    #ax.yticks(0, 1, 0.5)
    plt.yticks(yi)
    #cont_ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    
   
    ax.tick_params(axis='y', labelsize=30)
    ax.tick_params(axis='x', labelsize=30)
    
    
    all_lines = total_lines + tc_lines
    #ax.legend(all_lines, ["TC removed", "Total rainfall"],loc='upper right', ncol = 3, prop=dict(size=22))
    ax.legend(all_lines, ["SPEItc", "SPEIo"],loc='upper left', prop=dict(size=26))
    #plt.savefig('/Users/dimitrisherrera/Desktop/TC_contributions/Figures_TC_contributions/Figure_3C.png', dpi=300, bbox_inches='tight')
    #plt.savefig("Season_2005"+str(date)+".png")
    return(plt.show())

# =============================================================================
# PDF plot function 
# =============================================================================

def plot_pdf(datax, datay, fig_width, fig_height, mode):
    '''
    Parameters
    ----------
    datax : TYPE
        DESCRIPTION.
    datay : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    if mode=='dark':
        plt.style.use('dark_background')
    else:
        plt.style.use('default')
        
    fig, ax = plt.subplots(linewidth=5)
    fig.set_size_inches(fig_width, fig_height)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(2.5)
    
    # Histograms
    plt.hist(datax, bins=40, color='sienna', alpha=0.6, density= True)
    plt.hist(datay, bins=40, color='olivedrab', alpha=0.6, density= True)
             
    # Kernel dencity function
    x_lines = sns.kdeplot(datax, bw_method=0.2, linewidth = 3.5, color='saddlebrown', label="Ztc", fill=True)
    y_lines = sns.kdeplot(datay, bw_method=0.2, linewidth = 3.5, color='darkgreen', label="Zo", fill=True)

    ax.set_xlim([-1, 1])
    #ax.xaxis.set_major_locator(mdates.YearLocator(5, month = 1, day = 28))
    #ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.set_ylabel("Density", fontsize=40, labelpad=10)
    ax.set_ylim([0, 2])
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    #ax.yticks(0, 1, 0.5)
    #cont_ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    
   
    ax.tick_params(axis='y', labelsize=40)
    ax.tick_params(axis='x', labelsize=40)
    
    ax.legend(loc="upper left", prop=dict(size=35))
    #sns.move_legend(x_lines, "center right")
    #all_lines = x_lines + y_lines
    #ax.legend(x_lines, y_lines, ["Zo", "Zr"],loc='upper right', ncol = 3, prop=dict(size=23))
    #plt.savefig('/Users/dimitrisherrera/Desktop/TC_contributions/Figures_TC_contributions/Figure_4B.png', dpi=300, bbox_inches='tight')
    return (plt.show)

# =============================================================================
# Plot climatology function
# =============================================================================

def plot_clima(clim1, clim2, clim3, mode):
    time  = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    if mode=='dark':
        plt.style.use('dark_background')
    else:
        plt.style.use('default')
        
    fig, ax = plt.subplots(linewidth=5)
    fig.set_size_inches(15, 5)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(2.5)
        
    ax.fill_between(time, clim1, color='black', alpha=0.3)
    ax.fill_between(time, clim2, color='black', alpha=0.5)
    ax.fill_between(time, clim3, color='black', alpha=0.7)
  
    
    ax.plot(time, clim1, lw=3.5, alpha=0.3, linestyle="-", color='black', label="CHIRPS")
    ax.plot(time, clim2, lw=3.5, alpha=0.5, linestyle="-", color='black', label="CHIRPStc")
    ax.plot(time, clim3, lw=3.5, alpha=0.7, linestyle="-", color='black', label="TC")
   
    
    ax.set_ylim([-0.1, 249])
    ax.set_xlim([0, 11])
    #ax.xticks(np.linspace(0,13,1))
    ax.set_ylabel("mm", fontsize=40, labelpad=10)
    ax.tick_params(axis='y', labelsize=40)
    ax.tick_params(axis='x', labelsize=35)
    plt.xticks(rotation=45)
    
    
    
    ax.legend(loc="upper left", prop=dict(size=35))
    #plt.savefig('/Users/dimitrisherrera/Desktop/TC_contributions/Figures_TC_contributions/Figure_4B2.png', dpi=300, bbox_inches='tight')
  
    return (plt.show())

# =============================================================================
# Plot boxplot function
# =============================================================================

def plot_box(data1, data2, data3, data4, mode):
    
    if mode=='dark':
        plt.style.use('dark_background')
    else:
        plt.style.use('default')
    flierprops = dict(marker='o', markerfacecolor='None', markersize=14,  markeredgecolor='black')  
    
    PROPS = {
    'boxprops':{'edgecolor':'white'},
    'medianprops':{'color':'white'},
    'whiskerprops':{'color':'white'},
    'capprops':{'color':'white'}}
    
    fruit_weights = [
         data1,
         data2,
         data3,
         data4
          ] 
    colors = ['olivedrab', 'sienna', 'olivedrab', 'sienna']
    xvalues = ['SPEIo', 'SPEItc', 'Zo', 'Ztc']
    
    fig, ax = plt.subplots(linewidth=5)
    fig.set_size_inches(15, 10)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(3.0)
        
    if mode=='dark':
        plt.axhline(0.0, linestyle=(0,(5,10)), c='white', alpha=1, zorder=0)
        sns.boxplot(data=fruit_weights, palette=colors,  linewidth=4, whis=[1, 99], flierprops=flierprops, **PROPS) # will be used to label x-ticks
        
    else:
        plt.axhline(0.0, linestyle=(0,(5,10)), c='black', alpha=1, zorder=0)
        sns.boxplot(data=fruit_weights, palette=colors,  linewidth=4, whis=[1, 99], flierprops=flierprops, boxprops=dict(alpha=0.8)) # will be used to label x-ticks
        
    ax.set_ylabel('$\\sigma$', fontsize=50)
    ax.tick_params(axis='y', labelsize=48)
    ax.tick_params(axis='x', labelsize=48)
    #plt.xticks(rotation=45)
    plt.xticks(np.arange(4), xvalues)
    # fill with colors
    #for patch, color in zip(bplot['boxes'], colors):
    #    patch.set_facecolor(color)
    #plt.savefig('/Users/dimitrisherrera/Desktop/TC_contributions/Figures_TC_contributions/Figure_4C-dark.png', dpi=300, bbox_inches='tight')

    return plt.show()

# =============================================================================
# Terrarin background function
# =============================================================================
# Since Stadia is now providing background maps/images:
    
class StadiaStamen(cimgt.Stamen):
    
    def _image_url(self, tile):
         x,y,z = tile
         url = f"https://tiles.stadiamaps.com/tiles/stamen_terrain_background/{z}/{x}/{y}.png?api_key=d3ce896e-23c3-43d2-9019-6fe95c2011ef"
         return url

def rect_from_bound(xmin, xmax, ymin, ymax):
    """Returns list of (x,y)'s for a rectangle"""
    xs = [xmax, xmin, xmin, xmax, xmax]
    ys = [ymax, ymax, ymin, ymin, ymax]
    return [(x, y) for x, y in zip(xs, ys)]

def terrain_back(country):
    # request data for use by geopandas
    resolution = '10m'
    category = 'cultural'
    name = 'admin_0_countries'
    
    shpfilename = shapereader.natural_earth(resolution, category, name)
    df = geopandas.read_file(shpfilename)
    
    # get geometry of a country
    poly = [df.loc[df['ADMIN'] == country]['geometry'].values[0]]
    
    stamen_terrain = StadiaStamen("terrain-background")
    
    # projections that involved
    st_proj = stamen_terrain.crs  #projection used by Stamen images
    ll_proj = ccrs.PlateCarree()  #CRS for raw long/lat
    
    # create fig and axes using intended projection
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(1, 1, 1, projection=st_proj)
    #ax.add_geometries(poly, crs=ll_proj, facecolor='none', edgecolor='black')
    
    pad1 = .1  #padding, degrees unit
    exts = [poly[0].bounds[0] - pad1, poly[0].bounds[2] + pad1, poly[0].bounds[1] - pad1, poly[0].bounds[3] + pad1];
    ax.set_extent(exts, crs=ll_proj)
    
    # make a mask polygon by polygon's difference operation
    # base polygon is a rectangle, another polygon is simplified switzerland
    msk = Polygon(rect_from_bound(*exts)).difference( poly[0].simplify(0.01) )
    msk_stm  = st_proj.project_geometry (msk, ll_proj)  # project geometry to the projection used by stamen
    
    # get and plot Stamen images
    ax.add_image(stamen_terrain, 8) # this requests image, and plot
    
    # plot the mask using semi-transparency (alpha=0.65) on the masked-out portion
    ax.add_geometries( msk_stm, st_proj, zorder=12, facecolor='white', edgecolor='none', alpha=0.65)
    ax.add_geometries(poly, crs=ll_proj, facecolor='none', edgecolor='black')
    
    ax.gridlines(draw_labels=True)
    # Valle Nuevo samples (Speer et al. 2004)
    Speer = plt.plot(-70.638611, 18.778, \
             color='black', linestyle='None', marker='^', markersize=8, label='Speer et al. (2004)', transform=ccrs.PlateCarree())
    plt.plot(-70.64, 18.778, \
             color='black', linewidth=0.0, marker='^', markersize=8, transform=ccrs.PlateCarree())
    plt.plot(-70.69, 18.81, \
             color='black', linewidth=0.0, marker='^', markersize=8, transform=ccrs.PlateCarree())
    # Pico Duarte samples (Speer et al. 2004)
    plt.plot(-71.005, 19.035, \
         color='black', linewidth=0, marker='^', markersize=8, transform=ccrs.PlateCarree())
          
    # Cornell/UA samples
    Sampled = plt.plot(-70.695, 18.82, \
             color='red', linestyle='None', marker='^', markersize=8, label='Sampled in 2019', transform=ccrs.PlateCarree())
    plt.plot(-70.60, 18.74, \
             color='red', linewidth=0, marker='^', markersize=8, transform=ccrs.PlateCarree())   
       
    # Proposed samples
    # Cordillera Central
    Proposed = plt.plot(-71.22, 19.11, \
             color='red', linestyle='None', marker='X', markersize=8, label='Proposed sampling', transform=ccrs.PlateCarree())
   
    
    # Sierra de Bahoruco
    plt.plot(-71.666, 18.28, \
             color='red', linewidth=0, marker='X', markersize=8, transform=ccrs.PlateCarree())
    plt.plot(-71.45, 18.158, \
             color='red', linewidth=0, marker='X', markersize=8, transform=ccrs.PlateCarree())
    
        # Sierra de Neyba
    plt.plot(-71.558, 18.654, \
             color='red', linewidth=0, marker='X', markersize=8, transform=ccrs.PlateCarree())
    plt.plot(-71.483, 18.63, \
              color='red', linewidth=0, marker='X', markersize=8, transform=ccrs.PlateCarree())
    plt.plot(-71.77, 18.687, \
              color='red', linewidth=0, marker='X', markersize=8, transform=ccrs.PlateCarree())
           
    plt.legend(loc = "lower right").set_zorder(100)
    #plt.savefig('/Users/dimitrisherrera/Desktop/Figure_XX_P4Clim.png', dpi=400, bbox_inches='tight')

    
    return (plt.show())

# =============================================================================
# Highlight a country
# =============================================================================

def highlight(countries):
    proj = ccrs.PlateCarree()
    ax = plt.axes(projection=proj)
    # Show only Africa
    ax.set_extent([-100.0, -55.0, 5.0, 36], crs=ccrs.PlateCarree())
    #ax.stock_img()

    ax.add_feature(cf.COASTLINE, lw=2)
    ax.add_feature(cf.BORDERS, linewidth=1.5, color='black')
    ax.add_feature(cf.LAND.with_scale('50m'), facecolor='lightgrey')
    # Make figure larger
    plt.gcf().set_size_inches(20, 10)

    # Read Natural Earth data

    shpfilename = shpreader.natural_earth(resolution='110m',
                                          category='cultural',
                                          name='admin_0_countries')
    reader = shpreader.Reader(shpfilename)
    kenya = [country for country in reader.records() if country.attributes["NAME_LONG"] == countries][0]

    # Display Kenya's shape
    shape_feature = ShapelyFeature([kenya.geometry], ccrs.PlateCarree(), facecolor="black", edgecolor='black', lw=1)
    ax.add_feature(shape_feature)
    
   # plt.savefig('/Users/dimitrisherrera/Desktop/Figure_XXmini_P4Clim.png', dpi=400, bbox_inches='tight')

    return (plt.show())
    

#fig = plt.figure(figsize=(7,7))
#ax = fig.add_subplot(111, projection=lcc)
#ax.contourf(lon[1290:1420], lat[100:280], pval[100:280,1290:1420], transform=ccrs.PlateCarree(), colors='none',levels=[.5,1.5], hatches=[7*'/',7*'/'])
             
#fig = plt.figure(figsize=(7,7))
#ax = fig.add_subplot(111, projection=lcc)
#ax.contourf(lons[645:709], lats[50:140], concatenated[0,50:140,645:709], transform=ccrs.PlateCarree(), cmap='rainbow')
#ax.coastlines(resolution='50m', linewidth=1.2, color='black')

