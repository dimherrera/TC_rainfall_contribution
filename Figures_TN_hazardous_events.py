#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 22:42:42 2024

@author: dimitrisherrera
"""
# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.offsetbox import AnchoredText
import cartopy.io.img_tiles as cimgt


import matplotlib.colors
import seaborn as sns
import geopandas as gpd
import matplotlib.colors as colors
from osgeo import gdal
from osgeo import osr
import cartopy
import cartopy.io.shapereader as shapereader
import geopandas
from shapely.geometry import Polygon

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
        cont_lines = cont_ax.plot(x, contm,'white', lw=2.5, alpha=0.7)
        tc_lines = tc_ax.plot(x, y, color='white', alpha=0.3)
        h_lines = tc_ax.plot(x, z, color='white', alpha=0.4)
        
        cont_ax.fill_between(x, contm,color='white', alpha=0.7)
        tc_ax.fill_between(x, y, color='white', alpha=0.3)
        tc_ax.fill_between(x, z, color='white', alpha=0.4)
        
    else:
        #plt.style.use('default')
        cont_lines = cont_ax.plot(x, contm,'black', lw=2.5, alpha=0.7)
        tc_lines = tc_ax.plot(x, y, color='black', alpha=0.3)
        h_lines = tc_ax.plot(x, z, color='black', alpha=0.4)
        
        cont_ax.fill_between(x, contm,color='black', alpha=0.7)
        tc_ax.fill_between(x, y, color='black', alpha=0.3)
        tc_ax.fill_between(x, z, color='black', alpha=0.4)
    
    
    tc_ax.set_xlim([1996, 2022])
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

class FixPointNormalize(matplotlib.colors.Normalize):
    """ 
    Inspired by https://stackoverflow.com/questions/20144529/shifted-colorbar-matplotlib
    Subclassing Normalize to obtain a colormap with a fixpoint 
    somewhere in the middle of the colormap.

    This may be useful for a `terrain` map, to set the "sea level" 
    to a color in the blue/turquise range. 
    """
    def __init__(self, vmin=None, vmax=None, sealevel=0, col_val = 0.21875, clip=False):
        # sealevel is the fix point of the colormap (in data units)
        self.sealevel = sealevel
        # col_val is the color value in the range [0,1] that should represent the sealevel.
        self.col_val = col_val
        matplotlib.colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.sealevel, self.vmax], [0, self.col_val, 1]
        return np.ma.masked_array(np.interp(value, x, y))

# Combine the lower and upper range of the terrain colormap with a gap in the middle
# to let the coastline appear more prominently.
# inspired by https://stackoverflow.com/questions/31051488/combining-two-matplotlib-colormaps
colors_undersea = plt.cm.terrain(np.linspace(0, 0.17, 56))
colors_land = plt.cm.terrain(np.linspace(0.25, 1, 200))
# combine them and build a new colormap
colors = np.vstack((colors_undersea, colors_land))
cut_terrain_map = matplotlib.colors.LinearSegmentedColormap.from_list('cut_terrain', colors)



def hillshade(array, azimuth, angle_altitude):

    # Source: http://geoexamples.blogspot.com.br/2014/03/shaded-relief-images-using-gdal-python.html

    x, y = np.gradient(array)
    slope = np.pi/2. - np.arctan(np.sqrt(x*x + y*y))
    aspect = np.arctan2(-x, y)
    azimuthrad = azimuth*np.pi / 180.
    altituderad = angle_altitude*np.pi / 180.


    shaded = np.sin(altituderad) * np.sin(slope) \
     + np.cos(altituderad) * np.cos(slope) \
     * np.cos(azimuthrad - aspect)
    return 255*(shaded + 1)/2

def read_geotiff(imagefile):
    """
    Read an image and compute the coordinates from a geoTIFF file
    """
    ds = gdal.Open(imagefile, gdal.GA_ReadOnly)
    ds.GetProjectionRef()

    # Read the array and the transformation
    arr = ds.ReadAsArray()
    # Read the geo transform
    trans = ds.GetGeoTransform()
    # Compute the spatial extent
    extent = (trans[0], trans[0] + ds.RasterXSize*trans[1],
              trans[3] + ds.RasterYSize*trans[5], trans[3])

    # Get the info on the projection
    proj = ds.GetProjection()
    inproj = osr.SpatialReference()
    inproj.ImportFromWkt(proj)

    # Compute the coordinates
    x = np.arange(0, ds.RasterXSize)
    y = np.arange(0, ds.RasterYSize)

    xx, yy = np.meshgrid(x, y)
    lon = trans[1] * xx + trans[2] * yy + trans[0]
    lat = trans[4] * xx + trans[5] * yy + trans[3]

    # Transpose
    arr = np.transpose(arr, (0, 1))

    return lon, lat, arr, inproj, extent

def rect_from_bound(xmin, xmax, ymin, ymax):
    """Returns list of (x,y)'s for a rectangle"""
    xs = [xmax, xmin, xmin, xmax, xmax]
    ys = [ymax, ymax, ymin, ymin, ymax]
    return [(x, y) for x, y in zip(xs, ys)]

# Load data
df = pd.read_csv('/Users/dimitrisherrera/Downloads/Thesis Publication Data/TNHazardandHarmfulHazard.csv', skipinitialspace=True, usecols=['YEAR', 'EVENT_TYPE', 'IMPACTS'])
df = df.set_index('YEAR')
df.index = pd.to_datetime(df.index, format='%Y')

# Flash flood events
filter_flash = df['EVENT_TYPE']=='Flash Flood'
flash = df[filter_flash]
flash = pd.DataFrame(flash).sort_index()
#filter_flash_impact = flash['IMPACTS']==1
#flash = flash[filter_flash_impact]
flashsum = flash['IMPACTS'].resample('1Y').sum()
flashsum.plot(color='blue')

# Tornadoes
filter_tor = df['EVENT_TYPE']=='Tornado'
tor = df[filter_tor]
tor = pd.DataFrame(tor).sort_index()
#filter_tor_impact = tor['IMPACTS']==1
#tor = tor[filter_tor_impact]
torsum = tor['IMPACTS'].resample('1Y').sum()
torsum.plot(color='red')

# Thunderstorm wind
filter_thur = df['EVENT_TYPE']=='Thunderstorm Wind'
thunder = df[filter_thur]
thunder = pd.DataFrame(thunder).sort_index()
#filter_thur_impact = thunder['IMPACTS']==1
#thunder = thunder[filter_thur_impact]
thundersum = thunder['IMPACTS'].resample('1Y').sum()
thundersum.plot(color='k')

# An easier way to plot bars
# Load data
df0 = pd.read_csv('/Users/dimitrisherrera/Downloads/Thesis Publication Data/TNHazardandHarmfulHazard.csv', skipinitialspace=True, usecols=['YEAR', 'EVENT_TYPE', 'IMPACTS'])

df0['YEAR'] = pd.to_datetime(df0['YEAR'], format='%Y')
df0.insert(3, "DUMMY", np.ones(19100))

# Group all harzardous event types:
#dfg = df0.groupby([df0.YEAR.dt.date, 'EVENT_TYPE'])['DUMMY'].sum().reset_index()

# Group all harmful event types:
dfg = df0.groupby([df0.YEAR.dt.date, 'EVENT_TYPE'])['IMPACTS'].sum().reset_index()
dfg['YEAR'] =  pd.DatetimeIndex(dfg['YEAR']).year

dfp = dfg.pivot(index='YEAR', columns='EVENT_TYPE', values='IMPACTS')

# Plotting as stacked bars
ax = dfp.plot.bar(stacked=True, color={"Tornado": "red", "Flash Flood": "orange", "Thunderstorm Wind": "royalblue"}, xlabel='', figsize=(10, 6), rot=0)#, title='Sum of Daily Category Hours')
plt.xticks(fontsize=18)  # for xticks
plt.yticks(fontsize=18)
ax.xaxis.set_major_locator(ticker.MultipleLocator(6))
ax.set_ylabel('Overall Harmful Events', fontsize=19)
#ax.set_ylim([0, 1500])
plt.legend(bbox_to_anchor=(0.01, 1), loc='upper left', prop=dict(size=18))
plt.show()


def main():
    lon2, lat2, arr2, inproj2, extent2 = read_geotiff('/Users/dimitrisherrera/Downloads/Tennessee_DEM.tif')

    lat = lat2[:,0]

    lon = lon2[0,:]
    
    reader = shapereader.Reader('/Users/dimitrisherrera/Downloads/countyl010g_shp_nt00964/countyl010g.shp')

    counties = list(reader.geometries())

    COUNTIES = cfeature.ShapelyFeature(counties, ccrs.PlateCarree())
    
    resolution = '10m'
    
    category = 'cultural'
    
    name = 'admin_1_states_provinces'

    shpfilename = shapereader.natural_earth(resolution, category, name)
    
    df = geopandas.read_file(shpfilename)
    poly = [df.loc[df['adm1_code'] == 'USA-3551']['geometry'].values[0]]
    
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    #ax.set_extent([-91, -81, 34.5, 37], crs=ccrs.PlateCarree())
    st_proj = inproj2  #projection used by Stamen images
    ll_proj = ccrs.PlateCarree()  #CRS for raw long/lat
    
    # Put a background image on for nice sea rendering.
    #ax.stock_img()
    pad1 = .1  #padding, degrees unit
    exts = [poly[0].bounds[0] - pad1, poly[0].bounds[2] + pad1, poly[0].bounds[1] - pad1, poly[0].bounds[3] + pad1];
    # Create a feature for States/Admin 1 regions at 1:50m from Natural Earth
    states_provinces = cfeature.NaturalEarthFeature(
        category='cultural',
        name='admin_1_states_provinces_lines',
        scale='50m',
        facecolor='none')
    #stamen_terrain = StadiaStamen("terrain-background")
    SOURCE = 'Natural Earth'
    LICENSE = 'public domain'
    cax = ax.contourf(lon2[0:2800, 5000:17000], lat2[0:2800, 5000:17000], arr2[0:2800, 5000:17000], np.arange(-5, 500, 5),extent=extent2, cmap='terrain', vmin=0, vmax=500, origin='image')
    
    #ax.add_image(stamen_terrain, 8)
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(states_provinces, edgecolor='black')

    ax.add_feature(COUNTIES, facecolor='none', edgecolor='gray')
    ax.add_feature(states_provinces, edgecolor='black')
    #msk = Polygon(rect_from_bound(*exts)).difference( poly[0].simplify(0.01) )
    #msk_stm  = st_proj.project_geometry (msk, ll_proj)  # project geometry to the projection used by stamen
    
    # get and plot Stamen images
    #ax.add_image(stamen_terrain, 8) # this requests image, and plot
    
    # plot the mask using semi-transparency (alpha=0.65) on the masked-out portion
    #ax.add_geometries( msk_stm, st_proj, zorder=12, facecolor='white', edgecolor='none', alpha=0.65)
    ax.add_geometries(poly, crs=ll_proj, facecolor='none', edgecolor='black')
    ax.gridlines(draw_labels=True)

    # Read Natural Earth data
    
    shpfilename = shpreader.natural_earth(resolution='110m',
                                          category='cultural',
                                          name='admin_0_countries')
    reader = shpreader.Reader(shpfilename)
    kenya = [country for country in reader.records() if country.attributes["NAME_LONG"] == countries][0]
    
    # Display Kenya's shape
    shape_feature = ShapelyFeature([kenya.geometry], ccrs.PlateCarree(), facecolor="black", edgecolor='black', lw=1)
    ax.add_feature(shape_feature)


    plt.show()


if __name__ == '__main__':
    main()
    
    
    
    
    
    
    '''
    lon2, lat2, arr2, inproj2, extent2 = read_geotiff('/Users/dimitrisherrera/Downloads/Tennessee_DEM.tif')

    lat = lat2[:,0]

    lon = lon2[0,:]

    myproj = ccrs.PlateCarree()
    plt.figure(figsize=(8, 8))
    ax = plt.axes(projection=myproj)
    #ax.imshow(arr2, origin='upper', extent=extent2, transform=myproj)
    #cax = ax.contourf(lon2[1500:5500, 10000:13000], lat2[1500:5500, 10000:13000], arr2[1500:5500, 10000:13000], np.arange(0, 500, 10),extent=extent2, cmap='terrain', vmin=0, vmax=500, origin='image')

    # For TN
    cax = ax.contourf(lon2[0:2800, 5000:17000], lat2[0:2800, 5000:17000], arr2[0:2800, 5000:17000], np.arange(0, 500, 10),extent=extent2, cmap='terrain', vmin=0, vmax=500, origin='image')
    ax.coastlines(resolution='10m', color="black")
    #ax.add_feature(cfeature.NaturalEarthFeature('physical', 'ocean', '10m', edgecolor='face', facecolor='water'))
    ax.add_feature(cfeature.OCEAN,facecolor=cfeature.COLORS['water'])
    ax.add_feature(cfeature.LAKES.with_scale('10m'), facecolor=cfeature.COLORS['water'])
    ax.add_feature(cfeature.COASTLINE.with_scale('10m'), linewidth=2.5,facecolor=cfeature.COLORS['water'])#color="black")
    #ax.add_feature(cfeature.RIVERS.with_scale('10m'), linewidth=3.5,facecolor=cfeature.COLORS['water'])# color="blue")
    ax.add_feature(COUNTIES, facecolor='none', edgecolor='gray')

    states_provinces = cfeature.NaturalEarthFeature(
        category='cultural',
        name='admin_1_states_provinces_lines',
        scale='50m',
        facecolor='none')

    SOURCE = 'Natural Earth'
    LICENSE = 'public domain'

    ax.add_feature(states_provinces, edgecolor='black')
    plt.legend()'''
    
    
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 17:31:51 2024

@author: dimitrisherrera
"""

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import shapely.geometry as sgeom
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
import cartopy.feature as cfeature
from matplotlib.offsetbox import AnchoredText
import cartopy.io.img_tiles as cimgt
import matplotlib.colors
import seaborn as sns
import geopandas as gpd
import matplotlib.colors as colors
from osgeo import gdal
from osgeo import osr
import cartopy
import cartopy.io.shapereader as shapereader
import geopandas
from shapely.geometry import Polygon
from matplotlib.cm import ScalarMappable

'''def read_geotiff(imagefile):
    """
    Read an image and compute the coordinates from a geoTIFF file
    """
    ds = gdal.Open(imagefile, gdal.GA_ReadOnly)
    ds.GetProjectionRef()

    # Read the array and the transformation
    arr = ds.ReadAsArray()
    arr = np.where(arr < 0, 0, arr)
    # Read the geo transform
    trans = ds.GetGeoTransform()
    # Compute the spatial extent
    extent = (trans[0], trans[0] + ds.RasterXSize*trans[1],
              trans[3] + ds.RasterYSize*trans[5], trans[3])

    # Get the info on the projection
    proj = ds.GetProjection()
    inproj = osr.SpatialReference()
    inproj.ImportFromWkt(proj)

    # Compute the coordinates
    x = np.arange(0, ds.RasterXSize)
    y = np.arange(0, ds.RasterYSize)

    xx, yy = np.meshgrid(x, y)
    lon = trans[1] * xx + trans[2] * yy + trans[0]
    lat = trans[4] * xx + trans[5] * yy + trans[3]

    # Transpose
    arr = np.transpose(arr, (0, 1))

    return lon, lat, arr, inproj, extent

def rect_from_bound(xmin, xmax, ymin, ymax):
    """Returns list of (x,y)'s for a rectangle"""
    xs = [xmax, xmin, xmin, xmax, xmax]
    ys = [ymax, ymax, ymin, ymin, ymax]
    return [(x, y) for x, y in zip(xs, ys)]

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

def shp_to_df(shp_filepath):
  data = {}
  shp = read_shp(shp_filepath)
  fields = shp.fields[1:] # Skip 'DeletionFlag'
  records = shp.records()

  for f in fields:
    data[f[0]] = []

  for record in records:
    for k, v in zip(fields, record):
      if type(v) == bytes:
        v = v.decode('ISO-8859-1') # or 'UTF-8'
      data[k[0]].append(v)

  return pd.DataFrame(data)
# derived from this example: https://scitools.org.uk/cartopy/docs/v0.15/examples/hurricane_katrina.html

def plot_states(df,projection,colors,annotation,title,edgecolor):
    fig = plt.figure(figsize=(13,7))
    cmap = plt.get_cmap('terrain')
    
    new_cmap = truncate_colormap(cmap, 0.1, 0.9)
    
    ax = plt.axes([0, 0, 1, 1],
                  projection=projection)
   
    #ax.set_extent([-90.5, -81, 35, 37], crs=ccrs.PlateCarree())
    
    lon2, lat2, arr2, inproj2, extent2 = read_geotiff('/Users/dimitrisherrera/Downloads/Tennessee_DEM.tif')

    lat = lat2[:,0]

    lon = lon2[0,:]
    
    reader_c = shapereader.Reader('/Users/dimitrisherrera/Downloads/countyl010g_shp_nt00964/countyl010g.shp')

    counties = list(reader_c.geometries())

    COUNTIES = cfeature.ShapelyFeature(counties, ccrs.PlateCarree())
    
    
    shpfilename = '/Users/dimitrisherrera/Downloads/cb_2018_us_state_5m/cb_2018_us_state_5m.shp'
    reader = shpreader.Reader(shpfilename)
    states = reader.records()
    values = list(df[title].unique())
    states_provinces = cfeature.NaturalEarthFeature(
        category='cultural',
        name='admin_1_states_provinces_lines',
        scale='10m',
        facecolor='none')
    #stamen_terrain = StadiaStamen("terrain-background")
    SOURCE = 'Natural Earth'
    LICENSE = 'public domain'

    for state in states:
        attribute = 'NAME'
        name = state.attributes[attribute]

        # get classification
        classification = df.loc[state.attributes[attribute]][title]

        ax.add_geometries(state.geometry, ccrs.PlateCarree(),
                          facecolor=(colors[values.index(classification)]),
                          label=state.attributes[attribute], alpha=0.8,
                          edgecolor='black',
                          linewidth=0.7)
        
    # legend
    #import matplotlib.patches as mpatches
    ax.add_feature(COUNTIES, facecolor='none', linewidth=0.5, edgecolor='gray')
    #handles = []
    #for i in range(len(values)):
   #     handles.append(mpatches.Rectangle((0, 0), 1, 1, facecolor='grey'))
   #     plt.legend(handles, values,
       #            loc='lower left', bbox_to_anchor=(0.025, -0.0), 
    #               fancybox=True, frameon=False, fontsize=5)

    # annotate
    cax = ax.contourf(lon2[0:2800, 5000:17000], lat2[0:2800, 5000:17000], arr2[0:2800, 5000:17000], np.arange(0, 1000, 10),extent=extent2, cmap=new_cmap, vmin=0, vmax=1000, origin='image')
    #ax.add_feature(states_provinces, edgecolor='black')
    #ax.annotate(annotation, xy=(0, 0),  xycoords='figure fraction',
    #        xytext=(0.0275, -0.025), textcoords='axes fraction',
    #        horizontalalignment='left', verticalalignment='center', fontsize=4,
    #        )
    cbar_ticks = list(np.arange(200, 1000, 200))
    cbar = plt.colorbar(cax, ticks=cbar_ticks, orientation='vertical', aspect=11.5, extendfrac='auto', extend='both', fraction=0.02, pad=0.01)
    #cbar.ax.set_xticklabels([0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])
    cbar.ax.tick_params(labelsize=17)
    cbar.set_label('Elevation (MASL)', fontsize=18, labelpad=9)

    #ax.gridlines(draw_labels=True)
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=2, color='None', alpha=0.5, linestyle='--')
    
    gl.xlocator = ticker.FixedLocator([-90, -87.5, -85, -82.5, -80.0])
    gl.ylocator = ticker.FixedLocator([35, 36, 37])
    gl.xlabel_style = {'size': 17, 'color': 'black'}
    gl.ylabel_style = {'size': 17, 'color': 'black'}
    #gls.top_labels=False   # suppress top labels
    gl.right_labels=False # suppress right labels
    #plt.title(title, fontsize=8)
    
    
    title = title+'.png'
    plt.savefig(title, bbox_inches='tight', pad_inches=.2, dpi=300)
    print('Saved: {}'.format(title))
    



    
def main():
    # df = pd.read_csv('states.csv')
    df = pd.read_csv('/Users/dimitrisherrera/Desktop/states.csv', index_col='State')

    # States Visited
    projection=ccrs.PlateCarree()
    title = 'States Visited'
    #colors = ['#71a2d6','#DDDDDD']
    #colors = ['None','#DDDDDD']
    colors = ['#DDDDDD', 'None']
    #colors = 'blue'
    annotation = ''
    plot_states(df,projection,colors,annotation,title,edgecolor='white')

    # 13 Original Colonies
    #projection = ccrs.LambertConformal()
    #title = '13 Original Colonies'
    #colors = ['#DDDDDD','#71a2d6']
    #annotation = ''
    #plot_states(df,projection,colors,annotation,title,edgecolor='white')

    print('Done.\n')


if __name__ == '__main__':
    main()'''
    
    
    
# Plotting counties with data map 
    
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
import matplotlib.pyplot as plt
from metpy.cbook import get_test_data

df = pd.read_csv('/Users/dimitrisherrera/Desktop/states.csv', index_col='State')

# Population TN
tnp = pd.read_csv('/Users/dimitrisherrera/Downloads/Thesis Publication Data/Tennessee_county.csv', usecols=["STCNTY", "E_TOTPOP"])
tnp['frac'] = tnp['E_TOTPOP']/100000
tnp = tnp.drop('E_TOTPOP', axis=1)
tnp.update(tnp[['STCNTY']].applymap('{}'.format))

# No. of hazardous events
tn = pd.read_csv('/Users/dimitrisherrera/Downloads/Thesis Publication Data/Thesis_ArcGISPro_Table.csv', usecols=["GEOID","Flash_Flood","Tornado","Thunderstorm_Wind" ])
tn['sum'] = tn["Flash_Flood"] + tn["Tornado"] + tn["Thunderstorm_Wind"]
tn = tn.drop(tn.columns[[1, 2, 3]], axis=1)
tn = tn.fillna(0.0)
tnn = tn['sum']/tnp['frac']
tnn2 = pd.concat([tn['GEOID'], tnn], axis=1)
tnn2.update(tnn2[['GEOID']].applymap('{}'.format))

TN = dict(tnn2.values)
title = 'States Visited'
colors = ['#DDDDDD', 'None']


# Open MetPy's counties shapefile using Cartopy's shape reader
counties = shpreader.Reader('/Users/dimitrisherrera/Downloads/cb_2018_us_county_5m/cb_2018_us_county_5m.shp')

# Loop over the records in the shapefile and generate a dictionary that
# maps geometry->tornado count
county_lookup = {rec.geometry: TN.get(rec.attributes['GEOID'], 0)
                 for rec in counties.records() if rec.attributes['STATEFP'] == '47'}

# Create a "styler" that returns the appropriate draw colors, etc.
# for a given geometry. This uses our lookup to find the counties
# that need coloring
def color_torns(geom):
    count = county_lookup.get(geom, 0)
    if count:
        cmap = plt.get_cmap('Oranges')
        norm = plt.Normalize(0, 2000)
        facecolor = cmap(norm(count))
    else:
        facecolor = 'none'
    return {'edgecolor': 'black', 'facecolor': facecolor}

legend_kwds= {
  'loc': 'bottom right',
  'bbox_to_anchor': (0.8, 0.9),
  'fmt': '{:<5.0f}',
  'frameon': False,
  'fontsize': 8,
  'title': 'Population'
}
classification_kwds={
  'bins':[1,10,25,50,100, 250, 500, 1000, 5000]
}

# Create figure with some maps
fig = plt.figure(figsize=(17, 7))
ax = fig.add_subplot(projection=ccrs.PlateCarree())
ax.add_feature(cfeature.COASTLINE)
#ax.add_feature(cfeature.STATES, linewidth=0.7)

shpfilename = '/Users/dimitrisherrera/Downloads/cb_2018_us_state_5m/cb_2018_us_state_5m.shp'
reader = shpreader.Reader(shpfilename)
states = reader.records()
values = list(df[title].unique())
states_provinces = cfeature.NaturalEarthFeature(
    category='cultural',
    name='admin_1_states_provinces_lines',
    scale='10m',
    facecolor='none')
#stamen_terrain = StadiaStamen("terrain-background")
SOURCE = 'Natural Earth'
LICENSE = 'public domain'

for state in states:
    attribute = 'NAME'
    name = state.attributes[attribute]

    # get classification
    classification = df.loc[state.attributes[attribute]][title]

    ax.add_geometries(state.geometry, ccrs.PlateCarree(),
                      facecolor=(colors[values.index(classification)]),
                      label=state.attributes[attribute], alpha=0.8,
                      edgecolor='black',
                      linewidth=0.3)
# Add the geometries from the shapefile, pass our styler function
cax = ax.add_geometries(counties.geometries(), ccrs.PlateCarree(), styler=color_torns, linewidth=0.3, alpha=0.7)
ax.set_extent((-90.8, -81, 34.65, 37.003))
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
              linewidth=2, color='None', alpha=0.5, linestyle='--')
gl.xlocator = ticker.FixedLocator([-90, -87.5, -85, -82.5, -80.0])
gl.ylocator = ticker.FixedLocator([35, 36, 37])
gl.xlabel_style = {'size': 17, 'color': 'black'}
gl.ylabel_style = {'size': 17, 'color': 'black'}
#gls.top_labels=False   # suppress top labels
gl.right_labels=False # suppress right labels

norm = matplotlib.colors.Normalize(vmin=0, vmax=2000)

sm = matplotlib.cm.ScalarMappable(cmap='Oranges', norm=norm)
cbar = plt.colorbar(sm, ticks=np.arange(400,2000,400), ax=ax, orientation='vertical', aspect=11.5, extendfrac='auto', extend='both', fraction=0.02, pad=0.01)
cbar.ax.tick_params(labelsize=17)
cbar.set_label('No. Hazards/100000 people', fontsize=18, labelpad=9)

plt.savefig('Fig_5_panel_A', bbox_inches='tight', pad_inches=.2, dpi=300)
#plt.title(title, fontsize=8)



